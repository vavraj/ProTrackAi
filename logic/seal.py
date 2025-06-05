import sys
import cv2
import mediapipe as mp
import time
from collections import deque
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to read ROIs from file
def read_rois_from_file(file_path):
    rois = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label, coords = line.split(":")
            x1, y1, x2, y2 = map(int, coords.strip().split())
            rois[label.strip()] = ((x1, y1), (x2, y2))
    return rois

class HandTracker:
    def __init__(self, history_length=15):
        self.history_length = history_length
        self.left_hand_history = deque(maxlen=history_length)
        self.right_hand_history = deque(maxlen=history_length)
        self.current_left_hand = None
        self.current_right_hand = None

    def update(self, hands):
        left_hand = None
        right_hand = None

        for hand in hands:
            if hand[2] == "Left":
                left_hand = hand
            else:
                right_hand = hand

        self.left_hand_history.append(left_hand)
        self.right_hand_history.append(right_hand)

        self.current_left_hand = self.get_stable_hand(self.left_hand_history)
        self.current_right_hand = self.get_stable_hand(self.right_hand_history)

    def get_stable_hand(self, hand_history):
        valid_hands = [hand for hand in hand_history if hand is not None]
        if len(valid_hands) < self.history_length // 2:
            return None
        return valid_hands[-1]

    def get_current_hands(self):
        return [hand for hand in [self.current_left_hand, self.current_right_hand] if hand is not None]

class ProcessTracker:
    def __init__(self, time_limit):
        self.time_limit = time_limit
        self.start_time = None
        self.end_time = None
        self.completed = False
        self.hand_side = None

    def start(self, current_time, hand_side):
        if not self.start_time:
            self.start_time = current_time
            self.hand_side = hand_side

    def end(self, current_time):
        if self.start_time and not self.end_time:
            self.end_time = current_time

    def update(self, current_time):
        if self.start_time and self.end_time and not self.completed:
            if (current_time - self.end_time) <= self.time_limit:
                self.completed = True
        return self.completed

    def reset(self):
        self.start_time = None
        self.end_time = None
        self.completed = False

class TMC8Worker(QThread):
    finished = pyqtSignal()
    error_signal = pyqtSignal(str)
    resume_signal = pyqtSignal()
    process_completed = pyqtSignal(int)
    process_failed = pyqtSignal(int)
    cycle_approved = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.message_display_duration = 2
        self.time_limit = 10
        self.total_cycles_approved = 0

        self.pressure_to_psh1 = ProcessTracker(self.time_limit)
        self.pressure_to_psh2 = ProcessTracker(self.time_limit)
        self.isolation_to_ish = ProcessTracker(self.time_limit)
        self.separation_to_ssh = ProcessTracker(self.time_limit)

        self.message_display_start_time = None
        self.paused = False
        self.error_message = ""
        self.hand_tracker = HandTracker(history_length=15)
        self.stop_flag = False

    def run(self):
        self.run_tmc8_process()
        self.finished.emit()

    def is_point_in_roi(self, x, y, roi):
        return roi[0][0] < x < roi[1][0] and roi[0][1] < y < roi[1][1]

    def detect_hands(self, frame, results, rois):
        hand_info = []
        handedness_confidence_threshold = 0.9
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                x, y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
                detected_rois = {roi_label: self.is_point_in_roi(x, y, roi) for roi_label, roi in rois.items()}
                
                handedness = results.multi_handedness[idx].classification[0]
                hand_side = handedness.label
                confidence_score = handedness.score
            
                if confidence_score >= handedness_confidence_threshold:
                    hand_info.append((hand_landmarks, detected_rois, hand_side))
        
        self.hand_tracker.update(hand_info)
        return self.hand_tracker.get_current_hands()

    def highlight_rois(self, frame, hand_info, rois):
        for hand_landmarks, detected_rois, hand_side in hand_info:
            for roi_label, detected in detected_rois.items():
                if detected:
                    roi = rois[roi_label]
                    cv2.rectangle(frame, roi[0], roi[1], (0, 255, 0), 2)
                    cv2.putText(frame, f"{hand_side} Hand in {roi_label}", (roi[0][0], roi[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    def check_hand_constraints(self, detected_rois, hand_side):
        if hand_side == "Left" and ("Pressure Seal Box" in detected_rois and detected_rois["Pressure Seal Box"]):
            return "Wrong hand used: Use right hand for Pressure Seal Box."
        if hand_side == "Right" and (
            ("Isolation Seal Box" in detected_rois and detected_rois["Isolation Seal Box"]) or
            ("Separation Seal Box" in detected_rois and detected_rois["Separation Seal Box"])
        ):
            return "Wrong hand used: Use left hand for Isolation or Separation Seal Box."
        return None

    def track_pressure_to_psh1(self, detected_rois, current_time, hand_side):
        if detected_rois.get("Pressure Seal Box"):
            self.pressure_to_psh1.start(current_time, hand_side)
        if detected_rois.get("PSH1"):
            if self.pressure_to_psh1.hand_side == hand_side:
                self.pressure_to_psh1.end(current_time)
                if self.pressure_to_psh1.update(current_time):
                    self.process_completed.emit(0)
        elif (detected_rois.get("ISH") or detected_rois.get("SSH")) and self.pressure_to_psh1.start_time and not self.pressure_to_psh1.completed and self.pressure_to_psh1.hand_side == hand_side:
            self.pause_with_error("Wrong process executed: Pressure Seal Box to ISH/SSH. Press reset button.")
            self.process_failed.emit(0)

    def track_pressure_to_psh2(self, detected_rois, current_time, hand_side):
        if detected_rois.get("Pressure Seal Box"):
            self.pressure_to_psh2.start(current_time, hand_side)
        if detected_rois.get("PSH2"):
            if self.pressure_to_psh2.hand_side == hand_side:
                self.pressure_to_psh2.end(current_time)
                if self.pressure_to_psh2.update(current_time):
                    self.process_completed.emit(1)
        elif (detected_rois.get("ISH") or detected_rois.get("SSH")) and self.pressure_to_psh2.start_time and not self.pressure_to_psh2.completed and self.pressure_to_psh2.hand_side == hand_side:
            self.pause_with_error("Wrong process executed: Pressure Seal Box to ISH/SSH. Press reset button.")
            self.process_failed.emit(1)

    def track_isolation_to_ish(self, detected_rois, current_time, hand_side):
        if detected_rois.get("Isolation Seal Box"):
            self.isolation_to_ish.start(current_time, hand_side)
        if detected_rois.get("ISH"):
            if self.isolation_to_ish.hand_side == hand_side:
                self.isolation_to_ish.end(current_time)
                if self.isolation_to_ish.update(current_time):
                    self.process_completed.emit(2)
        elif (detected_rois.get("PSH1") or detected_rois.get("PSH2") or detected_rois.get("SSH")) and self.isolation_to_ish.start_time and not self.isolation_to_ish.completed and self.isolation_to_ish.hand_side == hand_side:
            self.pause_with_error("Wrong process executed: Isolation Seal Box to PSH1/PSH2/SSH. Press reset button.")
            self.process_failed.emit(2)

    def track_separation_to_ssh(self, detected_rois, current_time, hand_side):
        if detected_rois.get("Separation Seal Box"):
            self.separation_to_ssh.start(current_time, hand_side)
        if detected_rois.get("SSH"):
            if self.separation_to_ssh.hand_side == hand_side:
                self.separation_to_ssh.end(current_time)
                if self.separation_to_ssh.update(current_time):
                    self.process_completed.emit(3)
        elif (detected_rois.get("PSH1") or detected_rois.get("PSH2") or detected_rois.get("ISH")) and self.separation_to_ssh.start_time and not self.separation_to_ssh.completed and self.separation_to_ssh.hand_side == hand_side:
            self.pause_with_error("Wrong process executed: Separation Seal Box to PSH1/PSH2/ISH. Press reset button.")
            self.process_failed.emit(3)

    def track_processes(self, hand_info, current_time):
        if self.paused:
            return
    
        for hand_landmarks, detected_rois, hand_side in hand_info:
            error_message = self.check_hand_constraints(detected_rois, hand_side)
            if error_message:
                self.pause_with_error("Wrong hand used. Press reset button.")
                return

            if hand_side == "Left":
                self.track_isolation_to_ish(detected_rois, current_time, hand_side)
                self.track_separation_to_ssh(detected_rois, current_time, hand_side)
            else:
                self.track_pressure_to_psh1(detected_rois, current_time, hand_side)
                self.track_pressure_to_psh2(detected_rois, current_time, hand_side)

        self.pressure_to_psh1.update(current_time)
        self.pressure_to_psh2.update(current_time)
        self.isolation_to_ish.update(current_time)
        self.separation_to_ssh.update(current_time)

    def pause_with_error(self, message):
        self.paused = True
        self.error_message = message
        self.error_signal.emit(self.error_message)

    def display_message(self, frame, current_time):
        if all(process.completed for process in [self.pressure_to_psh1, self.pressure_to_psh2, self.isolation_to_ish, self.separation_to_ssh]):
            if self.message_display_start_time is None:
                self.message_display_start_time = current_time
                self.total_cycles_approved += 1
                print("Cycle Approved!")
                self.cycle_approved.emit()
            
            if (current_time - self.message_display_start_time) <= self.message_display_duration:
                cv2.putText(frame, "Cycle Approved!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            if (current_time - self.message_display_start_time) > self.message_display_duration:
                self.reset_tracking_variables()
                self.message_display_start_time = None
        elif self.message_display_start_time is not None:
            if (current_time - self.message_display_start_time) <= self.message_display_duration:
                cv2.putText(frame, "Cycle Approved!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                self.message_display_start_time = None

    def reset_tracking_variables(self):
        self.pressure_to_psh1.reset()
        self.pressure_to_psh2.reset()
        self.isolation_to_ish.reset()
        self.separation_to_ssh.reset()
        self.paused = False
        self.error_message = ""
        self.cycle_approved.emit()

    def run_tmc8_process(self):
        rois = read_rois_from_file("roi_temp.txt")
        cap = cv2.VideoCapture(0)

        with mp_hands.Hands(static_image_mode=False, model_complexity = 1, min_detection_confidence=0.1, min_tracking_confidence=0.9, max_num_hands=2) as hands:
            while cap.isOpened() and not self.isInterruptionRequested() and not self.stop_flag:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)
                    current_time = time.time()

                    hand_info = self.detect_hands(frame, results, rois)
                    self.highlight_rois(frame, hand_info, rois)

                    if not self.paused:
                        self.track_processes(hand_info, current_time)
                        self.display_message(frame, current_time)

                    if self.error_message:
                        cv2.putText(frame, self.error_message, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    QApplication.instance().topLevelWidgets()[0].camera_widget.update_frame(frame)

                

        cap.release()

    def reset(self):
        self.reset_tracking_variables()
        self.paused = False
        self.error_message = ""
        self.resume_signal.emit()
        self.cycle_approved.emit()

class TMC8eWorker(QThread):
    finished = pyqtSignal()
    error_signal = pyqtSignal(str)
    resume_signal = pyqtSignal()
    process_completed = pyqtSignal(int)
    process_failed = pyqtSignal(int)
    cycle_approved = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.message_display_duration = 2
        self.time_limit = 10
        self.total_cycles_approved = 0

        self.pressure_to_psh1 = ProcessTracker(self.time_limit)
        self.pressure_to_psh2 = ProcessTracker(self.time_limit)
        self.isolation_to_ish = ProcessTracker(self.time_limit)
        self.isolation_to_ssh = ProcessTracker(self.time_limit)

        self.message_display_start_time = None
        self.paused = False
        self.error_message = ""
        self.hand_tracker = HandTracker(history_length=15)
        self.stop_flag = False
   
    def run(self):
        self.run_tmc8e_process()
        self.finished.emit()

    def is_point_in_roi(self, x, y, roi):
        return roi[0][0] < x < roi[1][0] and roi[0][1] < y < roi[1][1]

    def detect_hands(self, frame, results, rois):
        hand_info = []
        handedness_confidence_threshold = 0.9
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                x, y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
                detected_rois = {roi_label: self.is_point_in_roi(x, y, roi) for roi_label, roi in rois.items()}
                
                handedness = results.multi_handedness[idx].classification[0]
                hand_side = handedness.label
                confidence_score = handedness.score
            
                if confidence_score >= handedness_confidence_threshold:
                    hand_info.append((hand_landmarks, detected_rois, hand_side))
        
        self.hand_tracker.update(hand_info)
        return self.hand_tracker.get_current_hands()

    def highlight_rois(self, frame, hand_info, rois):
        for hand_landmarks, detected_rois, hand_side in hand_info:
            for roi_label, detected in detected_rois.items():
                if detected:
                    roi = rois[roi_label]
                    cv2.rectangle(frame, roi[0], roi[1], (0, 255, 0), 2)
                    cv2.putText(frame, f"{hand_side} Hand in {roi_label}", (roi[0][0], roi[0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    def check_hand_constraints(self, detected_rois, hand_side):
        if hand_side == "Left" and ("Pressure Seal Box" in detected_rois and detected_rois["Pressure Seal Box"]):
            return "Wrong hand used: Use right hand for Pressure Seal Box."
        if hand_side == "Right" and (
            ("Isolation Seal Box" in detected_rois and detected_rois["Isolation Seal Box"])
        ):
            return "Wrong hand used: Use left hand for Isolation or Separation Seal Box."
        if (
            "Separation Seal Box" in detected_rois and detected_rois["Separation Seal Box"]
        ):
            return "Wrong Seal Box for TMC8e."
        return None

    def track_pressure_to_psh1(self, detected_rois, current_time, hand_side):
        if detected_rois.get("Pressure Seal Box"):
            self.pressure_to_psh1.start(current_time, hand_side)
        if detected_rois.get("PSH1"):
            if self.pressure_to_psh1.hand_side == hand_side:
                self.pressure_to_psh1.end(current_time)
                if self.pressure_to_psh1.update(current_time):
                    self.process_completed.emit(0)
        elif (detected_rois.get("ISH") or detected_rois.get("SSH")) and self.pressure_to_psh1.start_time and not self.pressure_to_psh1.completed and self.pressure_to_psh1.hand_side == hand_side:
            self.pause_with_error("Wrong process executed: Pressure Seal Box to ISH/SSH. Press reset button.")
            self.process_failed.emit(0)

    def track_pressure_to_psh2(self, detected_rois, current_time, hand_side):
        if detected_rois.get("Pressure Seal Box"):
            self.pressure_to_psh2.start(current_time, hand_side)
        if self.pressure_to_psh2.hand_side == hand_side: 
            if detected_rois.get("PSH2"):
                self.pressure_to_psh2.end(current_time)
                if self.pressure_to_psh2.update(current_time):
                    self.process_completed.emit(1)
        elif (detected_rois.get("ISH") or detected_rois.get("SSH")) and self.pressure_to_psh2.start_time and not self.pressure_to_psh2.completed and self.pressure_to_psh2.hand_side == hand_side:
            self.pause_with_error("Wrong process executed: Pressure Seal Box to ISH/SSH. Press reset button.")
            self.process_failed.emit(1)

    def track_isolation_to_ish(self, detected_rois, current_time, hand_side):
        if detected_rois.get("Isolation Seal Box"):
            self.isolation_to_ish.start(current_time, hand_side)
        if detected_rois.get("ISH"):
            if self.isolation_to_ish.hand_side == hand_side:
                self.isolation_to_ish.end(current_time)
                if self.isolation_to_ish.update(current_time):
                    self.process_completed.emit(2)
        elif (detected_rois.get("PSH1") or detected_rois.get("PSH2")) and self.isolation_to_ish.start_time and not self.isolation_to_ish.completed and self.isolation_to_ish.hand_side == hand_side:
            self.pause_with_error("Wrong process executed: Isolation Seal Box to PSH1/PSH2. Press reset button.")
            self.process_failed.emit(2)

    def track_isolation_to_ssh(self, detected_rois, current_time, hand_side):
        if detected_rois.get("Isolation Seal Box"):
            self.isolation_to_ssh.start(current_time, hand_side)
        if detected_rois.get("SSH"):
            if self.isolation_to_ssh.hand_side == hand_side:
                self.isolation_to_ssh.end(current_time)
                if self.isolation_to_ssh.update(current_time):
                    self.process_completed.emit(3)
        elif (detected_rois.get("PSH1") or detected_rois.get("PSH2")) and self.isolation_to_ssh.start_time and not self.isolation_to_ssh.completed and self.isolation_to_ssh.hand_side == hand_side:
            self.pause_with_error("Wrong process executed: Isolation Seal Box to PSH1/PSH2. Press reset button.")
            self.process_failed.emit(3)

    def track_processes(self, hand_info, current_time):
        if self.paused:
            return
    
        for hand_landmarks, detected_rois, hand_side in hand_info:
            error_message = self.check_hand_constraints(detected_rois, hand_side)
            if error_message:
                self.pause_with_error("Wrong hand used. Press reset button.")
                return

            if hand_side == "Left":
                self.track_isolation_to_ish(detected_rois, current_time, hand_side)
                self.track_isolation_to_ssh(detected_rois, current_time, hand_side)
            else:
                self.track_pressure_to_psh1(detected_rois, current_time, hand_side)
                self.track_pressure_to_psh2(detected_rois, current_time, hand_side)

        self.pressure_to_psh1.update(current_time)
        self.pressure_to_psh2.update(current_time)
        self.isolation_to_ish.update(current_time)
        self.isolation_to_ssh.update(current_time)

    def pause_with_error(self, message):
        self.paused = True
        self.error_message = message
        self.error_signal.emit(self.error_message)

    def display_message(self, frame, current_time):
        if all(process.completed for process in [self.pressure_to_psh1, self.pressure_to_psh2, self.isolation_to_ish, self.isolation_to_ssh]):
            if self.message_display_start_time is None:
                self.message_display_start_time = current_time
                self.total_cycles_approved += 1
                print("Cycle Approved!")
                self.cycle_approved.emit()
            
            if (current_time - self.message_display_start_time) <= self.message_display_duration:
                cv2.putText(frame, "Cycle Approved!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            if (current_time - self.message_display_start_time) > self.message_display_duration:
                self.reset_tracking_variables()
                self.message_display_start_time = None
        elif self.message_display_start_time is not None:
            if (current_time - self.message_display_start_time) <= self.message_display_duration:
                cv2.putText(frame, "Cycle Approved!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                self.message_display_start_time = None

    def reset_tracking_variables(self):
        self.pressure_to_psh1.reset()
        self.pressure_to_psh2.reset()
        self.isolation_to_ish.reset()
        self.isolation_to_ssh.reset()
        self.paused = False
        self.error_message = ""
        self.cycle_approved.emit()

    def run_tmc8e_process(self):
        rois = read_rois_from_file("roi_temp.txt")
        cap = cv2.VideoCapture(0)

        with mp_hands.Hands(static_image_mode=False, model_complexity = 1, min_detection_confidence=0.1, min_tracking_confidence=0.9, max_num_hands=2) as hands:
            while cap.isOpened() and not self.isInterruptionRequested() and not self.stop_flag:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)
                    current_time = time.time()

                    hand_info = self.detect_hands(frame, results, rois)
                    self.highlight_rois(frame, hand_info, rois)

                    if not self.paused:
                        self.track_processes(hand_info, current_time)
                        self.display_message(frame, current_time)

                    if self.error_message:
                        cv2.putText(frame, self.error_message, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    QApplication.instance().topLevelWidgets()[0].camera_widget.update_frame(frame)

        cap.release()

    def reset(self):
        self.reset_tracking_variables()
        self.paused = False
        self.error_message = ""
        self.resume_signal.emit()
        self.cycle_approved.emit() 

class ROIThread(QThread):
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()

    def run(self):
        global rois, current_roi, drawing, roi_labels, current_label_index

        rois = []
        current_roi = []
        drawing = False
        roi_labels = ["Pressure Seal Box", "Isolation Seal Box", "Separation Seal Box", "PSH1", "PSH2", "ISH", "SSH"]
        current_label_index = 0

        def draw_rectangle(event, x, y, flags, param):
            global current_roi, drawing, rois, current_label_index

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                current_roi = [(x, y)]

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    img_copy = frame.copy()
                    cv2.rectangle(img_copy, current_roi[0], (x, y), (0, 255, 0), 2)
                    cv2.putText(img_copy, f"Defining: {roi_labels[current_label_index]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Frame', img_copy)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                current_roi.append((x, y))
                rois.append((current_roi[0], (x, y), roi_labels[current_label_index]))
                current_label_index += 1
                current_roi = []
                if current_label_index >= len(roi_labels):
                    print("Defined ROIs: ", rois)
                    save_rois_to_file(rois)
                    cv2.destroyAllWindows()

        def save_rois_to_file(rois):
            with open('roi_temp.txt', 'w') as f:
                for roi in rois:
                    f.write(f"{roi[2]}: {roi[0][0]} {roi[0][1]} {roi[1][0]} {roi[1][1]}\n")

        cap = cv2.VideoCapture(0)

        cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback('Frame', draw_rectangle)

        while current_label_index < len(roi_labels):
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            for roi in rois:
                cv2.rectangle(frame, roi[0], roi[1], (0, 255, 0), 2)
                cv2.putText(frame, roi[2], (roi[0][0], roi[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

            if current_label_index < len(roi_labels):
                cv2.putText(frame, f"Defining: {roi_labels[current_label_index]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Frame', frame)

            if cv2.waitKey(60) & 0xFF == ord('q'):
                break

        print("Final ROIs:", rois)
        cap.release()
        cv2.destroyAllWindows()
        self.finished.emit()

class CameraWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

    def update_frame(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seal Insertion Inspection")
        self.showFullScreen()
        self.initUI()
        self.tmc8_worker = None
        self.tmc8e_worker = None
        self.current_worker = None

        self.timer = QTimer(self)
        self.timer.start(100)

    def initUI(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout()

        left_layout = QVBoxLayout()

        # Camera widget (left side)
        self.camera_widget = CameraWidget()
        main_layout.addWidget(self.camera_widget, 1)

        self.cycle_approved_label = QLabel("")
        self.cycle_approved_label.setStyleSheet("font-size: 24px; font-weight: bold; color: green;")
        self.cycle_approved_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.cycle_approved_label)

        # Right side layout
        right_layout = QVBoxLayout()

        # Title
        title = QLabel("Seal Insertion Inspection")
        title.setStyleSheet("font-size: 44px; font-weight: bold; padding:20px")
        title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(title)

        # Spacer
        right_layout.addStretch(1)

        self.seal_status_labels = []
        seal_labels = ["Seal 1:", "Seal 2:", "Seal 3:", "Seal 4:"]
        for seal in seal_labels:
            seal_layout = QHBoxLayout()
            label = QLabel(seal)
            label.setStyleSheet("font-size: 38px;")
            seal_layout.addWidget(label)

            status_label = QLabel("")
            status_label.setStyleSheet("font-size: 38px; font-weight: bold;")
            seal_layout.addWidget(status_label)
            self.seal_status_labels.append(status_label)
            
            right_layout.addLayout(seal_layout)

        right_layout.addStretch(1)

        # Buttons
        button_layout = QHBoxLayout()
        tmc8_button = QPushButton("TMC 8")
        tmc8_button.setStyleSheet("font-size: 18px; padding: 10px;")
        tmc8_button.clicked.connect(self.run_tmc8)
        button_layout.addWidget(tmc8_button)

        tmc8e_button = QPushButton("TMC 8e")
        tmc8e_button.setStyleSheet("font-size: 18px; padding: 10px;")
        tmc8e_button.clicked.connect(self.run_tmc8e)
        button_layout.addWidget(tmc8e_button)

        reset_button = QPushButton("Reset")
        reset_button.setStyleSheet("font-size: 18px; padding: 10px;")
        reset_button.clicked.connect(self.reset_current_cycle)
        button_layout.addWidget(reset_button)

        roi_button = QPushButton("ROI")
        roi_button.setStyleSheet("font-size: 18px; padding: 10px;")
        roi_button.clicked.connect(self.define_roi)
        button_layout.addWidget(roi_button)

        right_layout.addLayout(button_layout)

        # Error label
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("font-size: 16px; color: red;")
        self.error_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.error_label)

        right_layout.addStretch(1)

        # Add right layout to main layout
        main_layout.addLayout(right_layout, 1)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def define_roi(self):
        self.stop_current_worker()
        self.roi_thread = ROIThread()
        self.roi_thread.finished.connect(self.on_roi_finished)
        self.roi_thread.start()

    def on_roi_finished(self):
        self.roi_thread = None

    def run_tmc8(self):
        self.stop_current_worker()
        if self.tmc8_worker is None or not self.tmc8_worker.isRunning():
            self.tmc8_worker = TMC8Worker()
            self.tmc8_worker.finished.connect(self.on_tmc8_finished)
            self.tmc8_worker.error_signal.connect(self.show_error_message)
            self.tmc8_worker.resume_signal.connect(self.clear_error_message)
            self.tmc8_worker.process_completed.connect(self.update_seal_status)
            self.tmc8_worker.process_failed.connect(self.update_seal_status_nok)
            self.tmc8_worker.cycle_approved.connect(self.on_cycle_approved)
            self.tmc8_worker.start()
            self.current_worker = self.tmc8_worker

    def run_tmc8e(self):
        self.stop_current_worker()
        if self.tmc8e_worker is None or not self.tmc8e_worker.isRunning():
            self.tmc8e_worker = TMC8eWorker()
            self.tmc8e_worker.finished.connect(self.on_tmc8e_finished)
            self.tmc8e_worker.error_signal.connect(self.show_error_message)
            self.tmc8e_worker.resume_signal.connect(self.clear_error_message)
            self.tmc8e_worker.process_completed.connect(self.update_seal_status)
            self.tmc8e_worker.process_failed.connect(self.update_seal_status_nok)
            self.tmc8e_worker.cycle_approved.connect(self.on_cycle_approved)
            self.tmc8e_worker.start()
            self.current_worker = self.tmc8e_worker

    def update_seal_status(self, process_index):
        self.seal_status_labels[process_index].setText("OK")
        self.seal_status_labels[process_index].setStyleSheet("font-size: 38px; font-weight: bold; color: green;")

    def update_seal_status_nok(self, process_index):
        self.seal_status_labels[process_index].setText("NOK")
        self.seal_status_labels[process_index].setStyleSheet("font-size: 38px; font-weight: bold; color: red;")

    def reset_seal_status(self):
        for label in self.seal_status_labels:
            label.setText("")

    def stop_current_worker(self):
        if self.current_worker:
            self.current_worker.stop_flag = True
            self.current_worker.wait()
            self.current_worker = None
        self.reset_seal_status()

    def on_tmc8_finished(self):
        self.tmc8_worker = None

    def on_tmc8e_finished(self):
        self.tmc8e_worker = None

    def on_cycle_approved(self):
        self.cycle_approved_label.setText("Cycle Approved!")
        self.reset_seal_status()
        QTimer.singleShot(2000, self.clear_cycle_approved)  # Clear after 2 seconds

    def clear_cycle_approved(self):
        self.cycle_approved_label.setText("")

    def reset_current_cycle(self):
        if self.tmc8_worker and self.tmc8_worker.isRunning():
            self.tmc8_worker.reset()
            self.clear_error_message()
        elif self.tmc8e_worker and self.tmc8e_worker.isRunning():
            self.tmc8e_worker.reset()
            self.clear_error_message()
        self.reset_seal_status()
        self.clear_cycle_approved()

    def show_error_message(self, message):
        self.error_label.setText(message)

    def clear_error_message(self):
        self.error_label.setText("")

    def closeEvent(self, event):
        self.stop_current_worker()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()
            QApplication.instance().quit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())