import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QGridLayout, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap
from datetime import datetime
# import time
from datetime import datetime, timedelta

class ValidationSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Validation System using Video Analytics")
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowFlags(self.windowFlags() | Qt.WindowCloseButtonHint)

        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Add spacing at the top
        self.main_layout.addSpacing(10)

        # Add title label
        self.title_label = QLabel("Process Validation System")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px;
            padding: 10px;
        """)
        self.main_layout.addWidget(self.title_label)

        # Create video display with increased size
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(QSize(800, 600))
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("""
            QLabel {
                margin: 20px;
                padding: 10px;
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 8px;
            }
        """)
        self.main_layout.addWidget(self.video_label)

        # Create statistics grid
        self.stats_layout = QGridLayout()
        self.stats_layout.setSpacing(20)
        
        stat_style = """
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
                padding: 5px;
            }
        """

        
        
        # Total Cycles
        self.total_cycles_label = QLabel("Total Cycles:")
        self.total_cycles_value = QLabel("0")
        self.total_cycles_label.setStyleSheet(stat_style)
        self.total_cycles_value.setStyleSheet(stat_style)
        self.stats_layout.addWidget(self.total_cycles_label, 0, 0)
        self.stats_layout.addWidget(self.total_cycles_value, 0, 1)
        
        # Correct Cycles
        self.correct_cycles_label = QLabel("Correct Cycles:")
        self.correct_cycles_value = QLabel("0")
        self.correct_cycles_label.setStyleSheet(stat_style)
        self.correct_cycles_value.setStyleSheet(stat_style)
        self.stats_layout.addWidget(self.correct_cycles_label, 0, 2)
        self.stats_layout.addWidget(self.correct_cycles_value, 0, 3)
        
        # Incorrect Cycles
        self.incorrect_cycles_label = QLabel("Incorrect Cycles:")
        self.incorrect_cycles_value = QLabel("0")
        self.incorrect_cycles_label.setStyleSheet(stat_style)
        self.incorrect_cycles_value.setStyleSheet(stat_style)
        self.stats_layout.addWidget(self.incorrect_cycles_label, 0, 4)
        self.stats_layout.addWidget(self.incorrect_cycles_value, 0, 5)

        self.main_layout.addSpacing(10)
        self.main_layout.addLayout(self.stats_layout)
        self.main_layout.addSpacing(10)

        # Create status message label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px;
        """)
        self.main_layout.addWidget(self.status_label)

        # Create Reset button
        self.reset_button = QPushButton("Reset Cycle")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 12px 24px;
                border-radius: 6px;
                font-size: 16px;
                min-width: 150px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.reset_button.clicked.connect(self.reset_cycle)
        self.main_layout.addWidget(self.reset_button)
        
        self.main_layout.addSpacing(20)


        # Initialize video capture
        self.cap = cv2.VideoCapture(0)

        # Initialize cycle counters
        self.total_cycles = 0
        self.correct_cycles = 0
        self.incorrect_cycles = 0

        # Load ROI definitions and process sequence
        self.roi_definitions = self.load_roi_definitions()
        self.process_sequence = self.load_process_sequence()
        
        self.last_detected_roi = None
        self.last_detected_roi = None
        self.last_detected_time = datetime.min
        # Initialize Mediapipe Hands
        self.mp_hands = mp.solutions.hands
        # self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.hands = self.mp_hands.Hands(model_complexity = 1, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.1, min_tracking_confidence=0.9)
        self.mp_draw = mp.solutions.drawing_utils

        self.previous_roi = None
        self.detected_sequence = []
        self.log_file = "hand_detection_log.txt"
        self.logged_rois = set()

        # Set up video timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

        self.is_paused = False

    def load_roi_definitions(self):
        roi_definitions = []
        try:
            with open('roi_definitions.txt', 'r') as f:
                for line in f:
                    label, x1, y1, x2, y2 = line.strip().split(',')
                    roi_definitions.append({
                        'label': label,
                        'start': (int(x1), int(y1)),
                        'end': (int(x2), int(y2)),
                        'color': (0, 255, 0)  # Default to green
                    })
        except FileNotFoundError:
            print("ROI definitions file not found")
        return roi_definitions
    
    def load_process_sequence(self):
        process_sequence = []
        try:
            with open('process_definitions.txt', 'r') as f:
                process_sequence = [line.strip() for line in f]
        except FileNotFoundError:
            print("Process definitions file not found")
        return process_sequence

    def update_frame(self):
        if self.is_paused:
            return  # Skip frame processing when paused

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)

            # Convert to RGB for Mediapipe
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)

            # Draw hand landmarks if detected
            detected_rois = set()
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            # Process each hand and check against ROIs
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_points = [
                        (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                        for lm in hand_landmarks.landmark
                    ]

                    for roi in self.roi_definitions:
                        points_in_roi = [
                            point for point in hand_points
                            if roi['start'][0] <= point[0] <= roi['end'][0] and
                               roi['start'][1] <= point[1] <= roi['end'][1]
                        ]

                        if len(points_in_roi) > 3:
                            current_time = datetime.now()
                            roi['color'] = (0, 0, 255)  # Highlight ROI in red
                            detected_rois.add(roi['label'])
                            if not self.detected_sequence or roi['label'] != self.detected_sequence[-1]:
                                self.detected_sequence.append(roi['label'])
                                self.log_presence_in_roi(roi['label'])
                                self.last_detected_roi = roi['label']
                                self.last_detected_time = current_time
                        else:
                            roi['color'] = (0, 255, 0)  # Revert to green

            self.logged_rois.intersection_update(detected_rois)
            self.check_sequence()

            # Draw ROIs on the frame
            for roi in self.roi_definitions:
                cv2.rectangle(frame, roi['start'], roi['end'], roi['color'], 2)
                label_position = (roi['start'][0], roi['start'][1] - 10)
                cv2.putText(frame, roi['label'], label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, roi['color'], 2)

            # Convert frame to Qt format
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Display image in the QLabel
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
    
    def log_detection(self, roi_label):
         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
         with open(self.log_file, 'a') as f:
             f.write(f"{roi_label},{timestamp}\n")

    def check_sequence(self):
        if self.detected_sequence == self.process_sequence:
            self.status_label.setText("Cycle successfully executed!")
            QApplication.processEvents()
            QTimer.singleShot(3000, self.clear_status_message) # Pause to display success message

            self.total_cycles += 1
            self.correct_cycles += 1
            self.update_cycle_counts()

            # self.append_log()  # Append detected sequence to log file
            self.detected_sequence.clear()

        # elif len(self.detected_sequence) > len(self.process_sequence) or \
        #         self.detected_sequence != self.process_sequence[:len(self.detected_sequence)]:
        if len(self.detected_sequence) == len(self.process_sequence) and \
                 self.detected_sequence != self.process_sequence[:len(self.detected_sequence)]: 
            # Sequence mismatch detected
            self.pause_video()
            self.status_label.setText("Incorrect cycle executed. Press reset button!")
            QApplication.processEvents()

    def pause_video(self):
        """Pause the video feed and processing."""
        self.is_paused = True

    def clear_status_message(self):
        self.detected_sequence.clear()
        self.status_label.setText("")

    def append_log(self):
        with open(self.log_file, 'a') as f:
            for roi in self.process_sequence:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{roi},{timestamp}\n")

    def reset_cycle(self):
        """Reset the current cycle."""
        # Reset cycle and clear error message
        self.status_label.setText("Cycle Reset")
        self.detected_sequence.clear()
        self.is_paused = False
        # Reset current cycle and update status
        self.status_label.setText("Cycle Reset")
        self.total_cycles += 1
        self.incorrect_cycles += 1
        self.update_cycle_counts()

    def update_cycle_counts(self):
        # Update the display of cycle counts
        self.total_cycles_value.setText(str(self.total_cycles))
        self.correct_cycles_value.setText(str(self.correct_cycles))
        self.incorrect_cycles_value.setText(str(self.incorrect_cycles))

    def log_presence_in_roi(self, roi_label):
        # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # log_entry = f"{roi_label},{timestamp}\n"
        current_time = datetime.now()
     
     # Debounce logic: Skip logging if the same ROI was logged within 500ms
        if self.last_detected_roi == roi_label and \
                (current_time - self.last_detected_time).total_seconds() < 0.5:
            return  # Skip logging to avoid duplication
 
     # Log the ROI
        self.last_detected_roi = roi_label
        self.last_detected_time = current_time
        log_entry = f"{roi_label},{current_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        with open(self.log_file, 'a') as f:
            f.write(log_entry)

    def closeEvent(self, event):
        # Clean up resources when closing
        self.cap.release()
        self.hands.close()

if __name__ == '__main__':
    app = QApplication([])
    window = ValidationSystem()
    window.show()
    app.exec_()

