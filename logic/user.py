import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap
from datetime import datetime, timedelta


class ValidationSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Validation System using Video Analytics")
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowFlags(self.windowFlags() | Qt.WindowCloseButtonHint)

        # ────────────────────────  MAIN LAYOUT  ────────────────────────
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.addSpacing(10)

        # Title
        self.title_label = QLabel("Process Validation System")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                font-size: 32px;
                font-weight: 700;
                letter-spacing: 1px;
                color: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:0,
                    stop:0 #8e44ad, stop:1 #3498db);
                padding: 12px;
            }
        """)
        self.main_layout.addWidget(self.title_label)

        # Video preview
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(QSize(800, 600))
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #12121a;
                border: 3px solid #2d2d3d;
                border-radius: 12px;
            }
        """)
        self.main_layout.addWidget(self.video_label)

        # ────────────────────────  STATS GRID  ────────────────────────
        self.stats_layout = QGridLayout()
        self.stats_layout.setSpacing(20)

        stat_style = """
            QLabel {
                font-size: 17px;
                font-weight: 600;
                padding: 6px 10px;
                border-radius: 6px;
                background-color: #2d2d3d;
            }
        """

        # Total cycles
        self.total_cycles_label = QLabel("Total Cycles:")
        self.total_cycles_value = QLabel("0")
        self.total_cycles_label.setStyleSheet(stat_style)
        self.total_cycles_value.setStyleSheet(stat_style)
        self.stats_layout.addWidget(self.total_cycles_label, 0, 0)
        self.stats_layout.addWidget(self.total_cycles_value, 0, 1)

        # Correct cycles
        self.correct_cycles_label = QLabel("Correct Cycles:")
        self.correct_cycles_value = QLabel("0")
        self.correct_cycles_label.setStyleSheet(stat_style)
        self.correct_cycles_value.setStyleSheet(stat_style)
        self.stats_layout.addWidget(self.correct_cycles_label, 0, 2)
        self.stats_layout.addWidget(self.correct_cycles_value, 0, 3)

        # Incorrect cycles
        self.incorrect_cycles_label = QLabel("Incorrect Cycles:")
        self.incorrect_cycles_value = QLabel("0")
        self.incorrect_cycles_label.setStyleSheet(stat_style)
        self.incorrect_cycles_value.setStyleSheet(stat_style)
        self.stats_layout.addWidget(self.incorrect_cycles_label, 0, 4)
        self.stats_layout.addWidget(self.incorrect_cycles_value, 0, 5)

        self.main_layout.addSpacing(10)
        self.main_layout.addLayout(self.stats_layout)
        self.main_layout.addSpacing(10)

        # Status message
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 19px;
                font-weight: 600;
                padding: 8px 14px;
                border-radius: 8px;
                background-color: #2d2d3d;
                qproperty-alignment: AlignCenter;
            }
        """)
        self.main_layout.addWidget(self.status_label)

        # Reset button
        self.reset_button = QPushButton("Reset Cycle")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #e67e22;
                color: #fff;
                padding: 14px 28px;
                font-size: 16px;
                font-weight: 600;
                border: none;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #aaa;
            }
        """)
        self.reset_button.clicked.connect(self.reset_cycle)
        self.main_layout.addWidget(self.reset_button)
        self.main_layout.addSpacing(20)

        # Countdown timer
        self.countdown_seconds = 0
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.update_countdown)

        # ────────────────────────  VIDEO / MP HANDS  ────────────────────────
        self.cap = cv2.VideoCapture(0)

        self.total_cycles = 0
        self.correct_cycles = 0
        self.incorrect_cycles = 0

        self.roi_definitions = self.load_roi_definitions()
        self.process_sequence = self.load_process_sequence()

        self.last_detected_roi = None
        self.last_detected_time = datetime.min

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=1, static_image_mode=False,
            max_num_hands=2, min_detection_confidence=0.1,
            min_tracking_confidence=0.9
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.previous_roi = None
        self.detected_sequence = []
        self.log_file = "hand_detection_log.txt"
        self.logged_rois = set()

        # Timer for video updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

        self.is_paused = False

    # ────────────────────────  COUNTDOWN HELPERS  ────────────────────────
    def start_reset_countdown(self, seconds: int = 3):
        self.countdown_seconds = seconds
        self.reset_button.setEnabled(False)
        self.update_countdown()
        self.countdown_timer.start(1000)

    def update_countdown(self):
        if self.countdown_seconds > 0:
            self.status_label.setText(
                f"Incorrect cycle – resetting in {self.countdown_seconds}s…")
            self.countdown_seconds -= 1
        else:
            self.countdown_timer.stop()
            self.reset_button.setEnabled(True)
            self.status_label.setText("Press “Reset Cycle” to start over.")

    # ────────────────────────  ROI / PROCESS CONFIG  ────────────────────────
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
                        'color': (0, 255, 0)  # default green
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

    # ────────────────────────  FRAME PROCESSING  ────────────────────────
    def update_frame(self):
        if self.is_paused:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        detected_rois = set()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks,
                                            self.mp_hands.HAND_CONNECTIONS)

            for hand_landmarks in results.multi_hand_landmarks:
                hand_points = [
                    (int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]))
                    for lm in hand_landmarks.landmark
                ]

                for roi in self.roi_definitions:
                    points_in_roi = [
                        p for p in hand_points
                        if roi['start'][0] <= p[0] <= roi['end'][0] and
                           roi['start'][1] <= p[1] <= roi['end'][1]
                    ]

                    if len(points_in_roi) > 3:
                        current_time = datetime.now()
                        roi['color'] = (0, 0, 255)  # red highlight
                        detected_rois.add(roi['label'])
                        if (not self.detected_sequence or
                                roi['label'] != self.detected_sequence[-1]):
                            self.detected_sequence.append(roi['label'])
                            self.log_presence_in_roi(roi['label'])
                            self.last_detected_roi = roi['label']
                            self.last_detected_time = current_time
                    else:
                        roi['color'] = (0, 255, 0)  # revert to green

        self.logged_rois.intersection_update(detected_rois)
        self.check_sequence()

        # Draw ROIs
        for roi in self.roi_definitions:
            cv2.rectangle(frame, roi['start'], roi['end'], roi['color'], 2)
            label_pos = (roi['start'][0], roi['start'][1] - 10)
            cv2.putText(frame, roi['label'], label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, roi['color'], 2)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h,
                          bytes_per_line, QImage.Format_RGB888)
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    # ────────────────────────  LOGIC HELPERS  ────────────────────────
    def check_sequence(self):
        if self.detected_sequence == self.process_sequence:
            self.status_label.setText("Cycle successfully executed!")
            QApplication.processEvents()
            QTimer.singleShot(3000, self.clear_status_message)
            self.total_cycles += 1
            self.correct_cycles += 1
            self.update_cycle_counts()
            self.detected_sequence.clear()

        elif (len(self.detected_sequence) == len(self.process_sequence) and
              self.detected_sequence != self.process_sequence[:len(self.detected_sequence)]):
            self.pause_video()
            self.start_reset_countdown(3)

    def pause_video(self):
        self.is_paused = True

    def clear_status_message(self):
        self.detected_sequence.clear()
        self.status_label.setText("")

    def reset_cycle(self):
        self.status_label.setText("Cycle Reset")
        self.detected_sequence.clear()
        self.is_paused = False
        self.total_cycles += 1
        self.incorrect_cycles += 1
        self.update_cycle_counts()

    def update_cycle_counts(self):
        self.total_cycles_value.setText(str(self.total_cycles))
        self.correct_cycles_value.setText(str(self.correct_cycles))
        self.incorrect_cycles_value.setText(str(self.incorrect_cycles))

    def log_presence_in_roi(self, roi_label):
        current_time = datetime.now()
        if (self.last_detected_roi == roi_label and
                (current_time - self.last_detected_time).total_seconds() < 0.5):
            return
        self.last_detected_roi = roi_label
        self.last_detected_time = current_time
        with open(self.log_file, 'a') as f:
            f.write(f"{roi_label},{current_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ────────────────────────  CLEAN-UP  ────────────────────────
    def closeEvent(self, event):
        self.cap.release()
        self.hands.close()
        super().closeEvent(event)


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)

    # Global dark theme base
    app.setStyleSheet("""
        QWidget {
            font-family: 'Segoe UI', sans-serif;
            color: #ecf0f1;
            background-color: #1e1e2e;
        }
    """)

    window = ValidationSystem()
    window.show()
    sys.exit(app.exec_())
