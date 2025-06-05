import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap


class ROIDefiner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Admin Mode 1 – ROI Definition")
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowFlags(self.windowFlags() | Qt.WindowCloseButtonHint)

        # ───────────────────────  MAIN LAYOUT  ───────────────────────
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addSpacing(10)

        # Title
        self.title_label = QLabel("Region of Interest Definition System")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            QLabel {
                font-size: 32px;
                font-weight: 700;
                letter-spacing: 1px;
                color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
                                       stop:0 #8e44ad, stop:1 #3498db);
                padding: 12px;
            }
        """)
        self.layout.addWidget(self.title_label)

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
                margin: 20px;
                padding: 10px;
            }
        """)
        self.layout.addWidget(self.video_label)

        # ───────────────────────  BUTTON BAR  ───────────────────────
        self.button_layout = QHBoxLayout()
        base_btn = """
            QPushButton {
                color: #fff;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: 600;
                border: none;
                border-radius: 8px;
                min-width: 150px;
                margin: 10px;
            }
            QPushButton:disabled { background-color: #555; color: #aaa; }
        """

        self.start_button = QPushButton("Start Drawing ROI")
        self.start_button.setStyleSheet(
            base_btn + "QPushButton { background-color: #27ae60; }"
                        "QPushButton:hover { background-color: #1f8c4d; }")

        self.save_button = QPushButton("Save ROI")
        self.save_button.setStyleSheet(
            base_btn + "QPushButton { background-color: #3498db; }"
                        "QPushButton:hover { background-color: #2980b9; }")

        self.clear_button = QPushButton("Clear ROIs")
        self.clear_button.setStyleSheet(
            base_btn + "QPushButton { background-color: #e74c3c; }"
                        "QPushButton:hover { background-color: #c0392b; }")

        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.save_button)
        self.button_layout.addWidget(self.clear_button)
        self.layout.addLayout(self.button_layout)

        # ROI label input
        self.roi_label_input = QLineEdit()
        self.roi_label_input.setPlaceholderText("Enter ROI Label")
        self.roi_label_input.setMaximumWidth(400)
        self.roi_label_input.setMinimumWidth(200)
        self.roi_label_input.setStyleSheet("""
            QLineEdit {
                padding: 12px;
                font-size: 16px;
                background-color: #1e1e2e;
                color: #ecf0f1;
                border: 2px solid #2d2d3d;
                border-radius: 6px;
                margin: 10px;
            }
            QLineEdit:focus {
                border: 2px solid #3498db;
            }
        """)
        self.layout.addWidget(self.roi_label_input, alignment=Qt.AlignCenter)

        # ───────────────────────  SIGNALS  ───────────────────────
        self.start_button.clicked.connect(self.start_drawing)
        self.save_button.clicked.connect(self.save_roi)
        self.clear_button.clicked.connect(self.clear_rois)

        # ───────────────────────  VIDEO / ROI SETUP  ───────────────────────
        self.cap = cv2.VideoCapture(0)
        self.drawing = False
        self.roi_start = None
        self.roi_end = None
        self.rois = []
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ───────────────────────  FRAME UPDATE  ───────────────────────
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        self.display_frame = frame.copy()

        # Draw saved ROIs
        for roi in self.rois:
            cv2.rectangle(self.display_frame, roi['start'], roi['end'], (0, 255, 0), 2)
            cv2.putText(self.display_frame, roi['label'],
                        (roi['start'][0], roi['start'][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw current ROI while drawing
        if self.drawing and self.roi_start:
            cv2.rectangle(self.display_frame, self.roi_start, self.roi_end, (255, 0, 0), 2)

        # Convert to Qt image / show
        rgb_image = cv2.cvtColor(self.display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_img = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        scaled = QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

        # Update scaling for accurate mouse mapping
        self.scale_factor = scaled.width() / w
        self.offset_x = (self.video_label.width() - scaled.width()) // 2
        self.offset_y = (self.video_label.height() - scaled.height()) // 2

    # ───────────────────────  DRAWING HANDLERS  ───────────────────────
    def start_drawing(self):
        self.drawing = True
        self.video_label.mousePressEvent = self.mouse_press
        self.video_label.mouseMoveEvent = self.mouse_move
        self.video_label.mouseReleaseEvent = self.mouse_release

    def mouse_press(self, event):
        if self.drawing:
            self.roi_start = self.get_frame_coordinates(event.x(), event.y())

    def mouse_move(self, event):
        if self.drawing and self.roi_start:
            self.roi_end = self.get_frame_coordinates(event.x(), event.y())

    def mouse_release(self, event):
        if self.drawing:
            self.roi_end = self.get_frame_coordinates(event.x(), event.y())
            self.drawing = False

    def get_frame_coordinates(self, x, y):
        return (
            int((x - self.offset_x) / self.scale_factor),
            int((y - self.offset_y) / self.scale_factor)
        )

    # ───────────────────────  SAVE / CLEAR  ───────────────────────
    def save_roi(self):
        if self.roi_start and self.roi_end:
            label = self.roi_label_input.text()
            if label:
                self.rois.append({'start': self.roi_start,
                                  'end': self.roi_end,
                                  'label': label})
                self.roi_start = self.roi_end = None
                self.roi_label_input.clear()
                self.save_rois_to_file()

    def clear_rois(self):
        self.rois.clear()
        self.save_rois_to_file()

    def save_rois_to_file(self):
        with open('roi_definitions.txt', 'w') as f:
            for roi in self.rois:
                f.write(f"{roi['label']},{roi['start'][0]},{roi['start'][1]},"
                        f"{roi['end'][0]},{roi['end'][1]}\n")

    # ───────────────────────  CLEAN-UP  ───────────────────────
    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)


# ─────────────────────────────  ENTRY  ─────────────────────────────
if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QWidget {
            font-family: 'Segoe UI', sans-serif;
            background-color: #1e1e2e;
            color: #ecf0f1;
        }
    """)

    window = ROIDefiner()
    window.show()
    sys.exit(app.exec_())
