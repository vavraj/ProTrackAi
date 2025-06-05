import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QLineEdit, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap

class ROIDefiner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Admin Mode 1 - ROI Definition")
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowFlags(self.windowFlags() | Qt.WindowCloseButtonHint)

        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Add spacing at the top
        self.layout.addSpacing(10)

        # Add title
        self.title_label = QLabel("Region of Interest Definition System")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px;
            padding: 10px;
        """)
        self.layout.addWidget(self.title_label)

        # Video display with enhanced size
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
        self.layout.addWidget(self.video_label)

        # Button layout with improved styling
        self.button_layout = QHBoxLayout()
        button_style = """
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
        """
        
        self.start_button = QPushButton("Start Drawing ROI")
        self.start_button.setStyleSheet(button_style)
        
        self.save_button = QPushButton("Save ROI")
        self.save_button.setStyleSheet(button_style)
        
        self.clear_button = QPushButton("Clear ROIs")
        self.clear_button.setStyleSheet(button_style.replace("#3498db", "#e74c3c")
                                               .replace("#2980b9", "#c0392b"))
        
        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.save_button)
        self.button_layout.addWidget(self.clear_button)
        self.layout.addLayout(self.button_layout)

        # ROI Label input with styling
        self.roi_label_input = QLineEdit()
        self.roi_label_input.setPlaceholderText("Enter ROI Label")
        self.roi_label_input.setStyleSheet("""
            QLineEdit {
                padding: 12px;
                font-size: 16px;
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                margin: 10px;
            }
            QLineEdit:focus {
                border: 2px solid #3498db;
            }
        """)
        self.roi_label_input.setMaximumWidth(400)
        self.roi_label_input.setMinimumWidth(200)
        self.layout.addWidget(self.roi_label_input, alignment=Qt.AlignCenter)

        # Connect buttons
        self.start_button.clicked.connect(self.start_drawing)
        self.save_button.clicked.connect(self.save_roi)
        self.clear_button.clicked.connect(self.clear_rois)

        # Initialize video capture and variables
        self.cap = cv2.VideoCapture(0)
        self.drawing = False
        self.roi_start = None
        self.roi_end = None
        self.rois = []
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # Set up video timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.display_frame = frame.copy()
            
            # Draw existing ROIs
            for roi in self.rois:
                cv2.rectangle(self.display_frame, roi['start'], roi['end'], (0, 255, 0), 2)
                cv2.putText(self.display_frame, roi['label'], 
                          (roi['start'][0], roi['start'][1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Draw current ROI if drawing
            if self.drawing and self.roi_start:
                cv2.rectangle(self.display_frame, self.roi_start, self.roi_end, (255, 0, 0), 2)

            # Convert to Qt format and display
            rgb_image = cv2.cvtColor(self.display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
            
            # Update scale factor and offset for accurate coordinate conversion
            self.scale_factor = scaled_pixmap.width() / w
            self.offset_x = (self.video_label.width() - scaled_pixmap.width()) // 2
            self.offset_y = (self.video_label.height() - scaled_pixmap.height()) // 2

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
        frame_x = int((x - self.offset_x) / self.scale_factor)
        frame_y = int((y - self.offset_y) / self.scale_factor)
        return (frame_x, frame_y)

    def save_roi(self):
        if self.roi_start and self.roi_end:
            label = self.roi_label_input.text()
            if label:
                self.rois.append({
                    'start': self.roi_start,
                    'end': self.roi_end,
                    'label': label
                })
                self.roi_start = None
                self.roi_end = None
                self.roi_label_input.clear()
                self.save_rois_to_file()

    def clear_rois(self):
        self.rois.clear()
        self.save_rois_to_file()

    def save_rois_to_file(self):
        with open('roi_definitions.txt', 'w') as f:
            for roi in self.rois:
                f.write(f"{roi['label']},{roi['start'][0]},{roi['start'][1]},{roi['end'][0]},{roi['end'][1]}\n")

    def closeEvent(self, event):
        self.cap.release()

if __name__ == '__main__':
    app = QApplication([])
    window = ROIDefiner()
    window.show()
    app.exec_()
