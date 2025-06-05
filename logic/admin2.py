import cv2                     # still imported in case you extend later
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QComboBox, QTableWidget, QTableWidgetItem, QSizePolicy
)
from PyQt5.QtCore import Qt, QSize


class ProcessDefiner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Admin Mode 2 – Process Definition")
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowFlags(self.windowFlags() | Qt.WindowCloseButtonHint)

        # ───────────────────────  MAIN LAYOUT  ───────────────────────
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addSpacing(10)

        # Title
        self.title_label = QLabel("Process Sequence Definition System")
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

        # ───────────────────────  ROI SELECTION  ───────────────────────
        roi_selection_widget = QWidget()
        roi_selection_widget.setStyleSheet("""
            QWidget {
                background-color: #12121a;
                border: 2px solid #2d2d3d;
                border-radius: 8px;
                margin: 10px;
                padding: 15px;
            }
        """)
        roi_selection_layout = QVBoxLayout(roi_selection_widget)

        self.roi_list_label = QLabel("Available Regions of Interest:")
        self.roi_list_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: 600;
                color: #ecf0f1;
                margin-bottom: 5px;
            }
        """)

        self.roi_list_combo = QComboBox()
        self.roi_list_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                font-size: 14px;
                min-width: 200px;
                color: #ecf0f1;
                background-color: #1e1e2e;
                border: 2px solid #2d2d3d;
                border-radius: 6px;
            }
            QComboBox:hover, QComboBox:focus {
                border: 2px solid #3498db;
            }
            QComboBox QListView {
                background-color: #1e1e2e;
                color: #ecf0f1;
            }
        """)

        roi_selection_layout.addWidget(self.roi_list_label)
        roi_selection_layout.addWidget(self.roi_list_combo)
        self.layout.addWidget(roi_selection_widget)

        # ───────────────────────  PROCESS TABLE  ───────────────────────
        self.process_table = QTableWidget(0, 1)
        self.process_table.setHorizontalHeaderLabels(["From ROI"])
        self.process_table.horizontalHeader().setStretchLastSection(True)
        self.process_table.setMinimumHeight(300)

        self.process_table.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e2e;
                border: 2px solid #2d2d3d;
                border-radius: 8px;
                margin: 10px;
                color: #ecf0f1;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #2d2d3d;
                color: #ecf0f1;
                padding: 8px;
                border: 1px solid #2d2d3d;
                font-weight: 600;
                font-size: 14px;
            }
        """)
        self.layout.addWidget(self.process_table)

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
        self.add_step_button = QPushButton("Add Step")
        self.add_step_button.setStyleSheet(
            base_btn + "QPushButton { background-color: #27ae60; }"
                        "QPushButton:hover { background-color: #1f8c4d; }")

        self.save_button = QPushButton("Save Process")
        self.save_button.setStyleSheet(
            base_btn + "QPushButton { background-color: #3498db; }"
                        "QPushButton:hover { background-color: #2980b9; }")

        self.clear_button = QPushButton("Clear Process")
        self.clear_button.setStyleSheet(
            base_btn + "QPushButton { background-color: #e74c3c; }"
                        "QPushButton:hover { background-color: #c0392b; }")

        self.button_layout.addWidget(self.add_step_button)
        self.button_layout.addWidget(self.save_button)
        self.button_layout.addWidget(self.clear_button)
        self.layout.addLayout(self.button_layout)

        # ───────────────────────  SIGNALS  ───────────────────────
        self.add_step_button.clicked.connect(self.add_process_step)
        self.save_button.clicked.connect(self.save_process)
        self.clear_button.clicked.connect(self.clear_process)

        # Load ROI data
        self.load_roi_definitions()

    # ───────────────────────  ROI + PROCESS HELPERS  ───────────────────────
    def load_roi_definitions(self):
        self.roi_definitions = []
        try:
            with open('roi_definitions.txt', 'r') as f:
                for line in f:
                    label, x1, y1, x2, y2 = line.strip().split(',')
                    self.roi_definitions.append({
                        'label': label,
                        'start': (int(x1), int(y1)),
                        'end': (int(x2), int(y2))
                    })
                    self.roi_list_combo.addItem(label)
        except FileNotFoundError:
            pass

    def add_process_step(self):
        row = self.process_table.rowCount()
        self.process_table.insertRow(row)

        combo_style = """
            QComboBox {
                padding: 6px;
                min-width: 140px;
                color: #ecf0f1;
                background-color: #1e1e2e;
                border: 1px solid #2d2d3d;
                border-radius: 4px;
            }
            QComboBox:hover, QComboBox:focus {
                border: 1px solid #3498db;
            }
            QComboBox QListView {
                background-color: #1e1e2e;
                color: #ecf0f1;
            }
        """
        from_combo = QComboBox()
        from_combo.addItems([roi['label'] for roi in self.roi_definitions])
        from_combo.setStyleSheet(combo_style)
        self.process_table.setCellWidget(row, 0, from_combo)

    def save_process(self):
        steps = []
        for row in range(self.process_table.rowCount()):
            steps.append(self.process_table.cellWidget(row, 0).currentText())
        with open('process_definitions.txt', 'w') as f:
            f.write('\n'.join(steps))

    def clear_process(self):
        self.process_table.setRowCount(0)


# ─────────────────────────────  ENTRY POINT  ─────────────────────────────
if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)

    # Global dark palette
    app.setStyleSheet("""
        QWidget {
            font-family: 'Segoe UI', sans-serif;
            background-color: #1e1e2e;
            color: #ecf0f1;
        }
    """)

    window = ProcessDefiner()
    window.show()
    sys.exit(app.exec_())
