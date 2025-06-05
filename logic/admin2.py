import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QTableWidget, QTableWidgetItem, QSizePolicy)
from PyQt5.QtCore import Qt, QSize

class ProcessDefiner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Admin Mode 2 - Process Definition")
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowFlags(self.windowFlags() | Qt.WindowCloseButtonHint)

        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Add spacing at the top
        self.layout.addSpacing(10)

        # Add title with consistent styling
        self.title_label = QLabel("Process Sequence Definition System")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
            margin: 10px;
            padding: 10px;
        """)
        self.layout.addWidget(self.title_label)

        # ROI Selection Area with improved styling
        roi_selection_widget = QWidget()
        roi_selection_widget.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin: 10px;
                padding: 15px;
            }
        """)
        roi_selection_layout = QVBoxLayout(roi_selection_widget)

        # ROI List Label and Combo with styling
        self.roi_list_label = QLabel("Available Regions of Interest:")
        self.roi_list_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        """)
        
        self.roi_list_combo = QComboBox()
        self.roi_list_combo.setStyleSheet("""
            QComboBox {
                padding: 8px;
                border: 2px solid #bdc3c7;
                border-radius: 6px;
                font-size: 14px;
                min-width: 200px;
            }
            QComboBox:hover {
                border: 2px solid #3498db;
            }
        """)
        
        roi_selection_layout.addWidget(self.roi_list_label)
        roi_selection_layout.addWidget(self.roi_list_combo)
        self.layout.addWidget(roi_selection_widget)

        # Process Table with improved styling
        self.process_table = QTableWidget(0, 1)
        self.process_table.setHorizontalHeaderLabels(["From ROI"])

        self.process_table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin: 10px;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QHeaderView::section {
                background-color: #f8f9fa;
                padding: 8px;
                border: 1px solid #dee2e6;
                font-weight: bold;
                font-size: 14px;
            }
        """)
        self.process_table.horizontalHeader().setStretchLastSection(True)
        self.process_table.setMinimumHeight(300)
        self.layout.addWidget(self.process_table)

        # Button layout with consistent styling
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
        
        self.add_step_button = QPushButton("Add Step")
        self.add_step_button.setStyleSheet(button_style)
        
        self.save_button = QPushButton("Save Process")
        self.save_button.setStyleSheet(button_style)
        
        self.clear_button = QPushButton("Clear Process")
        self.clear_button.setStyleSheet(button_style.replace("#3498db", "#e74c3c")
                                                   .replace("#2980b9", "#c0392b"))
        
        self.button_layout.addWidget(self.add_step_button)
        self.button_layout.addWidget(self.save_button)
        self.button_layout.addWidget(self.clear_button)
        self.layout.addLayout(self.button_layout)

        # Connect buttons
        self.add_step_button.clicked.connect(self.add_process_step)
        self.save_button.clicked.connect(self.save_process)
        self.clear_button.clicked.connect(self.clear_process)

        # Load ROI definitions
        self.load_roi_definitions()

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
        row_count = self.process_table.rowCount()
        self.process_table.insertRow(row_count)

        # Create and style comboboxes
        combo_style = """
            QComboBox {
                padding: 5px;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                min-width: 120px;
            }
            QComboBox:hover {
                border: 1px solid #3498db;
            }
        """

        from_roi_combo = QComboBox()
        from_roi_combo.addItems([roi['label'] for roi in self.roi_definitions])
        from_roi_combo.setStyleSheet(combo_style)
        self.process_table.setCellWidget(row_count, 0, from_roi_combo)

    def save_process(self):
        process_steps = []
        for row in range(self.process_table.rowCount()):
            from_roi = self.process_table.cellWidget(row, 0).currentText()
            process_steps.append(f"{from_roi}")

        
        with open('process_definitions.txt', 'w') as f:
            f.write('\n'.join(process_steps))

    def clear_process(self):
        self.process_table.setRowCount(0)

if __name__ == '__main__':
    app = QApplication([])
    window = ProcessDefiner()
    window.show()
    app.exec_()
