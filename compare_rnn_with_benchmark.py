import os
import json
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QAbstractItemView
from PyQt5.QtCore import Qt

class VideoAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Video Analysis App')
        self.setGeometry(100, 100, 800, 600)

        layout = QHBoxLayout()

        # Left side - RNN JSON files
        rnn_layout = QVBoxLayout()
        self.rnn_button = QPushButton('Read RNN JSON files from folder', self)
        self.rnn_button.clicked.connect(self.read_rnn_json_folder)
        self.rnn_table = QTableWidget()
        self.rnn_table.setColumnCount(3)
        self.rnn_table.setHorizontalHeaderLabels(['Video Name', 'Action Count', 'Actions'])
        self.rnn_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.rnn_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        rnn_layout.addWidget(self.rnn_button)
        rnn_layout.addWidget(self.rnn_table)

        # Right side - Ground Truth JSON files
        gt_layout = QVBoxLayout()
        self.gt_button = QPushButton('Import Ground Truth JSON folder', self)
        self.gt_button.clicked.connect(self.read_ground_truth_json_folder)
        self.gt_table = QTableWidget()
        self.gt_table.setColumnCount(3)
        self.gt_table.setHorizontalHeaderLabels(['Video Name', 'Action Count', 'Actions'])
        self.gt_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.gt_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        gt_layout.addWidget(self.gt_button)
        gt_layout.addWidget(self.gt_table)

        # Compare button
        compare_button = QPushButton('Compare RNN with Ground Truth', self)
        compare_button.clicked.connect(self.compare_results)

        # Add layouts to main layout
        layout.addLayout(rnn_layout)
        layout.addLayout(gt_layout)
        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(compare_button)
        self.setLayout(main_layout)

    def read_rnn_json_files(self, folder_path):
        json_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))

        self.rnn_table.setRowCount(len(json_files))
        for row, json_file in enumerate(json_files):
            with open(json_file, 'r') as file:
                data = json.load(file)
                for video in data:
                    video_name = os.path.splitext(os.path.basename(video['video_url']))[0]
                    action_count = len(video['tricks'])
                    actions = ', '.join([f"{trick['labels'][0]} ({trick['start']}s - {trick['end']}s)" for trick in video['tricks']])
                    self.rnn_table.setItem(row, 0, QTableWidgetItem(video_name))
                    self.rnn_table.setItem(row, 1, QTableWidgetItem(str(action_count)))
                    self.rnn_table.setItem(row, 2, QTableWidgetItem(actions))

    def read_ground_truth_json_files(self, folder_path):
        json_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))

        row_count = 0
        for json_file in json_files:
            with open(json_file, 'r') as file:
                data = json.load(file)
                row_count += len(data)

        self.gt_table.setRowCount(row_count)
        row = 0
        for json_file in json_files:
            with open(json_file, 'r') as file:
                data = json.load(file)
                for video in data:
                    video_name = os.path.splitext(os.path.basename(video['video_url']))[0]
                    action_count = len(video['tricks'])
                    actions = ', '.join([f"{trick['labels'][0]} ({trick['start']}s - {trick['end']}s)" for trick in video['tricks']])
                    self.gt_table.setItem(row, 0, QTableWidgetItem(video_name))
                    self.gt_table.setItem(row, 1, QTableWidgetItem(str(action_count)))
                    self.gt_table.setItem(row, 2, QTableWidgetItem(actions))
                    row += 1

    def read_rnn_json_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select RNN JSON Folder')
        if folder_path:
            self.read_rnn_json_files(folder_path)

    def read_ground_truth_json_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Ground Truth JSON Folder')
        if folder_path:
            self.read_ground_truth_json_files(folder_path)

    def compare_results(self):
        # Placeholder for comparing RNN results with ground truth
        pass

if __name__ == '__main__':
    app = QApplication([])
    window = VideoAnalysisApp()
    window.show()
    app.exec_()