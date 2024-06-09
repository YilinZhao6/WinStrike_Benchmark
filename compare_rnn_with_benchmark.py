import os
import json
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QAbstractItemView, QProgressBar, QMessageBox
from PyQt5.QtCore import Qt
from datetime import datetime

class VideoAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.rnn_data = []
        self.gt_data = []
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

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(30, 40, 200, 25)

        # Add layouts to main layout
        layout.addLayout(rnn_layout)
        layout.addLayout(gt_layout)
        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(compare_button)
        main_layout.addWidget(self.progress_bar)
        self.setLayout(main_layout)

    def read_rnn_json_files(self, folder_path):
        json_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))

        self.rnn_table.setRowCount(len(json_files))
        self.rnn_data = []
        for row, json_file in enumerate(json_files):
            with open(json_file, 'r') as file:
                data = json.load(file)
                for video in data:
                    video_name = os.path.basename(video['video_url'])  # Include the file extension
                    action_count = len(video['tricks'])
                    actions = ', '.join([f"{trick['labels'][0]} ({trick['start']}s - {trick['end']}s)" for trick in video['tricks']])
                    self.rnn_table.setItem(row, 0, QTableWidgetItem(video_name))
                    self.rnn_table.setItem(row, 1, QTableWidgetItem(str(action_count)))
                    self.rnn_table.setItem(row, 2, QTableWidgetItem(actions))
                    video['video_name'] = video_name
                    self.rnn_data.append(video)

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
        self.gt_data = []
        row = 0
        for json_file in json_files:
            with open(json_file, 'r') as file:
                data = json.load(file)
                for video in data:
                    video_name = video['video_url'].split('-')[1]
                    action_count = len(video['tricks'])
                    actions = ', '.join([f"{trick['labels'][0]} ({trick['start']}s - {trick['end']}s)" for trick in video['tricks']])
                    self.gt_table.setItem(row, 0, QTableWidgetItem(video_name))
                    self.gt_table.setItem(row, 1, QTableWidgetItem(str(action_count)))
                    self.gt_table.setItem(row, 2, QTableWidgetItem(actions))
                    self.gt_data.append(video)
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
        if not self.rnn_data or not self.gt_data:
            QMessageBox.warning(self, "Warning", "Please import both RNN and Ground Truth data before comparing.")
            return

        rnn_video_names = set(video['video_name'] for video in self.rnn_data)
        gt_video_names = set(video['video_url'].split('-')[1] for video in self.gt_data)

        common_videos = rnn_video_names.intersection(gt_video_names)
        rnn_only_videos = rnn_video_names - gt_video_names
        gt_only_videos = gt_video_names - rnn_video_names

        matched_right_type_count = 0
        matched_wrong_type_count = 0
        mismatch_gt_count = 0
        mismatch_rnn_count = 0

        total_videos = len(common_videos)
        processed_videos = 0

        video_results = []

        for video_name in common_videos:
            rnn_video = next((video for video in self.rnn_data if video['video_name'] == video_name), None)
            gt_video = next((video for video in self.gt_data if video['video_url'].split('-')[1] == video_name), None)

            if rnn_video and gt_video:
                rnn_tricks = rnn_video['tricks']
                gt_tricks = gt_video['tricks']

                matched_tricks = []
                for gt_trick in gt_tricks:
                    for rnn_trick in rnn_tricks:
                        if gt_trick['start'] <= rnn_trick['end'] and rnn_trick['start'] <= gt_trick['end']:
                            matched_tricks.append((gt_trick, rnn_trick))
                            break

                matched_right_type = sum(1 for gt_trick, rnn_trick in matched_tricks if gt_trick['labels'] == rnn_trick['labels'])
                matched_wrong_type = len(matched_tricks) - matched_right_type
                mismatch_gt = len(gt_tricks) - len(matched_tricks)
                mismatch_rnn = len(rnn_tricks) - len(matched_tricks)

                matched_right_type_count += matched_right_type * 2
                matched_wrong_type_count += matched_wrong_type * 2
                mismatch_gt_count += mismatch_gt
                mismatch_rnn_count += mismatch_rnn

                video_results.append({
                    'video_name': video_name,
                    'gt_tricks': gt_tricks,
                    'rnn_tricks': rnn_tricks,
                    'matched_tricks': matched_tricks,
                    'matched_right_type': matched_right_type,
                    'matched_wrong_type': matched_wrong_type,
                    'mismatch_gt': mismatch_gt,
                    'mismatch_rnn': mismatch_rnn
                })
            else:
                video_results.append({
                    'video_name': video_name,
                    'error': 'Video not found in either RNN or Ground Truth data'
                })

            processed_videos += 1
            self.progress_bar.setValue(int(processed_videos / total_videos * 100))  # Convert to integer

        total_actions = matched_right_type_count + matched_wrong_type_count + mismatch_gt_count + mismatch_rnn_count

        html_content = f"""
        <html>
        <head>
            <title>Benchmark Export - {datetime.now().strftime('%m-%d-%H-%M')}</title>
        </head>
        <body>
            <h1>Benchmark Export</h1>
            <h2>Video Overview</h2>
            <p>Videos in both RNN and Ground Truth: {len(common_videos)}</p>
            <ul>
                {''.join(f'<li>{video}</li>' for video in common_videos)}
            </ul>
            <p>Videos only in RNN: {len(rnn_only_videos)}</p>
            <ul>
                {''.join(f'<li>{video}</li>' for video in rnn_only_videos)}
            </ul>
            <p>Videos only in Ground Truth: {len(gt_only_videos)}</p>
            <ul>
                {''.join(f'<li>{video}</li>' for video in gt_only_videos)}
            </ul>
        """

        if total_actions > 0:
            html_content += f"""
            <h2>Benchmark Statistics</h2>
            <p>Matched with Right Type: {matched_right_type_count} ({matched_right_type_count / total_actions * 100:.2f}%)</p>
            <p>Matched with Wrong Type: {matched_wrong_type_count} ({matched_wrong_type_count / total_actions * 100:.2f}%)</p>
            <p>Mismatch - Exists in Ground Truth, but not in RNN: {mismatch_gt_count} ({mismatch_gt_count / total_actions * 100:.2f}%)</p>
            <p>Mismatch - Exists in RNN, but not in Ground Truth: {mismatch_rnn_count} ({mismatch_rnn_count / total_actions * 100:.2f}%)</p>
            """
        else:
            html_content += "<h2>Benchmark Statistics</h2><p>No actions found to compare.</p>"

        for result in video_results:
            if 'error' in result:
                html_content += f"<h2>Video: {result['video_name']}</h2><p>{result['error']}</p>"
            else:
                html_content += f"""
                <h2>Video: {result['video_name']}</h2>
                <p>Marker Legend:</p>
                <ul>
                    <li><span style="color: blue;">✓</span> - Matched with the right type</li>
                    <li><span style="color: red;">✗</span> - Matched with the wrong type</li>
                    <li><span style="color: blue;">★</span> - Exists in one timeline but not the other</li>
                </ul>
                <p>Action Color Legend:</p>
                <ul>
                    <li><span style="color: #4B9CD3;">Forehand Strike</span></li>
                    <li><span style="color: orange;">Backhand Strike</span></li>
                    <li><span style="color: green;">Serve</span></li>
                </ul>
                <div style="display: flex;">
                    <div style="width: 50%;">
                        <h3>Ground Truth Timeline</h3>
                        {''.join(self.generate_timeline_html(result['gt_tricks'], result['matched_tricks'], True))}
                    </div>
                    <div style="width: 50%;">
                        <h3>RNN Timeline</h3>
                        {''.join(self.generate_timeline_html(result['rnn_tricks'], result['matched_tricks'], False))}
                    </div>
                </div>
                """

                total_actions = len(result['gt_tricks']) + len(result['rnn_tricks'])
                if total_actions > 0:
                    html_content += f"""
                    <p>Matched with Right Type: {result['matched_right_type'] * 2} ({result['matched_right_type'] * 2 / total_actions * 100:.2f}%)</p>
                    <p>Matched with Wrong Type: {result['matched_wrong_type'] * 2} ({result['matched_wrong_type'] * 2 / total_actions * 100:.2f}%)</p>
                    <p>Mismatch - Exists in Ground Truth, but not in RNN: {result['mismatch_gt']} ({result['mismatch_gt'] / len(result['gt_tricks']) * 100:.2f}%)</p>
                    <p>Mismatch - Exists in RNN, but not in Ground Truth: {result['mismatch_rnn']} ({result['mismatch_rnn'] / len(result['rnn_tricks']) * 100:.2f}%)</p>
                    """
                else:
                    html_content += "<p>No actions found in this video.</p>"

        html_content += "</body></html>"

        file_name = f"Benchmark_Export_{datetime.now().strftime('%m-%d-%H-%M')}.html"
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(html_content)

        QMessageBox.information(self, "Export Complete", f"Benchmark exported to {file_name}")

    def generate_timeline_html(self, tricks, matched_tricks, is_ground_truth):
        colors = {
            'Forehand Strike': '#4B9CD3',  # Columbia University's blue
            'Backhand Strike': 'orange',
            'Serve': 'green'
        }

        html_content = ""
        for trick in tricks:
            color = colors.get(trick['labels'][0], 'gray')  # Default color if action not found in colors
            if is_ground_truth:
                matched_trick = next((rnn_trick for gt_trick, rnn_trick in matched_tricks if gt_trick == trick), None)
            else:
                matched_trick = next((gt_trick for gt_trick, rnn_trick in matched_tricks if rnn_trick == trick), None)

            if matched_trick:
                if is_ground_truth:
                    if trick['labels'] == matched_trick['labels']:
                        marker = '<span style="color: white;">✓</span>'
                    else:
                        marker = '<span style="color: red;">✗</span>'
                else:
                    if trick['labels'] == next((gt_trick for gt_trick, rnn_trick in matched_tricks if rnn_trick == trick), None)['labels']:
                        marker = '<span style="color: white;">✓</span>'
                    else:
                        marker = ''
            else:
                marker = '<span style="color: yellow;">★</span>'
                
            html_content += f'<div style="background-color: {color}; margin-bottom: 5px; height: 30px; line-height: 30px;">{trick["labels"][0]} ({trick["start"]:.2f}s - {trick["end"]:.2f}s) {marker}</div>'

        return html_content
    
if __name__ == '__main__':
    app = QApplication([])
    window = VideoAnalysisApp()
    window.show()
    app.exec_()