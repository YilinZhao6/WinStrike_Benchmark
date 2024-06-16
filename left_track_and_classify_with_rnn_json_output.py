import time
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
import cv2
from collections import deque
import os
from extract_human_pose import HumanPoseExtractor
from pathlib import Path
import json

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is being used:", physical_devices[0])
else:
    print("No GPU found. Using CPU.")

print("Num GPUs Available:", len(physical_devices))

columns = [
    "nose_y",
    "nose_x",
    "left_shoulder_y",
    "left_shoulder_x",
    "right_shoulder_y",
    "right_shoulder_x",
    "left_elbow_y",
    "left_elbow_x",
    "right_elbow_y",
    "right_elbow_x",
    "left_wrist_y",
    "left_wrist_x",
    "right_wrist_y",
    "right_wrist_x",
    "left_hip_y",
    "left_hip_x",
    "right_hip_y",
    "right_hip_x",
    "left_knee_y",
    "left_knee_x",
    "right_knee_y",
    "right_knee_x",
    "left_ankle_y",
    "left_ankle_x",
    "right_ankle_y",
    "right_ankle_x",
]

class ShotCounter:
    """
    Pretty much the same principle than in track_and_classify_frame_by_frame
    except that we dont have any history here, and confidence threshold can be much higher.
    """

    MIN_FRAMES_BETWEEN_SHOTS = 60

    def __init__(self):
        self.nb_history = 30
        self.probs = np.zeros(4)

        self.nb_forehands = 0
        self.nb_backhands = 0
        self.nb_serves = 0

        self.last_shot = "neutral"
        self.frames_since_last_shot = self.MIN_FRAMES_BETWEEN_SHOTS

        self.results = []
        
    def update(self, probs, frame_id):
        """Update current state with shot probabilities"""

        if len(probs) == 4:
            self.probs = probs
        else:
            self.probs[0:3] = probs

        if (
            probs[0] > 0.98
            and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS
        ):
            self.nb_backhands += 1
            self.last_shot = "backhand"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
        elif (
            probs[1] > 0.98
            and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS
        ):

            self.nb_forehands += 1
            self.last_shot = "forehand"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
        elif (
            len(probs) > 3
            and probs[3] > 0.98
            and self.frames_since_last_shot > self.MIN_FRAMES_BETWEEN_SHOTS
        ):
            self.nb_serves += 1
            self.last_shot = "serve"
            self.frames_since_last_shot = 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})

        self.frames_since_last_shot += 1


def compute_recall_precision(gt, shots):
    """
    Assess your results against a Groundtruth
    like number of misses (recall) and number of false positives (precision)
    """

    gt_numpy = gt.to_numpy()
    nb_match = 0
    nb_misses = 0
    nb_fp = 0
    fp_backhands = 0
    fp_forehands = 0
    fp_serves = 0
    for gt_shot in gt_numpy:
        found_match = False
        for shot in shots:
            if shot["Shot"] == gt_shot[0]:
                if abs(shot["FrameID"] - gt_shot[1]) <= 30:
                    found_match = True
                    break
        if found_match:
            nb_match += 1
        else:
            nb_misses += 1

    for shot in shots:
        found_match = False
        for gt_shot in gt_numpy:
            if shot["Shot"] == gt_shot[0]:
                if abs(shot["FrameID"] - gt_shot[1]) <= 30:
                    found_match = True
                    break
        if not found_match:
            nb_fp += 1
            if shot["Shot"] == "backhand":
                fp_backhands += 1
            elif shot["Shot"] == "forehand":
                fp_forehands += 1
            elif shot["Shot"] == "serve":
                fp_serves += 1

    precision = nb_match / (nb_match + nb_fp)
    recall = nb_match / (nb_match + nb_misses)

    print(f"Recall {recall*100:.1f}%")
    print(f"Precision {precision*100:.1f}%")

    print(
        f"FP: backhands = {fp_backhands}, forehands = {fp_forehands}, serves = {fp_serves}"
    )

class VideoClipMaker: 
    VIDEO_FRAMES = 60
    
    def __init__(self, fps, frame_width, frame_height, folder_path):
        self.fps = fps 
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.counter = 1
        self.frame_buffer = deque(maxlen=self.VIDEO_FRAMES)
        self.feature_buffer = deque(maxlen=self.VIDEO_FRAMES)
        self.folder_path = folder_path
        self.json_data = {}  # Add this line to initialize json_data attribute

    
    def addFrameAndFeature(self, frame, feature): 
        self.frame_buffer.append(frame)
        self.feature_buffer.append(feature)
        
    def translateLastFrameToBeginTime(self, frame, to_seconds=False):
        total_seconds = (frame - self.VIDEO_FRAMES) / self.fps
        if to_seconds:
            return total_seconds
        minutes, seconds = divmod(total_seconds, 60)
        return f"{int(minutes):02}-{int(seconds):02}"
    
    def createVideoClip(self, frame_id, shot_type):
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        clip_writer = cv2.VideoWriter_fourcc(*'MJPG')

        file_name = f'{self.folder_path}/clip_{self.counter}_{self.translateLastFrameToBeginTime(frame_id)}.mp4'
        print(f"fps is {self.fps} and frame_size is ({self.frame_width},{self.frame_height})")
        out = cv2.VideoWriter(
            file_name,
            clip_writer,
            self.fps,
            (self.frame_width, self.frame_height))

        for frame in self.frame_buffer:
            out.write(frame)
        out.release()
        print(f"Has create {file_name}.")

        shots_df = pd.DataFrame(
            np.concatenate(self.feature_buffer, axis=0),
            columns=columns
        )
        shots_df["shot"] = np.full(self.VIDEO_FRAMES, shot_type)
        outpath = Path(self.folder_path).joinpath(f"clip_{self.counter}_{shot_type}.csv")

        outpath.parent.mkdir(parents=True, exist_ok=True)

        shots_df.to_csv(outpath, index=False)
        print(f"saving csv to {outpath}")

        # Generate JSON data
        start_seconds = self.translateLastFrameToBeginTime(frame_id, to_seconds=True)
        end_seconds = self.translateLastFrameToBeginTime(frame_id, to_seconds=True) + 2

        trick_data = {
            "start": start_seconds,
            "end": end_seconds,
            "channel": 0,
            "labels": [shot_type.capitalize() + " Strike"]
        }

        if "tricks" not in self.json_data:
            self.json_data["tricks"] = []

        self.json_data["tricks"].append(trick_data)

        self.counter += 1

    def saveJsonFiles(self):
        for video_id, json_data in self.json_data.items():
            json_file_name = f'{self.folder_path}/video_{video_id}_timestamp.json'
            with open(json_file_name, 'w') as json_file:
                json.dump([json_data], json_file, indent=2)
            print(f"saving json to {json_file_name}")

    def convertTimeToSeconds(self, time_string):
            minutes, seconds = map(int, time_string.split('-'))
            return minutes * 60 + seconds

def scan_through_folder(src, dest, m1, left_handed):
    # Ensure source is a valid directory
    if not os.path.exists(src):
        print("Source directory does not exist.")
        return
    
    # Ensure destination directory exists, create if not
    if not os.path.exists(dest):
        os.makedirs(dest)
        print(f"Destination directory {dest} created.")

    for file in os.listdir(src):
        if file.endswith(".mp4"):
            source_file = os.path.join(src, file)
            dest_final_folder = os.path.join(dest, os.path.splitext(file)[0])
            if not os.path.exists(dest_final_folder):
                os.makedirs(dest_final_folder)
            video_clip_maker = process_file(m1, source_file, dest_final_folder, left_handed)
            
            # Save the combined JSON file for the video
            json_file_name = f'{dest_final_folder}/{os.path.splitext(file)[0]}_timestamp.json'
            with open(json_file_name, 'w') as json_file:
                json.dump([video_clip_maker.json_data], json_file, indent=2)
            print(f"saving json to {json_file_name}")

            

def process_file(m1, video_file_path, dest_path, left_handed):
    shot_counter = ShotCounter()
    
    cap = cv2.VideoCapture(video_file_path)

    assert cap.isOpened()

    ret, frame = cap.read()

    human_pose_extractor = HumanPoseExtractor(frame.shape)
    
    video_clip_maker = VideoClipMaker(
        30, 
        cap.get(cv2.CAP_PROP_FRAME_WIDTH), 
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        dest_path
    )
    video_clip_maker.video_file_path = video_file_path  # Assign the video file path

    video_clip_maker.json_data = {
    "video_url": os.path.basename(video_file_path),
    "id": 1,
    "tricks": []
    }

    NB_IMAGES = 30

    FRAME_ID = 0

    features_pool = []

    prev_time = time.time()

    num_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        num_frame += 1 
        corresponding_frame_id = round(num_frame * 30 / cv2.CAP_PROP_FPS) 
        if corresponding_frame_id < FRAME_ID :
            continue 
        FRAME_ID += 1
        
        total_second = FRAME_ID / 30
        minutes, seconds = divmod(total_second, 60)
        print(f"Start processing file {video_file_path} at {int(minutes):02}:{int(seconds):02}")

        assert frame is not None

        human_pose_extractor.extract(frame)

        human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])

        features = human_pose_extractor.keypoints_with_scores.reshape(17, 3)

        if left_handed:
            features[:, 1] = 1 - features[:, 1]

        features = features[features[:, 2] > 0][:, 0:2].reshape(1, 13 * 2)

        video_clip_maker.addFrameAndFeature(frame.copy(), features)
        features_pool.append(features)

        if len(features_pool) == NB_IMAGES:
            features_seq = np.array(features_pool).reshape(1, NB_IMAGES, 26)
            assert features_seq.shape == (1, 30, 26)
            probs = m1.__call__(features_seq)[0]
            shot_counter.update(probs, FRAME_ID)
            
            if shot_counter.frames_since_last_shot == 30:
                start_time = video_clip_maker.translateLastFrameToBeginTime(FRAME_ID - 30)
                end_time = video_clip_maker.translateLastFrameToBeginTime(FRAME_ID)
                
                if shot_counter.last_shot == "forehand":
                    print(f"Detected forehand shot from {start_time} to {end_time}")
                    video_clip_maker.createVideoClip(FRAME_ID, "forehand")
                elif shot_counter.last_shot == "backhand":
                    print(f"Detected backhand shot from {start_time} to {end_time}")
                    video_clip_maker.createVideoClip(FRAME_ID, "backhand")
                elif shot_counter.last_shot == "serve":
                    print(f"Detected serve shot from {start_time} to {end_time}")
                    video_clip_maker.createVideoClip(FRAME_ID, "serve")
            
            features_pool = features_pool[1:]

        human_pose_extractor.roi.update(human_pose_extractor.keypoints_pixels_frame)

    cap.release()
    
    return video_clip_maker  # Return the video_clip_maker instance

#Deal with the left-handed situation

if __name__ == "__main__":
    source_folder = "C:/Users/ashis/OneDrive/Desktop/rnn/test_left_handed"
    model_file = "C:/Users/ashis/OneDrive/Desktop/rnn/tennis_rnn_rafa.keras"
    dest_folder = "C:/Users/ashis/OneDrive/Desktop/rnn/test_left_handed_output"
    left_handed = True  # Set to True if the player is left-handed, False otherwise

    m1 = keras.models.load_model(model_file)
    scan_through_folder(source_folder, dest_folder, m1, left_handed)


 