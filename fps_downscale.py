import os
import cv2
import shutil
from tqdm import tqdm
from termcolor import colored

def convert_video_to_30fps(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of video files in the input folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith((".mp4", ".avi", ".mov"))]

    total_videos = len(video_files)
    processed_videos = 0

    for video_file in video_files:
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, video_file)

        # Open the video file
        video = cv2.VideoCapture(input_path)

        # Get the video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        original_duration = total_frames / fps

        # Check if the frame rate is greater than 30 fps
        if fps > 30:
            print(f"Processing video: {video_file}")
            print(f"Original frame rate: {fps:.2f} fps")

            # Calculate the frame indices to keep based on the desired frame rate
            desired_fps = 30
            frame_indices = [int(i * fps / desired_fps) for i in range(int(total_frames * desired_fps / fps))]

            # Create a VideoWriter object for the output video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, desired_fps, (width, height))

            frame_count = 0
            progress_bar = tqdm(total=len(frame_indices), unit="frames", desc="Progress")

            while True:
                ret, frame = video.read()
                if not ret:
                    break

                if frame_count in frame_indices:
                    out.write(frame)
                    progress_bar.update(1)

                frame_count += 1

            progress_bar.close()
            out.release()

            # Get the duration of the converted video
            converted_video = cv2.VideoCapture(output_path)
            converted_frames = int(converted_video.get(cv2.CAP_PROP_FRAME_COUNT))
            converted_duration = converted_frames / desired_fps
            converted_video.release()

            print(f"Original video duration: {original_duration:.2f} seconds")
            print(f"Converted video duration: {converted_duration:.2f} seconds")

            # Check if the duration difference is more than 1%
            duration_diff = abs(original_duration - converted_duration) / original_duration * 100
            if duration_diff > 1:
                print(colored(f"Warning: Duration difference is {duration_diff:.2f}%", "red"))

            print(f"Video converted to {desired_fps} fps and saved as: {output_path}")
        else:
            # Copy the video file to the output folder without conversion
            shutil.copy2(input_path, output_path)
            print(f"Video: {video_file} - Frame rate is already {fps:.2f} fps or lower")
            print(f"Video copied to: {output_path}")

        video.release()
        processed_videos += 1
        print(f"Processed {processed_videos}/{total_videos} videos")
        print("---")

    print("All videos processed successfully!")

# Example usage
input_folder = "C:/Users/ashis/OneDrive/Desktop/rnn/test_temp"
output_folder = "C:/Users/ashis/OneDrive/Desktop/rnn/video_conversion_output"
convert_video_to_30fps(input_folder, output_folder)