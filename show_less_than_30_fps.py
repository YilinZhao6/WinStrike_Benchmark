import os
import cv2

def get_videos_with_low_fps(folder_path):
    # Get the list of video files in the folder
    video_files = [f for f in os.listdir(folder_path) if f.endswith((".mp4", ".avi", ".mov"))]

    low_fps_videos = []

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)

        # Open the video file
        video = cv2.VideoCapture(video_path)

        # Get the video frame rate
        fps = video.get(cv2.CAP_PROP_FPS)

        # Check if the frame rate is smaller than 29 fps
        if fps < 29:
            low_fps_videos.append(video_file)

        video.release()

    return low_fps_videos

# Example usage
folder_path = "C:/Users/ashis/OneDrive/Desktop/rnn/test_temp"
low_fps_videos = get_videos_with_low_fps(folder_path)

print("Videos with frame rate smaller than 29 fps:")
for video_file in low_fps_videos:
    print(video_file)