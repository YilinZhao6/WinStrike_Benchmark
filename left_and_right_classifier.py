import os
import json
import shutil

def process_json_files(json_folder, video_folder, left_handed_folder, right_handed_folder):
    # Iterate over all JSON files in the specified folder
    for json_file in os.listdir(json_folder):
        if json_file.endswith(".json"):
            json_path = os.path.join(json_folder, json_file)
            
            # Read the JSON file
            with open(json_path, "r") as file:
                json_data = json.load(file)
            
            # Extract the video URL and choice from the JSON data
            for item in json_data:
                video_url = item["video_url"]
                choice = item["choice"]
                
                # Extract the video file name from the URL and remove the prefix
                video_file = os.path.basename(video_url).split("-", 1)[-1]
                
                # Check if the video file exists in the video folder
                video_path = os.path.join(video_folder, video_file)
                if os.path.exists(video_path):
                    # Determine the destination folder based on the choice
                    if choice == "Left-Hand Player":
                        destination_folder = left_handed_folder
                    else:
                        destination_folder = right_handed_folder
                    
                    # Move the video file to the corresponding folder
                    destination_path = os.path.join(destination_folder, video_file)
                    shutil.move(video_path, destination_path)
                    print(f"Moved {video_file} to {destination_folder}")
                else:
                    print(f"Video file {video_file} not found in {video_folder}")

# Specify the folder paths
json_folder = "C:/Users/ashis/OneDrive/Desktop/rnn/ground truth_test"
video_folder = "C:/Users/ashis/OneDrive/Desktop/rnn/video_conversion_output"
left_handed_folder = "C:/Users/ashis/OneDrive/Desktop/rnn/test_left_handed"
right_handed_folder = "C:/Users/ashis/OneDrive/Desktop/rnn/test_right_handed"

# Create the destination folders if they don't exist
os.makedirs(left_handed_folder, exist_ok=True)
os.makedirs(right_handed_folder, exist_ok=True)

# Process the JSON files and move the videos
process_json_files(json_folder, video_folder, left_handed_folder, right_handed_folder)