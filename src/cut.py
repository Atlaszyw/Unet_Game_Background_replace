import cv2
import os

def split_video_frames(video_path, output_folder):
    # Read video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through frames and save them as individual images
    for i in range(frame_count):
        # Read frame
        ret, frame = cap.read()

        # Skip frame if read failed
        if not ret:
            continue

        # Construct output filename
        filename = f"{output_folder}/frame_{i:06d}.jpg"

        # Save frame as image
        cv2.imwrite(filename, frame)

    # Release video capture
    cap.release()
    
split_video_frames("Maliao_demo/video/sky.mp4", "Maliao_demo/video/sky")
split_video_frames("Maliao_demo/video/space.mp4", "Maliao_demo/video/space")