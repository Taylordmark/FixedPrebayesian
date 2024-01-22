import cv2
import os

def extract_frames(video_path, output_folder, skip_frames=0):
    """Extracts frames from a video and saves them as PNG images in a specified folder.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Path to the output folder where frames will be saved.
        skip_frames (int, optional): Number of frames to skip between saving. Defaults to 0 (save every frame).
    """

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video capture object
    cap = cv2.VideoCapture(video_path)

    # Get the video frame count (approximate)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process each frame
    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break  # End of video

        if frame_count % (skip_frames + 1) == 0:
            # Save the frame as a PNG image
            frame_name = f"frame_{frame_count:04d}.png"  # Pad frame count with zeros for consistent format
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()

    print(f"Extracted {frame_count} frames from video '{video_path}' to folder '{output_folder}'")

# Example usage:
video_path = r"C:\Users\keela\Documents\Video Outputs\0000f77c-6257be58\0000f77c-6257be58.mov"
output_folder = r"C:\Users\keela\Documents\Video Outputs\0000f77c-6257be58\frames"
skip_frames = 0  # Save every 10th frame

extract_frames(video_path, output_folder, skip_frames)
