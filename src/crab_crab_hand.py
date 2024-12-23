import cv2
import sys
import os

WORDS_PER_SECOND = 2  # Number of words spoken per second

def main():
    # Check if enough arguments are provided
    if len(sys.argv) < 2:
        print("\033[91mUsage: python program.py <video directory>\033[0m")
        sys.exit(1)

    # Access command line arguments
    argument = sys.argv[1]
    
    # Process the argument
    video_path = argument
    output_folder = 'temp/frames/'  # Folder to save the frames
    # Ensure the output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create a video capture object
    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    # Check if video file opened successfully
    if not cap.isOpened():
        print("Error: Cannot open video.")

    # Get the frames per second (FPS) of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames_per_word = fps // WORDS_PER_SECOND

    # extract frames
    while True:
        ret, frame = cap.read()  # Read one frame at a time
        if not ret:
            break  # Exit loop when no more frames are available
        
        # Save frame as an image
        if frame_count % frames_per_word == 0:
            frame_filename = f"{output_folder}frame_{frame_count:05d}.jpg"
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    print(f"Extracted {frame_count // frames_per_word + 1} frames.")
    cap.release()


if __name__ == "__main__":
    main()