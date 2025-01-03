import sys
import os
import random

import cv2
# from gtts import gTTS
# from moviepy import VideoFileClip

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

AVG_WORDS_PER_SECOND = 3  # Number of words spoken per second
TIME_VARIATION_SECONDS = 0.1

def main():
    print("running...")

    # Check if enough arguments are provided
    if len(sys.argv) < 2:
        print("\033[91mUsage: python program.py <video directory>\033[0m")
        sys.exit(1)
    
    # Access command line arguments
    argument = sys.argv[1]
    video_path = argument

    images: list[list[int, Image.Image, str]] = [] # (frame count, image, image description)

    extract_images(video_path, images)
    generate_image_descriptions(images)


def extract_images(video_path, images):
    # Create a video capture object
    cap = cv2.VideoCapture(video_path)

    # Get the frames per second (FPS) of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    avg_frames_per_word = fps // AVG_WORDS_PER_SECOND
    frame_variation = int(fps * TIME_VARIATION_SECONDS)


    # generates a list of frame numbers to extract
    frame_numbers = [i for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), avg_frames_per_word)]
    for i in range(1, len(frame_numbers) -1):
        frame_numbers[i] += random.randint(-frame_variation, frame_variation)

    frame_count = 0

    # Check if video file opened successfully
    if not cap.isOpened():
        print("Error: Cannot open video.")

    # extract frames
    while True:
        ret, frame = cap.read()  # Read one frame at a time
        if not ret:
            break  # Exit loop when no more frames are available

        # Save frame as an image
        if frame_count in frame_numbers:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            images.append([frame_count, image, ""])

        frame_count += 1

    # images[1][1].show()

    cap.release()


def generate_image_descriptions(images):
    # Load the Blip model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    prompt = "one word description of image :"

    # Generate image descriptions
    input_number = 1
    for image in images:
        inputs = processor(images=image[1], text=prompt, return_tensors="pt")
        outputs = model.generate(**inputs)
        description = processor.decode(outputs[0], skip_special_tokens=True)
        if description.startswith(prompt):
            description = description[len(prompt):].strip()
        print(f"Description {input_number}/{len(images)}: {description}")
        image[2] = description
        input_number += 1

    # return images

if __name__ == "__main__":
    main()