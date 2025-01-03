import sys
import random
import re
import string
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tempfile import NamedTemporaryFile

import cv2
from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

AVG_WORDS_PER_SECOND = 2  # Number of words spoken per second
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
    process_descriptions(images)
    add_tts_audio(images, video_path)


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
        outputs = model.generate(**inputs, repetition_penalty=1.5)
        description = processor.decode(outputs[0], skip_special_tokens=True)
        if description.startswith(prompt):
            description = description[len(prompt):].strip()
        print(f"Description {input_number}/{len(images)}: {description}")
        image[2] = description
        input_number += 1

    # return images


def process_descriptions(images):
    # Create a list of image descriptions
    descriptions = [image[2] for image in images]

    words = []

    for i in range(len(descriptions)):
        cleaned_text = re.sub(f"[{string.punctuation}]", "", descriptions[i])
        word_list = cleaned_text.split()
        word_list = [word for word in word_list if len(word) > 1]   # Remove single character words

        if (i != 0 and descriptions[i] == descriptions[i-1]) or len(word_list) == 0:
            words.append(words[-1])
        else:
            words.append(word_list[random.randint(0, len(word_list) - 1)])

    for i in range(len(images)):
        images[i][2] = words[i]


def add_tts_audio(images, video_path):
    video = VideoFileClip(video_path)
    audios = []

    for image in images:
        if random.random() < 0.1:
            tts = gTTS(text=image[2]+"?", lang="en")
        else:
            tts = gTTS(text=image[2], lang="en")
        # Use NamedTemporaryFile to avoid overwriting
        with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_filename = temp_file.name
            tts.save(temp_filename)
            audio = AudioFileClip(temp_filename)
            audio = audio.with_start(image[0] / video.fps)
            audios.append(audio)
            
            # Remove the temporary file after processing
            os.remove(temp_filename)


    combined_audio = CompositeAudioClip(audios)
    combined_audio = combined_audio.subclipped(0, video.duration)
    video = video.with_audio(combined_audio)
    video.write_videofile("output.mp4", codec="libx264", audio_codec="aac")

        

if __name__ == "__main__":
    main()