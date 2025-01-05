# **Crab Crab Hand**

This script automatically generates videos in the style of this [video](https://www.youtube.com/watch?v=wIfvcWCZZ7w)

## **Dependencies**
Install required packages with:
```bash
pip install opencv-python Pillow gTTS moviepy transformers torch
```

## **Usage**
Run the script from the terminal:
```
python src/crab-crab-hand.py <input_video_path> [output_video_path] ["prompt for image recognition"]
```
- **`<input_video_path>`**: Input video file path.  
- **`[output_video_path]`** (optional): Output video file path (default: `output.mp4`).