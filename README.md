# Lane Detection System (Python, OpenCV)

A simple computer vision project for detecting lane lines in dashcam videos using classical image processing with Canny edge detection and Hough transform.

## Goal

- Detect lane lines in a video stream.
- Visualize the detected lanes on top of the video.
- Provide a clean, understandable pipeline in Python using OpenCV.

## Techniques Used

- Grayscale conversion
- Gaussian blur for smoothing / noise reduction
- Canny edge detection
- Region-of-interest masking (road area)
- Hough line transform for line detection
- Simple filtering and averaging of left / right lane lines
- Overlay of detected lanes on the original frame

## Installation

```bash
git clone https://github.com/<your-username>/lane-detection-opencv.git
cd lane-detection-opencv

# Optional virtual environment
# python -m venv .venv
# source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate     # Windows

pip install -r requirements.txt
