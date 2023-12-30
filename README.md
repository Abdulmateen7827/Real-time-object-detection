# Real-Time Face Mask Detection System using YOLOv8

## Overview

This project implements a real-time face mask detection system using the YOLOv8 model. The system is designed to identify whether a person is wearing a face mask or not in real-time video streams.

## Features

- Real-time face mask detection in video streams.
- Uses YOLOv8 for accurate and efficient object detection.
- Minimal setup required for deployment.

## Requirements

- Python 3.x
- [YOLOv8](https://github.com/WongKinYiu/yolov8) model weights
- Additional Python packages (specified in requirements.txt)

## Installation

1. Create a virtual env
2. Clone the repository
3. Install the dependencies:
`pip install -r requirements.txt`
4. Add file path to the video you want to predict in the main.py file
5. Run in the terminal:
`python main.py`