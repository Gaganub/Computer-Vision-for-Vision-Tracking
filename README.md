# Computer Vision for Vision Tracking

## Features
- Cross-platform
- Works with glasses
- Does not require high-end hardware, works well even with a 640*480 webcam
- Uses blob detection algorithm, but earlier versions used circle detection too.
- Highly extensible/flexible

## Requirements
- Python 3(will work with 2.7 if you install custom PyQT5 for it)
- PyQT 5(to install it for 2.7 use **pip install python-qt5** WARNING: Windows-only)
- OpenCV 3.4 +
- NumPy 1.15.2 +

## Guide
- To run: python main.py
- adjust thresholds for different lighting conditions(low for dark, high for bright)
- Detailed development guide: https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6

## Note: 
This repository has undergone several changes and updates, primarily focusing on training models and integrating YOLO and DeepSORT functionality. If you fork or clone this repository, you are advised to have all required frameworks and technologies installed to avoid any issues.

The user guide and the implementation with analysis through LLM will be published in Medium shortly
