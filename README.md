**Real-Time Person Tracking System**

This project demonstrates a real-time person tracking system using YOLOv5 for object detection and DeepSORT for tracking. It allows the user to dynamically input a suspect ID while the video is playing, and the system highlights that suspect in red.

**View the sample video output:**[LINK](https://drive.google.com/file/d/1YTuOeDnwtM8yrQo5wsMnh2rXW0Q5rIop/view?usp=drive_link)

**Features**
- Consistent **ID assignment** using DeepSORT tracker
- Lightweight model for faster inference on low-end GPUs or CPUs
- Tracks multiple individuals simultaneously
- Visualization of bounding boxes and object IDs

**Tech Stack**
- Python 
- YOLOv5n
- DeepSORT
- OpenCV
