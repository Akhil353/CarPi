import cv2
import time
import subprocess
import os
from ultralytics import YOLO


model = YOLO("yolov8n.pt")

video_path = "video.mp4"
video = cv2.VideoCapture(0)

while video.isOpened():
    # Read a frame from the video
    success, frame = video.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        cv2.waitKey(1)

