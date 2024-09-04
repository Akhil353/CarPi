import cv2
import time
import subprocess
import os
from ultralytics import YOLO, solutions


model = YOLO("yolov8n.pt")
names = model.model.names

video_path = "video.mp4"
video = cv2.VideoCapture(0)
w, h, fps = (int(video.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("distance_calculation.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
dist_obj = solutions.DistanceCalculation(names=names, view_img=True)

while video.isOpened():
    # Read a frame from the video
    success, frame = video.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        tracks = model.track(frame, persist=True, show=True)
        frame = dist_obj.start_process(frame, tracks)
        video_writer.write(frame)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

video.release()
video_writer.release()
cv2.destroyAllWindows()