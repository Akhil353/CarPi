import cv2
import time
import subprocess
import os


while True:
    time.wait(0.1)
    subprocess.Popen(["fswebcam", "input.jpg"])

    # Put detection code here once implemented

    for file in os.listdir('path/to/directory'):
        if file.startswith('input'):
            os.remove(file)
