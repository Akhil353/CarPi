import cv2
import time
import subprocess
import os
from model import NeuralNetwork


while True:
    # Take an image to be processed by neural network
    time.wait(0.1)
    subprocess.Popen(["fswebcam", "input.jpg"])

    # Put detection code here once implemented
    detector = NeuralNetwork()
    detector.load_model('model_file_name')
    action = detector.predict('input.jpg')


    for filename in os.listdir('path/to/directory'):
        if filename.startswith('input'):
            os.remove(filename)
