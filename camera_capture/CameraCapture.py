#!/usr/bin/env python3

import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image

def create_video_writer(video_cap, output_filename):

    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.gt(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer


video_cap = cv2.VideoCapture(0)

# creating folders for faces
# they are created in the project folder
pic_dir = os.path.join(os.getcwd(), 'Faces')
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)
labels = ["Ahegao", "Sad", "Happy", "Surprise", "Angry", "Neutral"]
for label in labels:
    dir = os.path.join(pic_dir, label)
    if not os.path.exists(dir):
        os.makedirs(dir)

def list_files(path):
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    return onlyfiles

def save_file(folderName, f_im):
    from string import ascii_letters as alphabet
    from random import choice
    while True:
        fileName = ""
        for i in range(6):
            fileName+=choice(alphabet)
        fileName+=".jpg"
        print(fileName)
        path = os.path.join(pic_dir, folderName)
        files = list_files(path)
        check = True
        for file in files:
            if file==fileName:
                check=False
                break
        if check:
            break      
    cv2.imwrite(os.path.join(path, fileName), f_im)
    print("Image saved!")

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
while True:
    ret, frame = video_cap.read()

    # if there are no more frames to process, break out of the loop
    if not ret:
        break

    # Convert into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    anyFaces = np.any(faces)
    if anyFaces:
        # offset needed to get the full face
        offset = 35
        face = faces[0]
        (x, y, w, h) = face
        xw = frame.shape[1]
        yw = frame.shape[0]
        # making sure that we don't get out of bounds of frame
        P1 = (x-offset if x-offset >= 0 else 0, y-offset if y-offset >= 0 else 0)
        P2 = (x+w+offset if x+w+offset <= xw else xw, y+h+offset if y+h+offset <= yw else yw)
        cv2.rectangle(frame, P1, P2, (255, 0, 0), 1)
        f_im = frame[y-offset:y+h+offset, x-offset:x+w+offset, :]
        back_color = (255,255,255)
        font_color = (0,0,0)
        cv2.rectangle(frame, (0, 0), (640, 20), back_color, -1)
        cv2.putText(frame, "Press (h)Happy, o(Ahegao), (a)Angry, (s)Surprise, (n)Neutral, (d)Sad, (q)uit", (12,15), cv2.QT_FONT_NORMAL, 0.5,font_color , 1)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # user interface
    if key == ord('h') and f_im is not None:
        save_file("Happy", f_im)
        
    if key == ord('o') and f_im is not None:
        save_file("Ahegao", f_im)
    
    if key == ord('a') and f_im is not None:
        save_file("Angry", f_im)

    if key == ord('s') and f_im is not None:
        save_file("Surprise", f_im)

    if key == ord('n') and f_im is not None:
        save_file("Neutral", f_im)

    if key == ord('d') and f_im is not None:
        save_file("Sad", f_im)

    # Check for the 'q' key to exit the loop
    elif key == ord('q'):
        break


video_cap.release()
cv2.destroyAllWindows()