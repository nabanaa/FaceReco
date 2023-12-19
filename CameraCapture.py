import numpy as np
import cv2
# import os
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time as t
from PIL import Image

def create_video_writer(video_cap, output_filename):

    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fcrame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer

video_cap = cv2.VideoCapture(0)
# change according to your operating system
# also you need to make all the emotion directories in pic_dir
pic_dir = '/home/piotrmika/Rok3/Python/pics/'

while True:
    ret, frame = video_cap.read()

    # if there are no more frames to process, break out of the loop
    if not ret:
        break

    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if not np.any(faces):
        continue
    # Draw rectangle around the faces
    # offset needed to get the full face
    offset = 30
    face = faces[0]
    (x, y, w, h) = face
    cv2.rectangle(frame, (x-offset, y-offset), (x+w+offset, y+h+offset), (255, 0, 0), 1)
    
    f_im = frame[y-offset:y+h+offset, x-offset:x+w+offset, :]
    back_color = (255,255,255)
    font_color = (0,0,0)
    cv2.rectangle(frame, (0, 0), (640, 20), back_color, -1)
    cv2.putText(frame, "Press (h)Happy, o(Ahegao), (a)Angry, (s)Surprise, (n)Neutral, (d)Sad, (q)uit", (12,15), cv2.QT_FONT_NORMAL, 0.5,font_color , 1)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    # user interface
    if key == ord('h'):
        cv2.imwrite(pic_dir+"/Happy/"+str(t.time())+'.jpg', f_im)
        print("Image saved!")
        
    if key == ord('o'):
        cv2.imwrite(pic_dir+"/Ahegao/"+str(t.time())+'.jpg', f_im)
        print("Image saved!")
    
    if key == ord('a'):
        cv2.imwrite(pic_dir+"/Angry/"+str(t.time())+'.jpg', f_im)
        print("Image saved!")

    if key == ord('s'):
        cv2.imwrite(pic_dir+"/Surprise/"+str(t.time())+'.jpg', f_im)
        print("Image saved!")

    if key == ord('n'):
        cv2.imwrite(pic_dir+"/Neutral/"+str(t.time())+'.jpg', f_im)
        print("Image saved!")

    if key == ord('d'):
        cv2.imwrite(pic_dir+"/Sad/"+str(t.time())+'.jpg', f_im)
        print("Image saved!")

    # Check for the 'q' key to exit the loop
    elif key == ord('q'):
        break


video_cap.release()
cv2.destroyAllWindows()
