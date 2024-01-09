#!/usr/bin/env python3

import cv2
from helper import create_video_writer
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

def softmax(x):
    exps = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exps / np.sum(exps)

# define some constants
CONFIDENCE_THRESHOLD = 0.7
GREEN = (0, 255, 0)

# initialize the video capture object
video_cap = cv2.VideoCapture(0)
# initialize the video writer object
writer = create_video_writer(video_cap, "output.mp4")
# models is where your model resides
# program has to be run through terminal in the main project folder
model_path = os.path.join(os.getcwd(), 'models', 'lite_emotions_model_efficientnet_b0.tflite')
with open(model_path, "rb") as f:
    lite_model_content = f.read()

interpreter = tf.lite.Interpreter(model_content=lite_model_content)
# This little helper wraps the TFLite Interpreter as a numpy-to-numpy function.
def lite_model(images):
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], images)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

def classify_face(face_img):
        return softmax(lite_model(face_img[None, ...].astype(np.float32)/255)[0])

# class names
class_names=["Ahegao", "Angry", "Happy", "Neutral", "Sad", "Surprised"]

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

    anyFaces = np.any(faces)
    if anyFaces:
        # Draw rectangle around the faces
        offset = 35
        face = faces[0]
        (x, y, w, h) = face
        xw = frame.shape[1]
        yw = frame.shape[0]
        P1 = (x-offset if x-offset >= 0 else 0, y-offset if y-offset >= 0 else 0)
        P2 = (x+w+offset if x+w+offset <= xw else xw, y+h+offset if y+h+offset <= yw else yw)
        cv2.rectangle(frame, P1, P2, (255, 0, 0), 1)
        f_im = frame[P1[0]:P2[0], P1[1]:P2[1], :]
        H,W = interpreter.get_input_details()[0]['shape'][1:3]
        resized_face = cv2.resize(f_im, (W, H))
    
        pred = classify_face(resized_face)
        pred.argmax()
        # print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")
        cv2.putText(frame, f"Prediction: {class_names[pred.argmax()]} Certainty: {round(pred.max()*100,2)}%", (15,40), cv2.QT_FONT_NORMAL, 1, (0, 0, 255), 2)
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
writer.release()
cv2.destroyAllWindows()