import PySimpleGUI as sg
from time import time 
import cv2
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

#
# work in progress
#
# game is 'playable'
# 
# make faces to score points
#

# do zrobienia: ramka kraszuje program, ustawianie thresholdu od jakiego zalicza punkt

# HELPER COPY
def create_video_writer(video_cap, output_filename):

    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer

# FACE DET COPIES
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

with open("lite_emotions_model_efficientnet_b0.tflite", "rb") as f:
    lite_model_content = f.read()

interpreter = tf.lite.Interpreter(model_content=lite_model_content)
# This little helper wraps the TFLite Interpreter as a numpy-to-numpy function.
def lite_model(images):
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], images)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

# class names
class_names=["Ahegao", "Angry", "Happy", "Neutral", "Sad", "Surprised"]

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# VIDEO 
video_playing = False


# TIMER
start_time = 0
timer_active = False

# PAUSE BUTTON
pause_active = False

# PLAY BUTTON
start_active = False

# NEW_GAME BUTTON
show_new_game = False
new_game_button_added = False

# MAKE-A-FACE MECHANICS
prev_rolled_class = None
face_prompt_str = 'Face prompt: '
start_showing_face_prompts = False
new_face_prompt = False
current_rolled_class = ''
score_str = 'Score: '

score = 0

# GUI 
sg.theme('black')
layout = [
    [sg.Text('', key='-FACE_PROMPT-')],             # face to be made
    [sg.VPush()],                                   # blank space
    [sg.Image(key='-IMAGE-')],                      # game window with camera
    [sg.VPush()],                                   # blank space
    [sg.Text(score_str + str(score), key='-SCORE-')],    # score primitive
    [sg.Text('Time', key='-TIME-')],                # time row
    [
        sg.Button('Play', key='-START-'),
        sg.Button('Pause', key='-PAUSE-'),
        sg.Button('New Game', key='-NEW_GAME-',visible=False),
    ]          
    ]

window_size_x, window_size_y = 800, 600;
window = sg.Window('make-a-face!', layout, size=(window_size_x,window_size_y), element_justification='center')


# EVENTS
while True:
    event, values = window.read(timeout=10)
    
    if event == sg.WIN_CLOSED:
        break
    elif event == '-START-':
        start_time = time()
        timer_active = True
        show_new_game = True
        video_playing = True
        start_showing_face_prompts = True
        new_face_prompt = True
        window['-START-'].Update('Restart')

    elif event == '-PAUSE-':
        if pause_active == False:
            # pause
            window['-PAUSE-'].Update('Unpause')
            timer_active = False
            video_playing = False
            pause_active = True
        else:
            # unpause
            window['-PAUSE-'].Update('Pause')
            timer_active = True
            video_playing = True
            pause_active = False
    elif event == '-NEW_GAME-':
        score = 0
        timer_active = True
        new_face_prompt = True
    
        
    if timer_active == True:
        elapsed_time = round(time() - start_time, 1)
        window['-TIME-'].update(elapsed_time)
        
        
        
    if show_new_game == True and new_game_button_added == False:
        new_game_button_added = True
        window['-NEW_GAME-'].update(visible=True)
        
    if video_playing == True:
        ret, frame = video_cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display the output
        # print(faces[0])
        if not np.any(faces):
            continue
        (x, y, w, h) = faces[0]
        f_im = frame[x:x+w, y:y+h, :]
        # print(f_im.shape)
        H,W = interpreter.get_input_details()[0]['shape'][1:3]
        resized_face = cv2.resize(f_im, (W, H))
        def classify_face(face_img):
            return softmax(lite_model(face_img[None, ...].astype(np.float32)/255)[0])
        
        pred = classify_face(resized_face)
        pred.argmax()
        # print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")
        cv2.putText(frame, f"Prediction: {class_names[pred.argmax()]}: Certainty: {pred.max()*100}\%", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        # show the frame to our screen
        writer.write(frame)
        
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['-IMAGE-'].update(data=imgbytes)
        
        ### game mechanics
        print(current_rolled_class + "" + class_names[pred.argmax()])
        if current_rolled_class == class_names[pred.argmax()]:
            score += 1
            window['-SCORE-'].update(score_str + str(score))
            new_face_prompt = True
        
    if start_showing_face_prompts == True:
        if new_face_prompt == True:
            new_face_prompt = False
            rolled_class_idx = random.randint(0,5)
            while rolled_class_idx == prev_rolled_class:
                rolled_class_idx = random.randint(0,5)
            current_rolled_class = class_names[rolled_class_idx]
            prev_rolled_class = rolled_class_idx
            window['-FACE_PROMPT-'].update(face_prompt_str + current_rolled_class)
            

            

        
# close cv2
video_cap.release()
writer.release()
cv2.destroyAllWindows()

# close pysimplegui
window.close()