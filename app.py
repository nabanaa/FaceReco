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

pause_counter_time = 0
pause_counter_timer_active = False

# PAUSE BUTTON
pause_active = False

# PLAY BUTTON
start_active = False

# NEW_GAME BUTTON
show_new_game = False
new_game_button_added = False

# PLAYER_NAME INPUT
show_player_name_input = True

# MAKE-A-FACE MECHANICS
prev_rolled_class = None
face_prompt_str = 'Face prompt: '
start_showing_face_prompts = False
new_face_prompt = False
current_rolled_class = ''
score_str = 'Score: '

#"name": highscore
player_data_dict = {}
current_player = None

score = 0

# GUI 
sg.theme('black')
# sg.set_options(button_element_size=(6,3))
layout = [
    [sg.Text('', key='-FACE_PROMPT-')],             # face to be made
    [sg.VPush()],                                   # blank space
    [sg.Image(key='-IMAGE-')],                      # game window with camera
    [sg.VPush()],                                   # blank space
    [sg.Text(score_str + str(score), key='-SCORE-')],    # score primitive
    [sg.Text('Time', key='-TIME-')],                # time row
    [
        sg.Button('Play', key='-START-',size=(7,3)),
        sg.Button('Pause', key='-PAUSE-',size=(7,3),visible=False),
        # new_game = restart + change player
        sg.Button('New Game', key='-NEW_GAME-',visible=False,size=(7,3)),
        sg.Input('Player', key='-PLAYER_NAME-',visible=True,size=(20,2))
    ]          
    ]

window_size_x, window_size_y = 800, 650;
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
        score = 0 # doesnt work
        window['-SCORE-'].Update(score_str + str(score))
        window['-PLAYER_NAME-'].Update(visible=False)
        if window['-START-'].get_text() == "Play":
            window['-PAUSE-'].Update(visible=True)
            window['-START-'].Update('Restart')
            
            ### select current player, initialize player data array
            if player_data_dict.get(values['-PLAYER_NAME-']) == None:
                player_data_dict.update({values['-PLAYER_NAME-']:[]})
            current_player = values['-PLAYER_NAME-']
        
    elif event == '-PAUSE-':
        if pause_active == False:
            # pause
            window['-PAUSE-'].Update('Unpause')
            timer_active = False
            video_playing = False
            pause_active = True
            pause_counter_time = time()            
        else:
            # unpause
            window['-PAUSE-'].Update('Pause')
            timer_active = True
            video_playing = True
            pause_active = False
            
            # pause timer - counting during pause prevention
            pause_counter_time = time() - pause_counter_time 
            ### the problem is that it gets overwritten in the if
            elapsed_time = elapsed_time - pause_counter_time
            pause_counter_time = 0
    elif event == '-NEW_GAME-':
        start_time = time()
        timer_active = True
        show_new_game = True
        video_playing = True
        start_showing_face_prompts = True
        new_face_prompt = True
        window['-START-'].Update('Restart')
        score = 0 # doesnt work
        window['-SCORE-'].Update(score_str + str(score))
        
        ### the new game button triggers pause if it was not triggered before
        ### and it also shows the player name input again, so the player can be changed 
        if pause_active == False:
            # pause
            window['-PAUSE-'].Update('Unpause')
            timer_active = False
            video_playing = False
            pause_active = True
            
        # show player name input
        window['-PLAYER_NAME-'].Update(visible = True)
        start_time = time()
        window['-START-'].Update('Play')
        window['-PAUSE-'].Update(visible=False)
        window['-NEW_GAME-'].Update(visible=False)
        
        
        
    if timer_active == True:
        ### WIP
        elapsed_time = round(time() - start_time, 1)
        window['-TIME-'].update(elapsed_time)
         
        
    if show_new_game == True and new_game_button_added == False:
        new_game_button_added = True
        window['-NEW_GAME-'].update(visible=True)
        
    if video_playing == True:
        ret, frame = video_cap.read()
        
        
        ### copied from faceDet 
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
        ### end of face det copy
        
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['-IMAGE-'].update(data=imgbytes)
        
        ### game mechanics
        # print(current_rolled_class + "" + class_names[pred.argmax()])
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