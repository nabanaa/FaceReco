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

class MakeAFace():
   # class attributes
    
    # define some constants
    CONFIDENCE_THRESHOLD = 0.7
    GREEN = (0, 255, 0)
    
    # class names
    class_names=["Ahegao", "Angry", "Happy", "Neutral", "Sad", "Surprised"]
    
    # instance attributes
    def __init__(self):    
        # initialize the video capture object
        self.video_cap = cv2.VideoCapture(0)
        # initialize the video writer object
        self.writer = __class__.create_video_writer(self.video_cap, "output.mp4")

        with open("lite_emotions_model_efficientnet_b0.tflite", "rb") as f:
            self.lite_model_content = f.read()
            
        self.interpreter = tf.lite.Interpreter(model_content=self.lite_model_content)
        # Load the cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # VIDEO 
        self.video_playing = False

        # TIMER
        self.start_time = 0
        self.timer_active = False

        self.paused_correction_time = 0
        self.pause_counter_time = 0
        self.pause_counter_timer_active = False

        # PAUSE BUTTON
        self.pause_active = False

        # PLAY BUTTON
        self.start_active = False

        # NEW_GAME BUTTON
        self.show_new_game = False
        self.new_game_button_added = False

        # PLAYER_NAME INPUT
        self.show_player_name_input = True

        # MAKE-A-FACE MECHANICS
        self.prev_rolled_class = None
        self.face_prompt_str = 'Face prompt: '
        self.start_showing_face_prompts = False
        self.new_face_prompt = False
        self.current_rolled_class = ''
        self.score_str = 'Score: '

        #"name": highscore
        self.player_data_dict = {}
        self.current_player = None
        self.score = 0

            
        # STATES
        self.STATE_START_SC = 0
        self.STATE_PLAY_SC = 1
        self.STATE_PAUSE_SC = 2        

        # GUI 
        sg.theme('black')
        # sg.set_options(button_element_size=(6,3))
        self.layout = [
            [sg.Text('', key='-FACE_PROMPT-')],             # face to be made
            [sg.VPush()],                                   # blank space
            [sg.Image(key='-IMAGE-')],                      # game window with camera
            [sg.VPush()],                                   # blank space
            [sg.Text(self.score_str + str(self.score), key='-SCORE-')],    # score primitive
            [sg.Text('Time', key='-TIME-')],                # time row
            [
                sg.Button('Play', key='-START-',size=(7,3)),
                sg.Button('Pause', key='-PAUSE-',size=(7,3),visible=False),
                # new_game = restart + change player
                sg.Button('New Game', key='-NEW_GAME-',visible=False,size=(7,3)),
                sg.Input('Player', key='-PLAYER_NAME-',visible=True,size=(20,2))
            ]          
            ]
        
        self.window_size_x, self.window_size_y = 800, 650;
        self.window = sg.Window('make-a-face!', self.layout, size=(self.window_size_x,self.window_size_y), element_justification='center')
        self.run_main_loop()
    
    # class methods

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
    
    
    # This little helper wraps the TFLite Interpreter as a numpy-to-numpy function.
    def lite_model(images, interpreter):
        interpreter.allocate_tensors()
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], images)
        interpreter.invoke()
        return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    
    # instance methods
    def run_main_loop(self):      
        # EVENTS
        while True:
            event, values = self.window.read(timeout=10)
            
            if event == sg.WIN_CLOSED:
                break
            elif event == '-START-':
                self.start_time = time()
                self.timer_active = True
                self.show_new_game = True
                self.video_playing = True
                self.start_showing_face_prompts = True
                self.new_face_prompt = True
                self.score = 0 # doesnt work
                self.window['-SCORE-'].Update(self.score_str + str(self.score))
                self.window['-PLAYER_NAME-'].Update(visible=False)
                if self.window['-START-'].get_text() == "Play":
                    self.window['-PAUSE-'].Update(visible=True)
                    self.window['-START-'].Update('Restart')
                    
                    ### select current player, initialize player data array
                    if self.player_data_dict.get(values['-PLAYER_NAME-']) == None:
                        self.player_data_dict.update({values['-PLAYER_NAME-']:[]})
                    self.current_player = values['-PLAYER_NAME-']
                
            elif event == '-PAUSE-':
                if self.pause_active == False:
                    # pause
                    self.window['-PAUSE-'].Update('Unpause')
                    self.timer_active = False
                    self.video_playing = False
                    self.pause_active = True
                    self.pause_counter_time = time()          
                else:
                    # unpause
                    self.window['-PAUSE-'].Update('Pause')
                    self.timer_active = True
                    self.video_playing = True
                    self.pause_active = False
                    
                    # pause timer - counting during pause prevention
                    self.pause_counter_time = time() - self.pause_counter_time 
                    self.paused_correction_time = self.pause_counter_time
                    ### the problem is that it gets overwritten in the if
                    self.pause_counter_time = 0
            elif event == '-NEW_GAME-':
                self.start_time = time()
                self.timer_active = True
                self.show_new_game = True
                self.video_playing = True
                self.start_showing_face_prompts = True
                self.new_face_prompt = True
                self.window['-START-'].Update('Restart')
                self.score = 0 # doesnt work
                self.window['-SCORE-'].Update(self.score_str + str(self.score))
                
                ### the new game button triggers pause if it was not triggered before
                ### and it also shows the player name input again, so the player can be changed 
                if self.pause_active == False:
                    # pause
                    self.window['-PAUSE-'].Update('Unpause')
                    self.timer_active = False
                    self.video_playing = False
                    self.pause_active = True
                    
                # show player name input
                self.window['-PLAYER_NAME-'].Update(visible = True)
                self.start_time = time()
                self.window['-START-'].Update('Play')
                self.window['-PAUSE-'].Update(visible=False)
                self.window['-NEW_GAME-'].Update(visible=False)
                
                
                
            if self.timer_active == True:
                ### WIP
                elapsed_time = round(time() - self.start_time - self.paused_correction_time, 1)
                self.paused_correction_time = 0
                self.window['-TIME-'].update(elapsed_time)
                
                
            if self.show_new_game == True and self.new_game_button_added == False:
                self.new_game_button_added = True
                self.window['-NEW_GAME-'].update(visible=True)
                
            if self.video_playing == True:
                ret, frame = self.video_cap.read()
                
                
                ### copied from faceDet 
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Detect faces
                
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
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
                H,W = self.interpreter.get_input_details()[0]['shape'][1:3]
                resized_face = cv2.resize(f_im, (W, H))
                def classify_face(face_img):
                    return __class__.softmax(__class__.lite_model(face_img[None, ...].astype(np.float32)/255, self.interpreter)[0])
                
                pred = classify_face(resized_face)
                pred.argmax()
                # print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")
                cv2.putText(frame, f"Prediction: {__class__.class_names[pred.argmax()]}: Certainty: {pred.max()*100}\%", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                # show the frame to our screen
                self.writer.write(frame)
                ### end of face det copy
                
                imgbytes = cv2.imencode('.png', frame)[1].tobytes()
                self.window['-IMAGE-'].update(data=imgbytes)
                
                ### game mechanics
                # print(current_rolled_class + "" + class_names[pred.argmax()])
                if self.current_rolled_class == __class__.class_names[pred.argmax()]:
                    self.score += 1
                    self.window['-SCORE-'].update(self.score_str + str(self.score))
                    self.new_face_prompt = True
                
            if self.start_showing_face_prompts == True:
                if self.new_face_prompt == True:
                    self.new_face_prompt = False
                    rolled_class_idx = random.randint(0,5)
                    while rolled_class_idx == self.prev_rolled_class:
                        rolled_class_idx = random.randint(0,5)
                    current_rolled_class = __class__.class_names[rolled_class_idx]
                    self.prev_rolled_class = rolled_class_idx
                    self.window['-FACE_PROMPT-'].update(self.face_prompt_str + current_rolled_class)
                    

                    

                
        # close cv2
        self.video_cap.release()
        self.writer.release()
        cv2.destroyAllWindows()

        # close pysimplegui
        self.window.close()



app = MakeAFace()

