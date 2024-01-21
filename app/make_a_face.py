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
        
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "lite_emotions_model_efficientnetv2-b0-21k-ft1k_adam.tflite")
        with open(file_path, "rb") as f:
            self.lite_model_content = f.read()
            
        self.interpreter = tf.lite.Interpreter(model_content=self.lite_model_content)
        # Load the cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        #self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # VIDEO 
        self.video_playing = False
        # TO BE MODIFIED,
        self.CAM_HEIGHT = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.CAM_WIDTH = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        self.H,self.W = self.interpreter.get_input_details()[0]['shape'][1:3]

        # TIMER
        self.start_time = 0
        self.timer_active = False

        self.total_paused_correction_time = 0
        self.pause_counter_time = 0
        self.pause_counter_timer_active = False

        # PAUSE BUTTON
        self.pause_active = False

        # PLAY BUTTON
        self.start_active = False

        # NEW_GAME BUTTON
        self.show_new_game = False
        self.new_game_button_added = False
        self.show_black_screen = True

        # PLAYER_NAME INPUT
        self.show_player_name_input = True

        # MAKE-A-FACE MECHANICS
        self.prev_rolled_class = None
        self.face_prompt_str = 'Face prompt: '
        self.start_showing_face_prompts = False
        self.new_face_prompt = False
        self.current_rolled_class = ''
        self.score_str = 'Score: '
        self.round_duration = 20;

        #"name": highscore
        self.player_data_dict = {}
        self.current_player = None
        self.score = 0

        self.end_screen = False
            
        # STATES
        self.STATE_START_SC = 0
        self.STATE_PLAY_SC = 1
        self.STATE_PAUSE_SC = 2        
        

        # GUI 
        sg.theme('black')
        # sg.set_options(button_element_size=(6,3))
        self.layout = [
            [sg.Text('', key='-FACE_PROMPT-',font=('Helvetica', 20, 'bold'), text_color='red')],             # face to be made
            [sg.VPush()],                                   # blank space
            [sg.Image(key='-IMAGE-')],                      # game window with camera
            [sg.VPush()],                                   # blank space
            [sg.Text(self.score_str + str(self.score), key='-SCORE-',  visible=False)],    # score primitive
            [sg.Text('Time', key='-TIME-')],                # time row
            [
                sg.Button('Play', key='-START-',size=(7,3)),
                sg.Button('Pause', key='-PAUSE-',size=(7,3),visible=False),
                # new_game = restart + change player
                sg.Button('New Game', key='-NEW_GAME-',visible=False,size=(7,3)),
                sg.Input('Player', key='-PLAYER_NAME-',visible=True,size=(20,2)),
                sg.ButtonMenu("Highscores", ["", "empty"], key='-HIGHSCORES-'),
                sg.Checkbox("No-Aheago", key='-NO_AHEAGO-',visible=True,background_color='white',text_color='black')
            ]          
            ]
        
        
        self.window_size_x, self.window_size_y = 800, 650
        self.window = sg.Window('make-a-face!', self.layout, size=(self.window_size_x,self.window_size_y), element_justification='center')
        # load highscores if its not empty, setup for values were loaded in
        if self.player_data_dict:
            dict_to_list = []
            for key, value in self.player_data_dict.items():
                dict_to_list.append(f"{key}: {value}")
            dict_to_list = ["highscores",dict_to_list]
            self.window['-HIGHSCORES-'].Update(menu_def=dict_to_list)
        
        self.run_main_loop()
    
    # class methods
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
    def handle_start(self, values):
        self.pause_active = False
        self.window['-PAUSE-'].Update('Pause')
        self.window['-NO_AHEAGO-'].Update(visible=False)
        self.window['-SCORE-'].Update(visible=True)
        self.window['-HIGHSCORES-'].Update(visible=False)
        self.end_screen = False
        self.show_black_screen = False
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
            self.window['-NEW_GAME-'].Update(visible=True)
            
            ### select current player, initialize player data array
            
            if self.player_data_dict.get(values['-PLAYER_NAME-']) == None:
                self.player_data_dict.update({values['-PLAYER_NAME-']:0})
                
            self.current_player = values['-PLAYER_NAME-']
            
    def handle_pause(self):
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
            correction_time = time() - self.pause_counter_time 
            # we add the correction over time
            self.total_paused_correction_time += correction_time
            self.pause_counter_time = 0
            
    def handle_new_game(self):
        self.window['-SCORE-'].Update(visible=False)
        self.end_screen = False
        ### timer setup
        self.total_paused_correction_time = 0
        self.pause_active = False
        self.window['-PAUSE-'].Update('Pause')

        ### setup rest
        self.start_showing_face_prompts = False
        self.show_black_screen = True
        self.start_time = time()
        self.timer_active = False
        self.show_new_game = True
        self.window['-START-'].Update('Restart')
        self.score = 0 # doesnt work
        self.window['-SCORE-'].Update(self.score_str + str(self.score))
        
        # make the screen black
        
        
        ### the new game button triggers pause if it was not triggered before
        ### and it also shows the player name input again, so the player can be changed 
        if self.pause_active == False:
            # pause
            self.window['-PAUSE-'].Update('Pause')
            self.timer_active = False
            self.window['-TIME-'].Update('Time')
            self.video_playing = False
            self.pause_active = False
            
        # show player name input
        self.window['-PLAYER_NAME-'].Update(visible = True)
        self.start_time = time()
        self.window['-START-'].Update('Play')
        self.window['-PAUSE-'].Update(visible=False)
        self.window['-NEW_GAME-'].Update(visible=False)
        
        # load highscores if its not empty
        self.window['-HIGHSCORES-'].Update(visible=True)
        if self.player_data_dict: 
            dict_to_list = []
            for key, value in self.player_data_dict.items():
                dict_to_list.append(f"{key}: {value}")
            
            dict_to_list = ["highscores",dict_to_list]
            self.window['-HIGHSCORES-'].Update(menu_definition=dict_to_list)
        self.window['-NO_AHEAGO-'].Update(visible=True)
    
    def classify_face(self, face_img):
        return __class__.softmax(__class__.lite_model(face_img[None, ...].astype(np.float32)/255, self.interpreter)[0])
    
    def run_main_loop(self):
        # EVENTS
        while True:
            event, values = self.window.read(timeout=10)
                
            if event == sg.WIN_CLOSED:
                break
            elif event == '-START-':
                self.handle_start(values)    
            elif event == '-PAUSE-':
                self.handle_pause()
            elif event == '-NEW_GAME-':
                self.handle_new_game()
                
                
            #### options
                
                
            if self.timer_active == True:
                ### WIP
                # print(str(self.total_paused_correction_time) + "    " + str(time() - self.start_time))
                elapsed_time = round(self.round_duration - (time() - self.start_time - self.total_paused_correction_time), 1)
                self.window['-TIME-'].update(elapsed_time)
                if elapsed_time <= 0:
                    ### show modified new game screen, clean up and save score
                    
                    # saving the highscore
                    if self.player_data_dict[self.current_player] < self.score:
                        self.player_data_dict[self.current_player] = self.score
                    
                    cache_player = self.current_player
                    cache_score = self.score
                    cache_duration = self.round_duration
                    self.handle_new_game()
                    self.window['-FACE_PROMPT-'].Update(f"{cache_player} scored: {cache_score} in {cache_duration} seconds!")
                    self.end_screen = True
                    
                                    
                
            if self.show_new_game == True and self.new_game_button_added == False:
                self.new_game_button_added = True
                self.window['-NEW_GAME-'].update(visible=True)
                
            # if self.video_playing:
            if True:
                ret, frame = self.video_cap.read()
                
                
                ### copied from faceDet 
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Detect faces
                
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                # Draw rectangle around the faces
                for (x, y, w, h) in faces:
                    ### make sure there are no out of bounds errors
                    x if x >= 0 else 0
                    y if y >= 0 else 0
                    xw = x+w if x+w <= self.CAM_WIDTH else self.CAM_WIDTH
                    yh = y+h if y+h <= self.CAM_HEIGHT else self.CAM_HEIGHT
                    
                    cv2.rectangle(frame, (x, y), (xw, yh), (255, 0, 0), 2)
                # Display the output
                # print(faces[0])
                if not np.any(faces):
                    continue
                (x, y, w, h) = faces[0]
                ### make sure there are no out of bounds errors
                x if x >= 0 else 0
                y if y >= 0 else 0
                xw = x+w if x+w <= self.CAM_WIDTH else self.CAM_WIDTH
                yh = y+h if y+h <= self.CAM_HEIGHT else self.CAM_HEIGHT
                f_im = frame[y:yh, x:xw, :]
                
                # WIP culprit
                resized_face = cv2.resize(f_im, (self.W, self.H))
                
                pred = self.classify_face(resized_face)
                pred.argmax()
                # print(f"Time to process 1 frame: {total * 1000:.0f} milliseconds")
                cv2.putText(frame, f"Prediction: {__class__.class_names[pred.argmax()]}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(frame, f"Certainty: {round(pred.max()*100, 2)}%", (10,65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                # show the frame to our screen
                #self.writer.write(frame)
                ### end of face det copy
                                
                if self.show_black_screen != True:
                    if self.video_playing:
                        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
                        self.window['-IMAGE-'].update(data=imgbytes)
                else:
                    # change the screen to all black
                    shape = (self.CAM_HEIGHT, self.CAM_WIDTH, 3)
                    zeros_array = np.zeros(shape, dtype=np.uint8)
                    imgbytes = cv2.imencode('.png', zeros_array)[1].tobytes()
                    self.window['-IMAGE-'].update(data=imgbytes)
                
                ### game mechanics
                if self.current_rolled_class == __class__.class_names[pred.argmax()]:
                    self.score += 1
                    self.window['-SCORE-'].update(self.score_str + str(self.score))
                    self.new_face_prompt = True
                
            if self.start_showing_face_prompts == True:
                if self.new_face_prompt == True:
                    self.new_face_prompt = False
                    if self.window['-NO_AHEAGO-'].get():
                        rolled_class_idx = random.randint(1,5)
                        while rolled_class_idx == self.prev_rolled_class:
                            rolled_class_idx = random.randint(1,5)
                    else:        
                        rolled_class_idx = random.randint(0,5)
                        while rolled_class_idx == self.prev_rolled_class:
                            rolled_class_idx = random.randint(0,5)
                    self.current_rolled_class = __class__.class_names[rolled_class_idx]
                    self.prev_rolled_class = rolled_class_idx
                    self.window['-FACE_PROMPT-'].update(self.face_prompt_str + self.current_rolled_class)
            else:
                if self.end_screen == False:
                    self.window['-FACE_PROMPT-'].update("")
                
                
        # close cv2
        self.video_cap.release()
        # self.writer.release()
        cv2.destroyAllWindows()

        # close pysimplegui
        self.window.close()



app = MakeAFace()

