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

# load the pre-trained YOLOv8n model
# model = YOLO("yolov8n.pt")
# model = TensorflowLiteClassificationModel("/tmp/lite_emotions_model_efficientnet_b0.tflite")
# model = tfkeras.models.load_model('model.h5')
# model = keras.models.load_model("model.keras")
# model = tf.keras.saving.load_model("model.keras")
# model = tf.saved_model.load("/content/modelmodel.keras")

# interpreter = tf.lite.Interpreter("/tmp/lite_emotions_model_efficientnet_b0.tflite")

with open("tmp/lite_emotions_model_efficientnet_b0.tflite", "rb") as f:
    lite_model_content = f.read()
# print("Wrote %sTFLite model of %d bytes." %
      # ("optimized " if optimize_lite_model else "", len(lite_model_content)))

interpreter = tf.lite.Interpreter(model_content=lite_model_content)
# This little helper wraps the TFLite Interpreter as a numpy-to-numpy function.
def lite_model(images):
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], images)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

# class names
class_names=["Ahegao", "Angry", "Happy", "Neutral", "Sad", "Surprised"]

while True:
    ret, frame = video_cap.read()

    # if there are no more frames to process, break out of the loop
    if not ret:
        break
        
    #preprocessing frame
    #cropping
    #resize do rozmiaru modelu z lite
    #classify

    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Convert into grayscale
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
    cv2.imshow("Frame", frame)
    writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
writer.release()
cv2.destroyAllWindows()

