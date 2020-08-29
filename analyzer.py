from tensorflow.keras.models import load_model
import threading
import numpy as np
import cv2 as cv
import pickle
import time
import os

CAMERA_PORT = 0
IMG_SIZE = 48
MODEL_DIR = './models/26-08-2020_22-33-45'
DATA_DIR = 'data'
CASCADE_CLASSIFIER_PATH = 'res/haarcascade_frontalface_default.xml'

ROI_BOX_COLOR = (255, 0, 0)
EMOTION_FONT_COLOR = (0, 255, 0)
FPS_FONT_COLOR = (0, 0, 255)
FONT = cv.FONT_HERSHEY_SIMPLEX

# this is a seperate thread that will handle capturing frames from the webcam
# the idea here is to implement double buffering so the main pipeline doesn't 
# need to waste time waiting around for frames
class CaptureThread(threading.Thread):
    def __init__(self, port):
        super(CaptureThread, self).__init__()
        self.terminated = False
        self.buffer_lock = threading.Lock()
        self.cap = cv.VideoCapture(port)
        self.A = None
        self.B = None
        self.start()

    def run(self):
        while not self.terminated:
            ret, frame = self.cap.read()

            if ret:
                # write the data
                self.A = frame.copy()

                # swap the identifiers
                with self.buffer_lock:
                    self.A, self.B = self.B, self.A

            # brief pause to keep the thread stable
            time.sleep(0.001)
        self.cap.release()

# load up the model, along with the categories
model = load_model(MODEL_DIR)
categories = pickle.load(open(os.path.join(DATA_DIR, 'categories.pickle'), 'rb'))

# create the window that will display the frames
WINDOW_NAME = 'Emotion Analyzer'
cv.namedWindow(WINDOW_NAME)

# initialize the cascade classifier for detecting faces
face_cascade = cv.CascadeClassifier(CASCADE_CLASSIFIER_PATH)

# initialize the capture thread
cap = CaptureThread(CAMERA_PORT)
start = time.time()

# wait until the first frame is captured
while cap.B is None:
    time.sleep(0.001)

while True:
    # grab the most recent frame from the capture thread
    with cap.buffer_lock:
        img = cap.B.copy()
    
    # only proceed if the frame was succeccfully retrieved
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # look for faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # draw the roi for the face
        upperLeft = (x, y)
        bottomRight = (x + w, y + h)
        img = cv.rectangle(img, upperLeft, bottomRight, ROI_BOX_COLOR, 2)

        # extract and prepare the roi for predictions
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv.resize(roi_gray, (IMG_SIZE, IMG_SIZE))
        data = np.array([roi_gray]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

        # get the prediction, draw it on the frame
        emotion = categories[model.predict_classes(data)[0]]
        cv.putText(img, emotion, (x, y - 10), FONT, 1, EMOTION_FONT_COLOR, 2)
    
    end = time.time()

    # write the current fps
    processingTimeMs = (end - start) * 1000
    fps = 1000 / processingTimeMs
    cv.putText(img, str(round(fps, 2)), (0, 25), FONT, 1, FPS_FONT_COLOR, 3)
    cv.imshow(WINDOW_NAME, img)
    start = time.time()
    
    key = cv.waitKey(1)
    if key == ord("q"):
        cap.terminated = True
        cap.join()
        break