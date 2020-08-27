import cv2 as cv
import numpy as np
from keras.models import load_model

MODEL_DIR = 'model'
CASCADE_CLASSIFIER = 'res/haarcascade_frontalface_default.xml'

model = load_model(os.path.join(MODEL_DIR))

IMG_SIZE = 48

WINDOW_NAME = 'Emotion Analyzer'

face_cascade = cv.CascadeClassifier(CASCADE_CLASSIFIER)

cv.namedWindow(WINDOW_NAME)

cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()
    if ret:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h), i in zip(faces, range(len(faces)):
            img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cv.imshow('face {0}'.format(i + 1))

        cv.imshow(WINDOW_NAME, img)
    else:
        print('failed to retrieve frame!')