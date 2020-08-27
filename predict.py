from keras.models import load_model
from os import listdir
from tqdm import tqdm
import numpy as np
import cv2 as cv
import pickle
import os

MODEL_DIR = 'model'
DATA_DIR = 'data'

model = load_model(os.path.join(MODEL_DIR))
categories = pickle.load(open(os.path.join(DATA_DIR, 'categories.pickle'), 'rb'))

# get the data
IMG_SIZE = 48

# images = listdir(os.path.join(DATA_DIR, 'test'))
# test_data = []
# for img in tqdm(images):
#     img_array = cv.imread(os.path.join(DATA_DIR, 'test', img), cv.IMREAD_GRAYSCALE)
#     new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
#     test_data.append(new_array)
# test = np.array(test_data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# prediction = model.predict_classes(test)

# for i in tqdm(range(len(prediction))):
#     os.rename(os.path.join(DATA_DIR, 'test', images[i]), os.path.join(DATA_DIR, 'test', '{0}{1}.jpg'.format(str(categories[prediction[i]]), i)))

accuracy = []
for category in categories:
    images = listdir(os.path.join(DATA_DIR, 'test'))
    test_data = []
    for img in tqdm(images):
        img_array = cv.imread(os.path.join(DATA_DIR, 'test', img), cv.IMREAD_GRAYSCALE)
        new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
        test_data.append(new_array)
    test = np.array(test_data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    prediction = model.predict_classes(test)