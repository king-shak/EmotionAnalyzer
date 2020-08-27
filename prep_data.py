from tqdm import tqdm
import numpy as np
import random
import pickle
import cv2
import os

DATADIR = './data/'
TEST_DATADIR = './data/test/'

# load up our categories
CATEGORIES = pickle.load(open(os.path.join(DATADIR, 'categories.pickle'), 'rb'))

# the size of the images the neural network will use
IMG_SIZE = 48

# go through, load and prepare the training and test data
training_data = []
def create_training_data():
	for category in tqdm(CATEGORIES):
		path = os.path.join(DATADIR, category)
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
				training_data.append([new_array, class_num])
			except Exception as e:
				pass

test_data = []
def create_test_data():
	for category in tqdm(CATEGORIES):
		path = os.path.join(TEST_DATADIR, category)
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
				test_data.append([new_array, class_num])
			except Exception as e:
				pass

create_training_data()
create_test_data()

# do final prep of the data
random.shuffle(training_data)
random.shuffle(test_data)

training_features = []
training_labels = []

for feature, label in training_data:
	training_features.append(feature)
	training_labels.append(label)

test_features = []
test_labels = []

for feature, label in test_data:
	test_features.append(feature)
	test_labels.append(label)

# TODO: figure out what exactly this does
training_features = np.array(training_features).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
training_labels = np.array(training_labels)

test_features = np.array(test_features).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_labels = np.array(test_labels)

# save the features and labels
pickle_out = open(os.path.join(DATADIR, 'features.pickle'), 'wb')
pickle.dump(training_features, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(DATADIR, 'labels.pickle'), 'wb')
pickle.dump(training_labels, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(DATADIR, 'test_features.pickle'), 'wb')
pickle.dump(test_features, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(DATADIR, 'test_labels.pickle'), 'wb')
pickle.dump(test_labels, pickle_out)
pickle_out.close()