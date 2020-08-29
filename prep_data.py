from tqdm import tqdm
import numpy as np
import cv2 as cv
import random
import pickle
import os

DATADIR = './data/'
TEST_DATADIR = './data/test/'

# load up our categories
CATEGORIES = pickle.load(open(os.path.join(DATADIR, 'categories.pickle'), 'rb'))

# the size of the images the neural network will use
IMG_SIZE = 48

# go through, load and prepare the training and test data
def create_data(data_dir):
	data = []
	for category in tqdm(CATEGORIES):
		path = os.path.join(data_dir, category)
		class_num = CATEGORIES.index(category)
		for img in os.listdir(path):
			try:
				img_array = cv.imread(os.path.join(path, img), cv.IMREAD_GRAYSCALE)
				new_array = cv.resize(img_array, (IMG_SIZE, IMG_SIZE))
				data.append([new_array, class_num])
			except Exception as e:
				pass
	return data

training_data = create_data(DATADIR)
test_data = create_data(TEST_DATADIR)

# shuffle the data, then seperate them into the features (images) and labels
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

# convert them to numpy arrays with the correct dimensions
training_features = np.array(training_features).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
training_labels = np.array(training_labels)

test_features = np.array(test_features).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_labels = np.array(test_labels)

# save the features and labels
# training
pickle_out = open(os.path.join(DATADIR, 'features.pickle'), 'wb')
pickle.dump(training_features, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(DATADIR, 'labels.pickle'), 'wb')
pickle.dump(training_labels, pickle_out)
pickle_out.close()

# test
pickle_out = open(os.path.join(DATADIR, 'test_features.pickle'), 'wb')
pickle.dump(test_features, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(DATADIR, 'test_labels.pickle'), 'wb')
pickle.dump(test_labels, pickle_out)
pickle_out.close()