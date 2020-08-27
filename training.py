import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import os

DATA_DIR = './data/'

# Opening the files about data
features = pickle.load(open(os.path.join(DATA_DIR, 'features.pickle'), 'rb'))
labels = pickle.load(open(os.path.join(DATA_DIR, 'labels.pickle'), 'rb'))

test_features = pickle.load(open(os.path.join(DATA_DIR, 'test_features.pickle'), 'rb'))
test_labels = pickle.load(open(os.path.join(DATA_DIR, 'test_labels.pickle'), 'rb'))

# normalizing data (a pixel goes from 0 to 255)
features = features / 255.0
test_features = test_features / 255.0

# Building the model
model = Sequential()
# 3 convolutional layers
model.add(Conv2D(50, (3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=features.shape[1:]))

model.add(Conv2D(75, (3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(125, (3, 3), strides=(1, 1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2 hidden layers
model.add(Flatten())

model.add(Dense(500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.2))

# The output layer
model.add(Dense(7, activation='softmax'))

# Compiling the model using some basic parameters
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy',
				optimizer=opt,
				metrics=['accuracy'])

# Training the model

# datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
# train_generator = datagen.flow(features, y=labels, batch_size=32, subset='training')
# validation_generator = datagen.flow(features, y=labels, batch_size=32, subset='validation')
# history = model.fit(train_generator, validation_data=validation_generator, epochs=20)

history = model.fit(features, labels, batch_size=64, validation_split=0.2, epochs=20)
evaluation = model.evaluate(x=test_features, y=test_labels)
print(model.metrics_names)
print(evaluation)

# Saving the model
now = datetime.now()
model.save(now.strftime('%d-%m-%Y %H:%M:%S'))

# Printing a graph showing the accuracy changes during the training phase
print(history.history.keys())
fig = plt.figure(1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
fig.savefig('model accuracy.png')