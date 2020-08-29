from shutil import rmtree, copyfile
from tqdm import tqdm
import pandas as pd
import pickle
import os

RAW_DATA_DIR = './raw data/'
DATA_DIR = './data/'

# essentially what we're doing is taking the raw data and splitting it into the individuals categories

# first, we get the categories from the index
index = pd.read_csv(os.path.join(RAW_DATA_DIR, 'legend.csv'), comment='#')
index.emotion = index.emotion.str.lower()
categories = list(index.emotion.unique())

# now we create the directories
def clear_directory(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            rmtree(os.path.join(root, d))

print('creating directories for each class...')
for category in tqdm(categories):
    path = os.path.join(DATA_DIR, category)
    if not os.path.exists(path):
        # create the directory if it doesn't exist
        os.makedirs(path)
    elif (len(os.listdir(path)) != 0):
        # if it does exist, clear it
        clear_directory(path)
print('done!')

# seperate the images into there respective categories
print('\ncopying images to respective directories...')
NUM_OF_IMAGES = index.shape[0]
with tqdm(position=0, leave=True):
    for i in tqdm(range(NUM_OF_IMAGES)):
        img = index.image[i]
        emotion = index.emotion[i]
        src = os.path.join(RAW_DATA_DIR, 'images', img)
        dst = os.path.join(DATA_DIR, emotion, img)
        copyfile(src, dst)
print('done!')

# save the categories for later use
print('\nsaving categories...')
categories_file = open(os.path.join(DATA_DIR, 'categories.pickle'), 'wb')
pickle.dump(categories, categories_file)
categories_file.close()
print('done!')