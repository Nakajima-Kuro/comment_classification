from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import glob
import numpy as np
from keras.utils import to_categorical
import _pickle as pickle
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

#%% Import Data
nX = []
ny = []
Good = ['Dataset/NHuy/Good/*.txt','Dataset/NAnh/Good/*.txt','Dataset/Long/Good/*.txt']

for link in Good:
    print(link)
    for filename in glob.glob(link):
        comment = open(filename, "r")
        nX.append(comment.read())
        ny.append(1)
        print(len(ny))
print("Import successedfully")