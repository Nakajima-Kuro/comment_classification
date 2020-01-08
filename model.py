#%%Import Lib
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical

import glob
import numpy as np
import _pickle as pickle
from sklearn.model_selection import train_test_split
from clean_text import clean

#%% Import Data
nX = []
ny = []
Good = ['Dataset/NHuy/Good/*.txt','Dataset/NAnh/Good/*.txt','Dataset/Long/Good/*.txt']
Bad = ['Dataset/NHuy/Bad/*.txt','Dataset/NAnh/Bad/*.txt','Dataset/Long/Bad/*.txt']
##Import Good comment
for link in Good:
    print(link)
    for filename in glob.glob(link):
        comment = open(filename, "r")
        nX.append(clean(comment.read()))
        ny.append("Good")
        print(len(ny))
##Import Bad comment
for link in Bad:
    print(link)
    for filename in glob.glob(link):
        comment = open(filename, "r")
        nX.append(clean(comment.read()))
        ny.append("Bad")
        print(len(ny))
del filename
del link
del Bad
del Good
print("Import successedfully")