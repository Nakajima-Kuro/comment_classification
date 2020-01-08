#%%Import Lib
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, GRU, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import glob
import numpy as np
import _pickle as pickle
from sklearn.model_selection import train_test_split
from clean_text import clean

#%% Import Data
X = []
y = []
Good = ['Dataset/NHuy/Good/*.txt','Dataset/NAnh/Good/*.txt','Dataset/Long/Good/*.txt']
Bad = ['Dataset/NHuy/Bad/*.txt','Dataset/NAnh/Bad/*.txt','Dataset/Long/Bad/*.txt']
##Import Good comment
for link in Good:
    print(link)
    for filename in glob.glob(link):
        comment = open(filename, "r")
        X.append(clean(comment.read()))
        y.append("Good")
        print(len(y))
##Import Bad comment
for link in Bad:
    print(link)
    for filename in glob.glob(link):
        comment = open(filename, "r")
        X.append(clean(comment.read()))
        y.append("Bad")
        print(len(y))
del filename
del link
del Bad
del Good
print("Import successedfully")

#%% Preprocessing
### Create sequence
vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
data = pad_sequences(sequences, maxlen=50)
### One-Hot Label
to_categorical(y, num_classes=2, dtype='float32')
### Train, test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Model
model = Sequential()
##Layer 1
model.add(Embedding(input_dim=1000, output_dim=128, input_length=None))
##Layer 2
model.add(GRU(128, activation='tanh', recurrent_activation='sigmoid', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', implementation=2, return_sequences=True))
##Layer 3
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

fit(X_train, y_train, batch_size=30, epochs=100, verbose=1, validation_data=(X_test, y_test), shuffle=True)