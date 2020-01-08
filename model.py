#%%Import Lib
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, GRU, Dropout, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import glob
import numpy as np
import _pickle as pickle
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
        y.append(1)
##Import Bad comment
for link in Bad:
    print(link)
    for filename in glob.glob(link):
        comment = open(filename, "r")
        X.append(clean(comment.read()))
        y.append(0)
del filename
del link
del Bad
del Good
print("Import successedfully")

#%% Preprocessing
#Create sequence
vocabulary_size = 20000
tokenizer = Tokenizer(num_words= vocabulary_size)
tokenizer.fit_on_texts(X)
sequences = tokenizer.texts_to_sequences(X)
data = pad_sequences(sequences, maxlen=200)
#One-Hot Label
y = to_categorical(y, num_classes=2, dtype='float32')
X = data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
del X
del data
del y
del sequences
print("Preprocess successfully")

#%% Model
model = Sequential()
##Layer 1
model.add(Embedding(input_dim=vocabulary_size, output_dim=16, input_length=200))
##Layer 2
model.add(GRU(8, activation='tanh', recurrent_activation='sigmoid', kernel_initializer='glorot_uniform', 
              recurrent_initializer='orthogonal', implementation=2, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
##Layer 3
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

opt = Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, amsgrad=True)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#Save the best model
checkpoint = ModelCheckpoint('BestModel.hdf5', monitor = 'val_accuracy', verbose = 2, save_best_only = True, mode = 'max')
callbacks_list = [checkpoint]

model.fit(X_train, y_train, batch_size=30, epochs=100, verbose=1, validation_data=(X_test, y_test), shuffle=True, callbacks = callbacks_list)