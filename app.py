from clean_text import clean
from flask import render_template
from flask import Flask,request
import os
from keras.models import load_model
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app=Flask(__name__)

port = int(os.environ.get("PORT", 5000))
@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        data = request.form["data"]   #This is the data for prediction
        data = text_to_word_sequence(clean(data))
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        sequences = tokenizer.texts_to_sequences([data])
        X = pad_sequences(sequences, maxlen=600)
        model = load_model('BestModel.hdf5')
        print(sequences)
        print(data)
        print(X)
        y = model.predict(X).flatten()
        print(y)
        if y[0] > y[1]:
            message = "This is a bad comment"
        else:
            message = "This is a good comment"
        return(render_template('index.html', result = message, data = request.form["data"]))
    return render_template('index.html')
if __name__ ==  '__main__':
    app.run(port=port)
