# from clean_text import clean
from flask import render_template
from flask import Flask,request
import os
# from keras.models import load_model
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# import pickle

app=Flask(__name__)

port = int(os.environ.get("PORT", 5000))
@app.route('/', methods=['GET','POST'])
def home():
    # if request.method == 'POST':
    #     data=request.form["data"]   #This is the data for prediction
    #     data = clean(data)
    #     with open('tokenizer.pickle', 'rb') as handle:
    #         tokenizer = pickle.load(handle)
    #     tokenizer.fit_on_texts(data)
    #     sequences = tokenizer.texts_to_sequences(data)
    #     data = pad_sequences(sequences, maxlen=200)
    #     model = load_model('BestModel.hdf5')
    #     y = model.predict(data)
    #     print(y)
    #     return(render_template('index.html', result = y))
    return render_template('index.html')
if __name__ ==  '__main__':
    app.run(port=port)
