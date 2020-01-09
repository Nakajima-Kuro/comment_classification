from flask import render_template
from flask import Flask,request
from keras.models import load_model
app=Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        data=request.form["data"]   #This is the data for prediction
        model = load_model('BestModel.hdf5')

    return render_template('index.html')


if __name__ ==  '__main__':
    app.run()