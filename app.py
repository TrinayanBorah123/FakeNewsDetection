# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 21:39:58 2020

@author: Trinayan Borah
"""

#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

#for importing our keras model
import tensorflow.keras.models
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences

#system level operations (like loading files)
import sys 
#for reading operating system data
import os
#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))

from load import * 
#initalize our flask app
app = Flask(__name__)
#global vars for easy reusability
global model
#initialize these variables
model= init()
nltk.download('stopwords')

@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")
@app.route('/predict',methods=['GET','POST'])
def predict():
    features=[x for x in request.form.values()]
    #final=[np.array(features)]
    test=str(features)
    test_data=[]
    review_test = re.sub('[^a-zA-Z]', ' ', test)
    review_test = review_test.lower()
    review_test = review_test.split()

    review_test = [ps.stem(word) for word in review_test if not word in stopwords.words('english')]
    review_test = ' '.join(review_test)
    test_data.append(review_test)
    onehot_repr_test_single=[one_hot(test_data[0],5000)] 
    onehot_repr_test_single
    sent_length=20
    embedded_docs_test_single=pad_sequences(onehot_repr_test_single,padding='pre',maxlen=sent_length)
    #print(embedded_docs_test_single)
    y_pred_output=model.predict_classes(embedded_docs_test_single)
    if (y_pred_output[0][0]==0):
        pred='Fake News'
    else:
        pred='Real news'
    return render_template("index.html",pred='Your news Prediction is: {} '.format(pred))
if __name__ == "__main__":
    app.run()