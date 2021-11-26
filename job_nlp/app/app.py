#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")
import re
import string
import numpy as np
import random
import pandas as pd
from collections import Counter
from gensim.parsing.preprocessing import STOPWORDS
import gensim
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import os
from joblib import load

from sklearn.feature_extraction.text import TfidfTransformer
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask

from flask import render_template,jsonify
# !pip install flask-restful
from flask_restful import Api, Resource, reqparse
from flask import request
from joblib import dump, load
#Defining App !
app = Flask(__name__,template_folder='templates')

@app.route('/',methods=['GET'])
def hello():
    return render_template("index.html")
model = load('model.joblib')


# @app.route('/predict', methods=['POST'])
# def predict():
#     # if request.method == 'POST':
#     data = request.data['input']
#     model_input = process_(data)
#     results = model.predict(model_input)
#     return jsonify({"prediction": results})
@app.route('/',methods=['POST'])
def predict_():
    infile=request.files['infile']
    text_path='./texts/'+infile.filename
    infile.save(text_path)
    texts=load(text_path)
    texts=pd.DataFrame(texts)
    texts=process_(texts)
    model = load('model.joblib')
    yhat=model.predict(texts)
    classification=yhat[0], 200
    return render_template("index.html",prediction=classification)
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
#

if __name__ == "__main__":
    app.run(port=3000,debug=True)

def process_(my_input):
    def clean_text(text):
        '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
        and remove words containing numbers.'''
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text
    clean_text(my_input)
    vect = CountVectorizer()
    my_input=vect.fit_transform(my_input)
    tfidf_transformer = TfidfTransformer()
    my_input=tfidf_transformer.fit_transform(my_input)
    return my_input

