import pickle
import os
import random
import sys
import json
import nltk
import math
import string
import re
import codecs
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.stem.porter import *
from nltk.corpus import stopwords
from wordcloud import WordCloud
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

if(len(sys.argv) <3):
    print("Not enough argument (model :'unigram', 'bigram', 'tfidf') and train or test")
    sys.exit()
model = sys.argv[1];
models = ['unigram', 'bigram', 'tfidf']
if model not in models:
    print("Wrong argument possible model :'unigram', 'bigram', 'tfidf' ")
    sys.exit()
model_path = "./models/" + model + "/"

if(sys.argv[2] == 'test'):
    path = "./testSet"
    print "RESULT FOR TEST"
if(sys.argv[2] == 'train'):
    path = "./modified_names"
    print "RESULT FOR TRAIN"

genres = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
movies= {}
movie_count = {}
scripts_genre ={}

#Read movie List
for genre in genres:
    mypath = path + '/' + genre + '/'
    movies[genre] = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    movie_count[genre] = len(movies[genre])
    scripts_genre[genre] = defaultdict()

#Read movie Scripts
for genre in genres:
    for m in movies[genre]:
        # print m
        file_object = codecs.open(path + '/' + genre + '/' + m, "r",encoding='utf-8', errors='ignore')
        this_script = file_object.read()
        scripts_genre[genre][m[:-4]] = this_script
        file_object.close()

#vectorize 
print "======================================================="
print "Vectorizing data"
X = [] 
for genre in genres:
    for m in scripts_genre[genre]:
        X.append(scripts_genre[genre][m])

#Read vectorized vocabularay from train set.
filename = model_path + "vectorizer.sav"
vectorizer = pickle.load(open(filename, 'rb'))
test_data_features = vectorizer.transform(X)
if model == 'tfidf':
    transformer = TfidfTransformer()
    test_data_features = transformer.fit_transform(test_data_features).toarray()
else:
    test_data_features = test_data_features.toarray()

#test Logistic Regression and Random forest
for g in genres:
    print "======================================================="
    print "%12s result" % g
    y = []
    title = []
    target_genre = g
    count =0
    for genre in genres:
        y_val = 1 if genre == target_genre else 0 
        for m in scripts_genre[genre]:
            y.append(y_val)
            count +=1

    #Load saved model
    filename = model_path +target_genre + "_" + 'LGR.sav'
    logistic = pickle.load(open(filename, 'rb'))
    filename = model_path +target_genre + "_" + 'rf.sav'
    forest = pickle.load(open(filename, 'rb'))
    filename = model_path +target_genre + "_" + 'SGD.sav'
    SGD = pickle.load(open(filename, 'rb'))
    #Logistic Regression
    print "Test with Logistic Regression"
    result = logistic.predict(test_data_features)
    score = logistic.score(test_data_features,y)
    prob = logistic.predict_proba(test_data_features)
    errors = [ (1 -prob[i][y[i]])**2 for i in range(count)]
    MSE = sum(errors)*1.0/count
    print "LogisticRegression score %1.7f" %score
    print "LogisticRegression MSE   %1.7f" %MSE
    #Random forest
    print "Test with random forest..."
    result = forest.predict(test_data_features)
    score = forest.score(test_data_features,y)
    prob = forest.predict_proba(test_data_features)
    errors = [ (1 -prob[i][y[i]])**2 for i in range(count)]
    MSE = sum(errors)*1.0/count
    print "RandomForest score       %1.7f" %score
    print "RandomForest MSE         %1.7f" %MSE

    print "Test with SGD..."
    result = SGD.predict(test_data_features)
    score = SGD.score(test_data_features,y)
    print "SGDprediction score      %1.7f" %score
