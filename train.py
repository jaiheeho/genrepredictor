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
from sklearn.linear_model import SGDClassifier


#model selection Unigram, Bigram, tfidf
if(len(sys.argv) ==1):
    print("Not enough argument")
    sys.exit()
model = sys.argv[1];
models = ['unigram', 'bigram', 'tfidf']
if model not in models:
    print("Wrong argument possible model :'unigram', 'bigram', 'tfidf' ")
    sys.exit()
model_path = "./models/" + model + "/"


path = "./modified_names"
genres = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
movies= {}
scripts_genre ={}

#Read movie List
for genre in genres:
    mypath = path + '/' + genre + '/'
    movies[genre] = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    scripts_genre[genre] = defaultdict()

#Read movie Scripts
for genre in genres:
    for m in movies[genre]:
        # print m
        file_object  = open(path + '/' + genre + '/' + m, 'r')
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

if model == 'unigram':
    vectorizer = CountVectorizer(analyzer = "word",                                
    tokenizer = None,                                 
    preprocessor = None,                              
    stop_words = None,                                
    max_features = 5000,
    encoding='utf-8') 
    train_data_features = vectorizer.fit_transform(X).toarray()

elif model == 'bigram':
    vectorizer = CountVectorizer(analyzer = "word",                                
    tokenizer = None,                                 
    preprocessor = None,                              
    stop_words = None,                                
    max_features = 5000,
    encoding='utf-8',
    #for bigram
    ngram_range=(1, 2)) 
    train_data_features = vectorizer.fit_transform(X).toarray()

elif model == 'tfidf':
    vectorizer = CountVectorizer(analyzer = "word",                                
    tokenizer = None,                                 
    preprocessor = None,                              
    stop_words = None,                                
    max_features = 5000,
    encoding='utf-8') 
    train_data_features = vectorizer.fit_transform(X)
    transformer = TfidfTransformer()
    test_data_features = transformer.fit_transform(train_data_features).toarray()

#save vectorizer
filename = model_path + "vectorizer.sav"
pickle.dump(vectorizer, open(filename, 'wb'))

#Train Logistic Regression and Random forest
for g in genres:
    print "======================================================="
    print "%12s result" % g
    y = []
    title = []
    target_genre = g
    count = 0;
    for genre in genres:
        y_val = 1 if genre == target_genre else 0 
        for m in scripts_genre[genre]:
            y.append(y_val)
            count +=1
    #Logistic Regression
    print "Training the Logistic Regression"
    logistic = LogisticRegression(max_iter=300)
    logistic.fit(train_data_features, y)
    result = logistic.predict(train_data_features)
    score = logistic.score(train_data_features,y)
    prob = logistic.predict_proba(train_data_features)
    errors = [ (y[i] -prob[i][y[i]])**2 for i in range(count)]
    MSE = sum(errors)*1.0/count
    print "Logistic Regression score %1.7f" %score
    print "Logistic Regression MSE %1.7f" %MSE

    print "Training the random forest..."
    forest = RandomForestClassifier(n_estimators = 50, max_depth = 8) 
    forest.fit( train_data_features, y)
    result = forest.predict(train_data_features)
    score = forest.score(train_data_features,y)
    prob = forest.predict_proba(train_data_features)
    errors = [ (y[i] -prob[i][y[i]])**2 for i in range(count)]
    MSE = sum(errors)*1.0/count
    print "Random Forest score       %1.7f" %score
    print "Random Forest MSE         %1.7f" %MSE

    print "Training the SGDClassifier..."
    SGD = SGDClassifier() 
    SGD.fit( train_data_features, y)
    result = SGD.predict(train_data_features)
    score = SGD.score(train_data_features,y)

    print "SGDClassifier score       %1.7f" %score
    # save the model to disk
    filename = model_path +target_genre + "_" + 'LGR.sav'
    pickle.dump(logistic, open(filename, 'wb'))    
    filename = model_path +target_genre + "_" + 'rf.sav'
    pickle.dump(forest, open(filename, 'wb'))    
    filename = model_path +target_genre + "_" + 'SGD.sav'
    pickle.dump(SGD, open(filename, 'wb'))   

