import pickle
import os
import random
from collections import defaultdict
import nltk
from nltk.stem.porter import *
import json
import math
import string
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import codecs

path = "./testSet"
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
    #[1:] if .DS_store include else just []
    for m in movies[genre][:]:
        print m
        file_object = codecs.open(path + '/' + genre + '/' + m, "r",encoding='utf-8', errors='ignore')
        this_script = file_object.read()
        scripts_genre[genre][m[:-4]] = this_script
        file_object.close()

#vectorize 
print "======================================================="
print "Vectorizing data"
X = [] 
movie_result_Logi = {}
movie_result_rf = {}

for genre in genres:
    for m in scripts_genre[genre]:
        X.append(scripts_genre[genre][m])
        movie_result_Logi[m] = defaultdict(float)
        movie_result_rf[m] = defaultdict(float)

#Read vectorized vocabularay from train set.
filename = "./models/vectorizer.sav"
vectorizer = pickle.load(open(filename, 'rb'))
test_data_features = vectorizer.transform(X)
test_data_features = test_data_features.toarray()

#test Logistic Regression and Random forest
for g in genres:
    #Logistic Regression
    filename = "./models/" +g + "_" + 'LGR.sav'
    logistic = pickle.load(open(filename, 'rb'))
    result_lgr = logistic.predict_proba(test_data_features)
    #Random forest
    filename = "./models/" +g + "_" + 'rf.sav'
    forest = pickle.load(open(filename, 'rb'))
    result_rf = forest.predict_proba(test_data_features)
    count =0;
    for genre in genres:
        for m in scripts_genre[genre]:
            movie_result_Logi[m][g] = result_lgr[count][1]
            movie_result_rf[m][g] = result_rf[count][1]
            count +=1

  
for g in genres:
    print "======================================================="
    print "points for each genre given genre"
    for m in scripts_genre[g]:
        print "======================================================="
        print "%-20s result" % m
        print "Logistic regression"
        for genre in movie_result_Logi[m] :
            print ("%-15s : %1.6f, " %(genre, movie_result_Logi[m][genre]))
        #Random forest
        print "Random Forest"
        for genre in movie_result_rf[m] :
            print ("%-15s : %1.6f, " %(genre, movie_result_rf[m][genre]))
        print "\n"


filename =  "./score/LGR_test.json"
with open(filename, 'w') as fp:
    json.dump(movie_result_Logi, fp)
fp.close()    
    
filename =  "./score/RF_test.json"
with open(filename, 'w') as fp:
    json.dump(movie_result_rf, fp)
fp.close()    
    



