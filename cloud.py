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
wordforcloud = {}

#Read vectorized vocabularay from train set.
filename = model_path +"vectorizer.sav"
vectorizer = pickle.load(open(filename, 'rb'))
vocab = vectorizer.get_feature_names()

#Read Genre list
for genre in genres:
    wordforcloud[genre] = defaultdict(int)

    #Load saved model
    filename = model_path+ genre + "_" + 'rf.sav'
    forest = pickle.load(open(filename, 'rb'))

    #Match importance and vacabulary
    f_import = forest.feature_importances_
    importance = [(f_import[i],i)for i in range(5000)]
    importance.sort()
    importance.reverse()

    #print importance of rf
    print "%11s :WordCloud based on Randomforest Importance for each genre" % genre
    for i in range(100):
        print "%1.4f&%s\\\\" %(importance[i][0], vocab[importance[i][1]])

    for i in range(100):
        wordforcloud[genre][vocab[importance[i][1]]] = importance[i][0]

    #Make wordcloud based on importance 
    print "Drawing wordcloud......"
    Mask = np.array(Image.open("./wordcloud_origin/" + genre +".jpeg"))
    wordcloud = WordCloud(background_color='white',
                          mask=Mask,
                          width = 2000,
                          height = 1500,
                             ).generate_from_frequencies(wordforcloud[genre])

    fig = plt.figure(dpi=720)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    #save word cloud
    fig.savefig("./wordcloud/" + model +"/"+genre +".png", dpi = fig.dpi)