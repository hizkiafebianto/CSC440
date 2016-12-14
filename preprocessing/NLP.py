# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 11:41:55 2016

@author: Hizkia
"""

import pandas as pd
import numpy as np
import os 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

#


## Loading Dataset
os.getcwd()
os.chdir("D:/MBA Journey/2016 - Fall Quarter/CSC240 - Data Mining/Project 2/CSC440")
data = pd.read_csv(os.getcwd() + "/data/combined.csv")
data.iloc[0:5,2]

# Split data into train and test data
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']


example = train.iloc[3,10]
example2 = example.lower()
example3 = CountVectorizer().build_tokenizer()(example2)

pd.DataFrame([[x,example3.count(x)] for x in set(example3)],
               columns = ['Word', 'Count'])

trainheadlines = []
for row in range(0,len(train.index)):
    trainheadlines.append(' '.join(str(x) for x in train.iloc[row,2:27]))

basicVectorizer = CountVectorizer()
basictrain = basicVectorizer.fit_transform(trainheadlines)
print(basictrain[0:2])
basicwords = basicVectorizer.get_feature_names()

testheadlines = []
for row in range(0,len(test.index)):
    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))
basictest = basicVectorizer.transform(testheadlines)
predictions = basicmodel.predict(basictest)

print(basictest)
basictest.shape
basictrain.shape

basicmodel = LogisticRegression()
basicmodel = basicmodel.fit(basictrain, train["Label"])


from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
print(sid.polarity_scores(example))
score = sid.polarity_scores(example)
score['neg']

from nltk.corpus import sentiwordnet as swn

breakdown = swn.senti_synset('breakdown.n.03')
print breakdown
swn.senti_synsets('not')
print(swn.senti_synset('not.r.01'))