# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 18:04:28 2016

@author: Hizkia
"""

import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import math
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
from keras.layers import LSTM, Input, Dense, merge, Reshape
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# LOADING DATASET
os.getcwd() # Setting directory
os.chdir("D:/MBA Journey/2016 - Fall Quarter/CSC240 - Data Mining/Project 2/CSC440")
data = pd.read_csv(os.getcwd() + "/data/combined.csv") # load headlines data
djia_data = pd.read_csv(os.getcwd() + "/data/djia.csv") # load DJIA prices
headlines_keys_sorted = data.keys()[2:len(data.keys())]

# SENTIMENT SCORING
# SentiWordNet Scores
def row_swn_scoring(row):
    ''' 
        This function results in a dictionary containing records of positive 
        score, negative score, and the respective sentence in the format of
        (pos_val, neg_val, sentence).
        
        Each sentiment score is calculated as the total of sentiment scores of
        each word in a sentence using SentiWordNet Corpus.
        
        The records are sorted in descending order based on positive value and
        are reordered so that the first record ("Top1") has the highest positive
        value. 
    '''
    row_scores = dict()
    temp_row = row[2:len(row)]
    # Generate SWN Scores
    for idx, sentence in enumerate(temp_row):
        # some headlines are missing
        if not(isinstance(sentence,str)):
            sum_neg = sum_pos = 0
            row_scores[temp_row.index[idx]]  = (sum_pos,sum_neg,"blank")
        else:
            tokens = CountVectorizer().build_tokenizer()(sentence.lower())
            sum_neg = sum_pos = 0    
            for each_word in tokens:
                synset = swn.senti_synsets(each_word)
                if (synset != []):
                    sum_pos += synset[0].pos_score()
                    sum_neg += synset[0].neg_score()
            row_scores[temp_row.index[idx]]  = (sum_pos,sum_neg,sentence)
    # Sort the scores so that headlines with the highest positive scores are in the
    # leftmost part of the array and headlines with low positive scores are in the 
    # rightmost part of the array. This is to remove the effect of headlines position
    # to the weights of our model. The order of the news headlines does not matter 
    # in reality.
    temp = sorted(row_scores.items(), key=lambda x: (x[1][0] - x[1][1]), reverse = True)
    keys = sorted(row_scores.keys(), key=lambda x: int(x[3:]))
    # Generate a dictionary with sorted keys
    scores_dict = dict()
    for idx, item in enumerate(temp):
        scores_dict[keys[idx]] = item[1]
    # Return the sorted dictionary
    return scores_dict


# Generate two data frames.
# swn_df1 stores tuples of (pos_score, neg_score, sentence) for each sorted headline
# swn_df2 stores positive scores and negative scores in separate columns
swn_df1 = pd.DataFrame([], columns = headlines_keys_sorted,\
                        index = range(0,len(data.index)))
swn_df2 = pd.DataFrame([], index = range(0,len(data.index)))

for idx in range(0,len(data.index)): 
    d = row_swn_scoring(data.loc[idx]) # dict of scores for a row
    for x, key in enumerate(headlines_keys_sorted):
        # Append d into data frame 1         
        swn_df1.loc[idx,key] = d[key]
        colname1 = "pv" + str(x+1)
        colname2 = "nv" + str(x+1)
         # Append d into data frame 2
        swn_df2.loc[idx,colname1] = d[key][0]
        swn_df2.loc[idx,colname2] = d[key][1]
    
# Check the data frames
swn_df1.loc[0:5]
swn_df2.loc[0:5]

# Vader Sentiment Analysis
def row_vader_scoring(row):
    ''' 
        This function results in a dictionary containing records of positive 
        score, neutral score, negative score, and the respective sentence in 
        the format of (pos_val, neut_val, neg_val, sentence).
        
        Each sentiment score is calculated as the total of sentiment scores of
        each word in a sentence using Vader Sentiment Analyzer.
        
        The records are sorted in descending order based on the compound value and
        are reordered so that the first record ("Top1") has the highest compound
        value. 
    '''
    row_scores = dict()
    temp_row = row[2:len(row)]
    sid = SentimentIntensityAnalyzer()
    # Generate SWN Scores
    for idx, sentence in enumerate(temp_row):
        # some headlines are missing
        if not(isinstance(sentence,str)):
            row_scores[temp_row.index[idx]]  = (0, 0, 0, 0, "blank")
        else:
            score = sid.polarity_scores(sentence)
            row_scores[temp_row.index[idx]]  = (score['pos'],score['neu'],\
                                                score['neg'], score['compound'],
                                                sentence)
    # Sort the scores so that headlines with the highest positive scores are in the
    # leftmost part of the array and headlines with low positive scores are in the 
    # rightmost part of the array. This is to remove the effect of headlines position
    # to the weights of our model. The order of the news headlines does not matter 
    # in reality.
    temp = sorted(row_scores.items(), 
                  key=lambda x: (x[1][3]), reverse = True)
    keys = sorted(row_scores.keys(), key=lambda x: int(x[3:]))
    # Generate a dictionary with sorted keys
    scores_dict = dict()
    for idx, item in enumerate(temp):
        scores_dict[keys[idx]] = item[1]
    # Return the sorted dictionary
    return scores_dict


#for key in headlines_keys_sorted:
    #print row_scores[key]    
#    print d[key]
#d = row_vader_scoring(data.loc[277])

# Generate two data frames.
# vad_df1 stores tuples of (pos_score, neu_score, neg_score, compound_score, 
# sentence) for each sorted headline
# vad_df2 stores positive scores, neutral scores, and negative scores in 
# separate columns
vad_df1 = pd.DataFrame([], columns = headlines_keys_sorted,\
                        index = range(0,len(data.index)))
vad_df2 = pd.DataFrame([], index = range(0,len(data.index)))

for idx in range(0,len(data.index)): 
    d = row_vader_scoring(data.loc[idx]) # dict of scores for a row
    for x, key in enumerate(headlines_keys_sorted):
        # Append d into data frame 1         
        vad_df1.loc[idx,key] = d[key]
        colname1 = "pv" + str(x+1)
        colname2 = "neut" + str(x+1)
        colname3 = "nv" + str(x+1)
         # Append d into data frame 2
        vad_df2.loc[idx,colname1] = d[key][0]
        vad_df2.loc[idx,colname2] = d[key][1]
        vad_df2.loc[idx,colname3] = d[key][2]

# Check the data frames
vad_df1.loc[0]
vad_df2.loc[0]

# NEURAL NETWORK MODEL
# SentiWordNet Scores
# headline input, meant to receive sequences of 50 integers
scores_input = Input(shape=(50,), dtype="float32", name='swn_scores')

# the second input is a sequence of DJIA index for each day. 
djia_input = Input(shape=(1,), dtype="float32", name="djia_index")

# Two encoder layers
encoder_1 = Dense(10, activation='relu')(scores_input)
encoder_2 = Dense(10, activation='relu')(djia_input)

# LSTM layers
x = merge([encoder_1,encoder_2], mode='concat')
x = Reshape((1, 20))(x)
lstm = LSTM(4, return_sequences=False, dropout_W=0.5)(x)

# Add a logistic regression on top
predictions = Dense(1)(lstm)

# Compile neural network model
model = Model(input=[scores_input, djia_input], output = predictions)
model.compile(optimizer='adam',
              loss = 'mae',
              metrics=['accuracy'])

# Train and Test Data
# Before we split the data, we need to scale the values.
scaler = MinMaxScaler(feature_range=(0,1))
djia_scaled = scaler.fit_transform(djia_data.loc[:,'Close'])
swn_df2_scaled = scaler.fit_transform(swn_df2)
swn_df2_scaled[0:2]

# We take data before 2015-01-01 for our training data set. There are 1611 
# records for train data.
train_size = int(len(data[data['Date']<'2015-01-01']))
test_size = len(data.index) - train_size
swn_train_Xs = swn_df2_scaled[0:(train_size-1),:]
swn_train_Xd = djia_scaled[0:train_size-1]
swn_train_y = djia_scaled[1:train_size]
swn_test_Xs = swn_df2_scaled[train_size-1:len(data.index)-1,:]
swn_test_Xd = djia_scaled[train_size-1:len(data.index)-1]
swn_test_y = djia_scaled[train_size:len(data.index)]

# Train the model
model.fit({'swn_scores':swn_train_Xs, 'djia_index':swn_train_Xd},
          swn_train_y,
          nb_epoch = 20,
          batch_size = 30,
          verbose = 2)

# Make predictions
trainPredict = model.predict({'swn_scores':swn_train_Xs, 'djia_index':swn_train_Xd},
                             batch_size = 30)
testPredict = model.predict({'swn_scores':swn_test_Xs, 'djia_index':swn_test_Xd})      
                             
# Plot Results
plt.plot(np.append(trainPredict,testPredict))
plt.plot(djia_scaled)
plt.show()


# Using Vader scores
# Neural Network Model
scores_input_vad = Input(shape=(75,), dtype="float32", name='vad_scores') 
djia_input_vad = Input(shape=(1,), dtype="float32", name="djia_index")
encoder_1_vad = Dense(20, activation='relu')(scores_input_vad)
encoder_2_vad = Dense(20, activation='relu')(djia_input_vad)
# LSTM layers
x_vad = merge([encoder_1_vad,encoder_2_vad], mode='concat')
x_vad = Reshape((1, 40))(x_vad)
lstm_vad = LSTM(4, return_sequences=False, dropout_W=0.2, dropout_U=0.2)(x_vad)
# Add a logistic regression on top
predictions_vad = Dense(1)(lstm_vad)
# Compile neural network model
model_vad = Model(input=[scores_input_vad, djia_input_vad], 
                  output = predictions_vad)
model_vad.compile(optimizer='adam',
              loss = 'mae',
              metrics=['accuracy'])

# Create train and test data
vad_df2_scaled = scaler.fit_transform(vad_df2)
vad_df2_scaled[0:2]
# We take data before 2015-01-01 for our training data set. There are 1611 
# records for train data.
vad_train_Xs = vad_df2_scaled[0:(train_size-1),:]
vad_train_Xd = djia_scaled[0:train_size-1]
vad_train_y = djia_scaled[1:train_size]
vad_test_Xs = vad_df2_scaled[train_size-1:len(data.index)-1,:]
vad_test_Xd = djia_scaled[train_size-1:len(data.index)-1]
vad_test_y = djia_scaled[train_size:len(data.index)]

# Train the model 
model_vad.fit({'vad_scores':vad_train_Xs, 'djia_index':vad_train_Xd},
              vad_train_y,
              nb_epoch = 20,
              batch_size = 30,
              verbose = 2)

# Make predictions
trainPredict_vad = model_vad.predict({'vad_scores':vad_train_Xs, 'djia_index':vad_train_Xd},
                             batch_size = 30)
testPredict_vad = model_vad.predict({'vad_scores':vad_test_Xs, 'djia_index':vad_test_Xd})      
                             
# Plot Results
plt.plot(np.append(trainPredict_vad,testPredict_vad))
plt.plot(djia_scaled)
plt.show()

plt.plot(testPredict_vad)
plt.show()











