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
import datetime as dt
import matplotlib.dates as mdates
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
from keras.layers import LSTM, Input, Dense, merge, Reshape, SimpleRNN
from keras.models import Model, Sequential
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
encoder_1 = Dense(15, activation='relu')(scores_input)
encoder_2 = Dense(15, activation='relu')(djia_input)

# LSTM layers
x = merge([encoder_1,encoder_2], mode='concat')
x = Reshape((1, 30))(x)
lstm = LSTM(4, return_sequences=False, dropout_W=0.2, dropout_U=0.2)(x)

# Add a logistic regression on top
predictions = Dense(1)(lstm)

# Compile neural network model
model = Model(input=[scores_input, djia_input], output = predictions)
model.compile(optimizer='adam',
              loss = 'mse',
              metrics=['accuracy'])

# Train and Test Data
# Before we split the data, we need to scale the values.
djia_scaler = MinMaxScaler(feature_range=(0,1))
scores_scaler = MinMaxScaler(feature_range=(0,1))
djia_scaled = djia_scaler.fit_transform(djia_data.loc[:,'Close'])
swn_df2_scaled = scores_scaler.fit_transform(swn_df2)

# We take data before 2015-01-01 for our training data set. There are 1611 
# records for train data.
train_size = int(len(data[data['Date']<'2015-01-01']))
test_size = len(data.index) - train_size
swn_train_Xs = swn_df2_scaled[test_size+1:len(data.index),:]
swn_train_Xd = djia_scaled[test_size+1:len(data.index)]
swn_train_y = djia_scaled[test_size:len(data.index)-1]
swn_test_Xs = swn_df2_scaled[1:test_size+1,:]
swn_test_Xd = djia_scaled[1:test_size+1]
swn_test_y = djia_scaled[0:test_size]


# Train the model
# swn_train_Xs[::-1] for reversing the data to match the timeline
start = dt.datetime.now()
model.fit({'swn_scores':swn_train_Xs[::-1], 'djia_index':swn_train_Xd[::-1]},
          swn_train_y[::-1],
          nb_epoch = 15,
          batch_size = 30,
          verbose = 2)
runtime_swn = dt.datetime.now() - start
print(runtime_swn)

# Make predictions
trainPredict = model.predict({'swn_scores':swn_train_Xs[::-1], 'djia_index':swn_train_Xd[::-1]},
                             batch_size = 30)
testPredict = model.predict({'swn_scores':swn_test_Xs[::-1], 'djia_index':swn_test_Xd[::-1]})      

# Plot Results
x = [dt.datetime.strptime(dm,'%Y-%m-%d').date() for dm in djia_data.loc[:,'Date']]
plt.plot(x[1:],np.append(testPredict[::-1],trainPredict[::-1]))
plt.plot(x,djia_scaled)
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
vad_df2_scaled = scores_scaler.fit_transform(vad_df2)

# We take data before 2015-01-01 for our training data set. There are 1611 
# records for train data.
vad_train_Xs = vad_df2_scaled[test_size+1:len(data.index),:]
vad_train_Xd = djia_scaled[test_size+1:len(data.index)]
vad_train_y = djia_scaled[test_size:len(data.index)-1]
vad_test_Xs = vad_df2_scaled[1:test_size+1,:]
vad_test_Xd = djia_scaled[1:test_size+1]
vad_test_y = djia_scaled[0:test_size]

# Train the model 
start = dt.datetime.now()
model_vad.fit({'vad_scores':vad_train_Xs[::-1], 'djia_index':vad_train_Xd[::-1]},
              vad_train_y[::-1],
              nb_epoch = 20,
              batch_size = 30,
              verbose = 2)
runtime_vad = dt.datetime.now() - start
print(runtime_vad)

# Make predictions
trainPredict_vad = model_vad.predict({'vad_scores':vad_train_Xs[::-1], 
                                      'djia_index':vad_train_Xd[::-1]},
                                       batch_size = 30)
testPredict_vad = model_vad.predict({'vad_scores':vad_test_Xs[::-1], 
                                     'djia_index':vad_test_Xd[::-1]})      
                             
# Plot Results
plt.plot(x[1:], np.append(testPredict_vad[::-1],trainPredict_vad[::-1]))
plt.plot(x,djia_scaled)
plt.show()


# PERFORMANCE MEASUREMENT
# Sentiword
train_pred_swn = djia_scaler.inverse_transform(trainPredict[::-1])
test_pred_swn = djia_scaler.inverse_transform(testPredict[::-1])
train_swn = djia_scaler.inverse_transform(swn_train_y)
test_swn = djia_scaler.inverse_transform(swn_test_y)
train_swn_score = math.sqrt(mean_squared_error(train_pred_swn,train_swn))
train_swn_score
test_swn_score = math.sqrt(mean_squared_error(test_pred_swn,test_swn))
test_swn_score

line1, = plt.plot(x[1:], np.append(test_pred_swn, train_pred_swn),
                  label='prediction')
line2, = plt.plot(x[:-1], np.append(test_swn, train_swn), 
                  label = 'actual')
legend = plt.legend(handles=[line1,line2],loc=2)
plt.ylabel("DJIA Index")
plt.title('Prices + SWN + LSTM')
plt.savefig("swn_LSTM.png")
plt.show()

# Vader 
# Invert predictions
train_pred_vad = djia_scaler.inverse_transform(trainPredict_vad[::-1])
test_pred_vad = djia_scaler.inverse_transform(testPredict_vad[::-1])
train_vad = djia_scaler.inverse_transform(vad_train_y)
test_vad = djia_scaler.inverse_transform(vad_test_y)

train_vad_score = math.sqrt(mean_squared_error(train_pred_vad,train_vad))
train_vad_score
test_vad_score = math.sqrt(mean_squared_error(test_pred_vad,test_vad))
test_vad_score

line1, = plt.plot(x[1:], np.append(test_pred_vad, train_pred_vad),
                  label='prediction')
line2, = plt.plot(x[:-1], np.append(test_vad, train_vad), 
                  label = 'actual')
legend = plt.legend(handles=[line1,line2],loc=2)
plt.ylabel("DJIA Index")
plt.title('Prices + VADER + LSTM')
plt.savefig("vad_LSTM.png")
plt.show()

# SIMPLE RNN MODELS
# SentiwordNet
# Model
scores_input_swnb = Input(shape=(50,), dtype="float32", name='swn_scores') 
djia_input_swnb = Input(shape=(1,), dtype="float32", name="djia_index")
encoder_1_swnb = Dense(15, activation='relu')(scores_input_swnb)
encoder_2_swnb = Dense(15, activation='relu')(djia_input_swnb)
# LSTM layers
x_swnb = merge([encoder_1_swnb,encoder_2_swnb], mode='concat')
x_swnb = Reshape((1, 30))(x_swnb)
RNN_swnb = SimpleRNN(4, return_sequences=False, dropout_W=0.2, dropout_U=0.2)(x_swnb)
# Add a logistic regression on top
predictions_swnb = Dense(1)(RNN_swnb)
# Compile neural network model
model_swnb = Model(input=[scores_input_swnb, djia_input_swnb], 
                  output = predictions_swnb)
model_swnb.compile(optimizer='adam',
              loss = 'mse',
              metrics=['accuracy'])
# Train the model
start = dt.datetime.now()
model_swnb.fit({'swn_scores':swn_train_Xs[::-1], 'djia_index':swn_train_Xd[::-1]},
          swn_train_y[::-1],
          nb_epoch = 15,
          batch_size = 30,
          verbose = 2)
runtime_swnb = dt.datetime.now()-start
print(runtime_swnb)
# Make predictions
trainPredict_swnb = model_swnb.predict({'swn_scores':swn_train_Xs[::-1], 
                                        'djia_index':swn_train_Xd[::-1]},
                                        batch_size = 30)
testPredict_swnb = model_swnb.predict({'swn_scores':swn_test_Xs[::-1], 
                                       'djia_index':swn_test_Xd[::-1]})      
# Plot Results
plt.plot(x[1:],np.append(testPredict_swnb[::-1],trainPredict_swnb[::-1]))
plt.plot(x,djia_scaled)
plt.show()
# Mean squared error calculation
train_pred_swnb = djia_scaler.inverse_transform(trainPredict_swnb[::-1])
test_pred_swnb = djia_scaler.inverse_transform(testPredict_swnb[::-1])
train_swnb = djia_scaler.inverse_transform(swn_train_y)
test_swnb = djia_scaler.inverse_transform(swn_test_y)
train_swnb_score = math.sqrt(mean_squared_error(train_pred_swnb,train_swnb))
train_swnb_score
test_swnb_score = math.sqrt(mean_squared_error(test_pred_swnb,test_swnb))
test_swnb_score

line1, = plt.plot(x[1:], np.append(test_pred_swnb, train_pred_swnb),
                  label='prediction')
line2, = plt.plot(x[:-1], np.append(test_swnb, train_swnb), 
                  label = 'actual')
legend = plt.legend(handles=[line1,line2],loc=2)
plt.ylabel("DJIA Index")
plt.title('Prices + SWN + SimpleRNN')
plt.savefig("swn_RNN.png")
plt.show()


# Vader
scores_input_vadb = Input(shape=(75,), dtype="float32", name='vad_scores') 
djia_input_vadb = Input(shape=(1,), dtype="float32", name="djia_index")
encoder_1_vadb = Dense(20, activation='relu')(scores_input_vadb)
encoder_2_vadb = Dense(20, activation='relu')(djia_input_vadb)
# LSTM layers
x_vadb = merge([encoder_1_vadb,encoder_2_vadb], mode='concat')
x_vadb = Reshape((1, 40))(x_vadb)
RNN_vadb = SimpleRNN(4, return_sequences=False, dropout_W=0.2, dropout_U=0.2)(x_vadb)
# Add a logistic regression on top
predictions_vadb = Dense(1)(RNN_vadb)
# Compile neural network model
model_vadb = Model(input=[scores_input_vadb, djia_input_vadb], 
                  output = predictions_vadb)
model_vadb.compile(optimizer='adam',
              loss = 'mse',
              metrics=['accuracy'])
start = dt.datetime.now()
model_vadb.fit({'vad_scores':vad_train_Xs[::-1], 'djia_index':vad_train_Xd[::-1]},
              vad_train_y[::-1],
              nb_epoch = 20,
              batch_size = 30,
              verbose = 2)
runtime_vadb = dt.datetime.now()-start
print(runtime_vadb)

# Make predictions
trainPredict_vadb = model_vadb.predict({'vad_scores':vad_train_Xs[::-1], 
                                        'djia_index':vad_train_Xd[::-1]},
                                        batch_size = 30)
testPredict_vadb = model_vadb.predict({'vad_scores':vad_test_Xs[::-1],
                                       'djia_index':vad_test_Xd[::-1]})                          
# Plot Results
plt.plot(x[1:],np.append(testPredict_vadb[::-1],trainPredict_vadb[::-1]))
plt.plot(x,djia_scaled)
plt.show() 
# Mean squared error calculation
train_pred_vadb = djia_scaler.inverse_transform(trainPredict_vadb[::-1])
test_pred_vadb = djia_scaler.inverse_transform(testPredict_vadb[::-1])
train_vadb = djia_scaler.inverse_transform(swn_train_y)
test_vadb = djia_scaler.inverse_transform(swn_test_y)
train_vadb_score = math.sqrt(mean_squared_error(train_pred_vadb,train_vadb))
train_vadb_score
test_vadb_score = math.sqrt(mean_squared_error(test_pred_vadb,test_vadb))
test_vadb_score      

line1, = plt.plot(x[1:], np.append(test_pred_vadb, train_pred_vadb),
                  label='prediction')
line2, = plt.plot(x[:-1], np.append(test_vadb, train_vadb), 
                  label = 'actual')
legend = plt.legend(handles=[line1,line2],loc=2)
plt.ylabel("DJIA Index")
plt.title('Prices + VADER + SimpleRNN')
plt.savefig("vad_RNN.png")
plt.show()


# LSTM MODEL WITHOUT NEWS HEADLINES AS INPUTS
dataset = djia_data.values[:,4:5].astype("float32")
djia_scaled = djia_scaler.fit_transform(dataset)
train = djia_scaled[test_size+1:len(djia_scaled)]
test = djia_scaled[1:test_size+1]
trainY = djia_scaled[test_size:len(djia_scaled)-1] 
testY = djia_scaled[0:test_size]

# Reshape the dataset for LSTM
train = np.reshape(train, (train.shape[0],1,train.shape[1]))
test = np.reshape(test, (test.shape[0],1,1))

# Sequential model with LSTM
model_LSTM = Sequential()
model_LSTM.add(LSTM(4, input_dim=1, return_sequences = False))
model_LSTM.add(Dense(1))
model_LSTM.compile(loss="mse", optimizer="adam")
start = dt.datetime.now()
model_LSTM.fit(train[::-1], trainY[::-1], nb_epoch=20, batch_size = 30, verbose=2)
runtime_LSTM = dt.datetime.now()-start
print(runtime_LSTM)

# Make predictions
trPredict = model_LSTM.predict(train[::-1])
tsPredict = model_LSTM.predict(test[::-1])
# plot
plt.plot(x[1:],np.append(tsPredict[::-1],trPredict[::-1]))
plt.plot(x,djia_scaled)
plt.show

# Performance measurement
train_pred_LSTM = djia_scaler.inverse_transform(trPredict[::-1])
test_pred_LSTM = djia_scaler.inverse_transform(tsPredict[::-1])
train_LSTM = djia_scaler.inverse_transform(swn_train_y) 
test_LSTM = djia_scaler.inverse_transform(swn_test_y)
train_LSTM_score = math.sqrt(mean_squared_error(train_pred_LSTM,train_LSTM))
train_LSTM_score
test_LSTM_score = math.sqrt(mean_squared_error(test_pred_LSTM,test_LSTM))
test_LSTM_score

line1, = plt.plot(x[1:], np.append(test_pred_LSTM, train_pred_LSTM),
                  label='prediction')
line2, = plt.plot(x[:-1], np.append(test_LSTM, train_LSTM), 
                  label = 'actual')
legend = plt.legend(handles=[line1,line2],loc=2)
plt.ylabel("DJIA Index")
plt.title('Prediction using Historical Prices Only')
plt.savefig("histprices_LSTM.png")
plt.show()


# Sequential model with SimpleRNN
model_RNN = Sequential()
model_RNN.add(SimpleRNN(4, input_dim=1, return_sequences = False))
model_RNN.add(Dense(1))
model_RNN.compile(loss="mse", optimizer="adam")
start = dt.datetime.now()
model_RNN.fit(train[::-1], trainY[::-1], nb_epoch=20, batch_size = 30, verbose=2)
runtime_RNN = dt.datetime.now()-start
print(runtime_RNN)

# Make predictions
trPredict_RNN = model_RNN.predict(train[::-1])
tsPredict_RNN = model_RNN.predict(test[::-1])
# plot
plt.plot(x[1:],np.append(tsPredict_RNN[::-1],trPredict_RNN[::-1]))
plt.plot(x,djia_scaled)
plt.show

# Performance measurement
train_pred_RNN = djia_scaler.inverse_transform(trPredict_RNN[::-1])
test_pred_RNN = djia_scaler.inverse_transform(tsPredict_RNN[::-1])
train_RNN = djia_scaler.inverse_transform(swn_train_y) 
test_RNN = djia_scaler.inverse_transform(swn_test_y)
train_RNN_score = math.sqrt(mean_squared_error(train_pred_RNN,train_RNN))
train_RNN_score
test_RNN_score = math.sqrt(mean_squared_error(test_pred_RNN,test_RNN))
test_RNN_score

line1, = plt.plot(x[1:], np.append(test_pred_RNN, train_pred_RNN),
                  label='prediction')
line2, = plt.plot(x[:-1], np.append(test_RNN, train_RNN), 
                  label = 'actual')
legend = plt.legend(handles=[line1,line2],loc=2)
plt.ylabel("DJIA Index")
plt.title('Prediction using Historical Prices Only')
plt.savefig("histprices_RNN.png")
plt.show()

# LINEAR REGRESSION
# SentiwordNet
from sklearn.linear_model import LinearRegression

trainX_swnl = swn_df2.values[test_size:len(swn_df2.values)]
trainY_swnl = djia_data.loc[test_size:len(swn_df2.values),'Close'].values
testX_swnl = swn_df2.values[0:test_size]
testY_swnl = djia_data.loc[0:test_size-1,'Close'].values


lin_reg = LinearRegression()
start = dt.datetime.now()
lin_reg.fit(trainX_swnl[::-1],trainY_swnl[::-1])
runtime_swnl = dt.datetime.now() - start
print(runtime_swnl)

tsPredict_swnl = lin_reg.predict(testX_swnl[::-1])
trPredict_swnl = lin_reg.predict(trainX_swnl[::-1])

### Save Figure
line1, = plt.plot(x, np.append(tsPredict_swnl[::-1],trPredict_swnl[::-1]),
                  label='prediction')
line2, = plt.plot(x, np.append(testY_swnl,trainY_swnl), label = 'actual')
legend = plt.legend(handles=[line1,line2],loc=2)
plt.ylabel("DJIA Index")
plt.title('Prediction using Linear Regression & SWN')
plt.savefig("linregSWN.png")
plt.show()

train_pred_swnl = trPredict_swnl[::-1]
test_pred_swnl = tsPredict_swnl[::-1]
train_swnl_score = math.sqrt(mean_squared_error(train_pred_swnl,trainY_swnl))
train_swnl_score
test_swnl_score = math.sqrt(mean_squared_error(test_pred_swnl,testY_swnl))
test_swnl_score

# VADER
trainX_vadl = vad_df2.values[test_size:len(vad_df2.values)]
trainY_vadl = djia_data.loc[test_size:len(vad_df2.values),'Close'].values
testX_vadl = vad_df2.values[0:test_size]
testY_vadl = djia_data.loc[0:test_size-1,'Close'].values

lin_reg_vadl = LinearRegression()
start = dt.datetime.now()
lin_reg_vadl.fit(trainX_vadl[::-1],trainY_vadl[::-1])
runtime_vadl = dt.datetime.now() - start
print(runtime_vadl)

tsPredict_vadl = lin_reg_vadl.predict(testX_vadl[::-1])
trPredict_vadl = lin_reg_vadl.predict(trainX_vadl[::-1])

### Save Figure
line1, = plt.plot(x, np.append(tsPredict_vadl[::-1],trPredict_vadl[::-1]),
                  label='prediction')
line2, = plt.plot(x, np.append(testY_vadl,trainY_vadl), label = 'actual')
legend = plt.legend(handles=[line1,line2],loc=2)
plt.ylabel("DJIA Index")
plt.title('Prediction using Linear Regression & VADER')
plt.savefig("linregVADER.png")
plt.show()

plt.plot(tsPredict_vadl)