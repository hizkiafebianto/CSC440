from __future__ import division
import csv
import sys
import itertools
import numpy as np
import nltk
import re
import skipthoughts
import h5py
import pandas
import collections
import preprocess
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.layers import Input, Embedding, Merge, Flatten, Dense, SimpleRNN, LSTM, Dropout, Reshape, GRU, TimeDistributed, RepeatVector
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.datasets import imdb
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

seed = 7
np.random.seed(seed)
max_news_length = 350
embedding_vecor_length = 8
top_words = 32136
look_back = 1
scaler = MinMaxScaler(feature_range=(0, 1))
# scaler_news = MinMaxScaler(feature_range=(0, 1))

'''
Reference: https://github.com/ryankiros/skip-thoughts
https://blog.keras.io/building-autoencoders-in-keras.html
Paper: Skip-Thought Vectors
Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, Sanja Fidler
'''

def load_data_class(filepath_news, filepath_stock):
	newslist = preprocess.read_news(filepath_news)
	sentences = preprocess.news_to_sentences(newslist)
	X = preprocess.sentences_to_nparray(sentences)
	stockprices = preprocess.read_stock(filepath_stock)
	y = preprocess.stock_process(stockprices)
	return X, y

def load_data_nonclass(filepath_news, filepath_stock):
	newslist = preprocess.read_news(filepath_news)
	sentences = preprocess.news_to_sentences(newslist)
	prices = preprocess.read_price(filepath_stock)
	prices = scaler.fit_transform(prices)
	news = preprocess.sentences_to_nparray(sentences)
	# news = scaler_news.fit_transform(news)
	hisprice, y = preprocess.data_process(prices, look_back)
	return news[look_back:], hisprice, y

def train_model_no_dropout(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = seed)
	model = Sequential()
	model.add(Embedding(top_words, embedding_vecor_length, input_length = max_news_length))
	model.add(LSTM(100))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# print(model.summary())
	model.fit(X_train, y_train, nb_epoch = 3, batch_size = 64)
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

def train_model_dropout(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = seed)
	model = Sequential()
	model.add(Embedding(top_words, embedding_vecor_length, input_length = max_news_length))
	model.add(LSTM(100, dropout_W = 0.5, dropout_U = 0.5))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# print(model.summary())
	model.fit(X_train, y_train, nb_epoch = 50, batch_size = 64)
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

def train_model_convnet(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = seed)
	model = Sequential()
	model.add(Embedding(top_words, embedding_vecor_length, input_length = max_news_length))
	model.add(Convolution1D(nb_filter = 32, filter_length = 3, border_mode = 'same', activation = 'relu'))
	model.add(MaxPooling1D(pool_length=2))
	model.add(LSTM(100))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# print(model.summary())
	model.fit(X_train, y_train, nb_epoch=3, batch_size=64)
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))



def train_model_combine(news, hisprice, y, prices):
	news_train, news_test, hisprice_train, hisprice_test, y_train, y_test = train_test_split(news, hisprice, y, test_size = 0.33, random_state = seed)
	# hisprice_train = np.reshape(hisprice_train, (hisprice_train.shape[0], 1, hisprice_train.shape[1]))
	# hisprice_test = np.reshape(hisprice_test, (hisprice_test.shape[0], 1, hisprice_test.shape[1]))



	# create and fit the LSTM network
	left_branch = Sequential()
	left_branch.add(Embedding(top_words, embedding_vecor_length, input_length = max_news_length))
	left_branch.add(LSTM(output_dim = 50, dropout_W = 0.5, dropout_U = 0.5, return_sequences = False))
	left_branch.add(Dense(10, activation='relu'))

	right_branch = Sequential()
	# right_branch.add(LSTM(output_dim = 50, activation='sigmoid', input_dim = look_back, return_sequences = False))
	right_branch.add(Dense(10, activation='relu', input_dim = 1))

	merged = Merge([left_branch, right_branch], mode = 'concat')
	predict_price = Sequential()
	predict_price.add(merged)
	predict_price.add(Reshape((1, 20)))
	predict_price.add(LSTM(output_dim = 10, return_sequences = False, input_shape = (1, 20)))
	# predict_price.add(Dense(50, activation = 'relu'))
	predict_price.add(Dense(1))
	predict_price.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
	print(predict_price.summary())

	predict_price.fit([news_train, hisprice_train], y_train, nb_epoch=50, batch_size = 64)
	
	# make predictions
	train_predict = predict_price.predict([news_train, hisprice_train])
	test_predict = predict_price.predict([news_test, hisprice_test])


	# invert predictions
	trainPredict = scaler.inverse_transform(train_predict)
	trainY = scaler.inverse_transform([y_train])
	testPredict = scaler.inverse_transform(test_predict)
	testY = scaler.inverse_transform([y_test])
	print trainPredict, trainY
	print testPredict, testY

	# calculate root mean squared error
	trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
	print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
	print('Test Score: %.2f RMSE' % (testScore))


	shift train predictions for plotting
	prices = np.asarray([prices]).T
	trainPredictPlot = np.empty_like(prices)
	trainPredictPlot[:, :] = np.nan
	trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
	print trainPredictPlot

	# shift test predictions for plotting
	testPredictPlot = np.empty_like(prices)
	testPredictPlot[:, :] = np.nan
	testPredictPlot[len(trainPredict)+(look_back):len(prices), :] = testPredict
	print testPredictPlot

	# plot baseline and predictions
	plt.plot(scaler.inverse_transform(prices[::-1]))
	plt.plot(trainPredictPlot)
	plt.plot(testPredictPlot)
	plt.show()

if __name__ == "__main__":
	news, hisprice, y = load_data_nonclass(sys.argv[1], sys.argv[2])
	prices = preprocess.read_price(sys.argv[2])
	train_model_combine(news, hisprice, y, prices)
	# X, y = load_data_class(sys.argv[1], sys.argv[2])
	# train_model_no_dropout(X, y)




