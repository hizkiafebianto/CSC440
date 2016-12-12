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
from keras.layers import Input, Embedding, merge, Flatten, Dense, SimpleRNN, LSTM, Dropout
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.datasets import imdb
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

seed = 7
np.random.seed(seed)
max_news_length = 350
embedding_vecor_length = 32
top_words = 32136

def load_data(filepath_news, filepath_stock):
	newslist = preprocess.read_news(filepath_news)
	sentences = preprocess.news_to_sentences(newslist)
	X = preprocess.sentences_to_nparray(sentences)
	stockprices = preprocess.read_stock(filepath_stock)
	y = preprocess.stock_process(stockprices)
	return X, y
	

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
	model.add(LSTM(100, dropout_W = 0.2, dropout_U = 0.2))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# print(model.summary())
	model.fit(X_train, y_train, nb_epoch = 3, batch_size = 64)
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

if __name__ == "__main__":
	X, y = load_data(sys.argv[1], sys.argv[2])
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = seed)
	train_model_convnet(X, y)




