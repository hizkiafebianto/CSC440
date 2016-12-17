'''
Reference: https://github.com/ryankiros/skip-thoughts
Paper: Skip-Thought Vectors
Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, Sanja Fidler
'''
import csv
import sys
import itertools
import numpy as np
import nltk
import re
import skipthoughts
import autoencoder
import h5py
import pandas
import collections
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.layers import Input, Embedding, merge, Flatten, Dense, SimpleRNN
from keras.models import Model
from keras.preprocessing import sequence
from keras.datasets import imdb
from nltk.corpus import stopwords





tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
model_skipthoughts = skipthoughts.load_model()
max_news = 1989
max_news_length = 445
embedding_dim = 4800


def read_csv(filepath):
	global n_news
	f = open(filepath)
	csv_f = list(csv.reader(f))
	news = list()
	for row in csv_f[1:]:
		news.append(list(row[2:]))
	return news


def string_clean(sentence):
	sentence = re.sub("[^a-zA-Z]"," ", sentence)
	sentence = re.sub("b"," ", sentence)
	words = tokenizer.tokenize(sentence.strip().lower())
	words = [word for word in words if len(word) > 2 and word not in stop_words]
	return words


def news_to_sentences(newslist):
	global max_news_length
	sentences = list()
	for news in newslist:
		sentence = list()
		for headline in news:
			sentence += string_clean(headline)
		if len(sentence) > max_news_length:
			max_news_length = len(sentence)
		sentences.append(sentence)
	return sentences



def word_embedding(newslist):
	embed_news = list()
	sentences = news_to_sentences(newslist)
	index = 0
	for sentence in sentences[0:1]:
		# sentence_pad = sequence.pad_sequences(sentence, maxlen = max_news_length)
		vector = skipthoughts.encode(model_skipthoughts, sentence)
		print vector
		save_hdf5(vector)
		vector_encoded = autoencoder.convnet_encoder(vector)
		print vector_encoded
		# vector_pad = np.zeros((max_news_length, embedding_dim))
		# vector_pad[:vector.shape[0],:vector.shape[1]] = vector
		# embed_news.append(vector_pad)
		# embed_news_array = np.asarray(embed_news, dtype = 'int32')
		# save_hdf5(embed_news)
		index += 1
		# print 'Saving embedded vector...'
		print 'Completed %d vectors...' % index
	return embed_news_array



def save_hdf5(data):
	with h5py.File('embednews.h5', mode = 'w') as hf:
		hf.create_dataset('data', data = data)
		hf.close()



def load_hdf5(filename):
	with h5py.File(filename, 'r') as hf:
		data = hf.get('data')
		numpy_array = np.array(data)
	return numpy_array


def main():
	sentences = '''
	sam is red
	hannah not red
	hannah is green
	bob is green
	bob not red
	sam not green
	sarah is red
	sarah not green'''.strip().split('\n')
	is_green = np.asarray([[0, 1, 1, 1, 1, 0, 0, 0]], dtype='int32').T

	lemma = lambda x: x.strip().lower().split(' ')
	sentences_lemmatized = [lemma(sentence) for sentence in sentences]
	words = set(itertools.chain(*sentences_lemmatized))
	# set(['boy', 'fed', 'ate', 'cat', 'kicked', 'hat'])

	# dictionaries for converting words to integers and vice versa
	word2idx = dict((v, i) for i, v in enumerate(words))
	idx2word = list(words)

	# convert the sentences a numpy array
	to_idx = lambda x: [word2idx[word] for word in x]
	sentences_idx = [to_idx(sentence) for sentence in sentences_lemmatized]
	sentences_array = np.asarray(sentences_idx, dtype='int32')

	# parameters for the model
	sentence_maxlen = 3
	n_words = len(words)
	n_embed_dims = 3

	# # put together a model to predict 
	# from keras.layers import Input, Embedding, merge, Flatten, SimpleRNN
	# from keras.models import Model

	input_sentence = Input(shape=(sentence_maxlen,), dtype='int32')
	input_embedding = Embedding(n_words, n_embed_dims)(input_sentence)
	color_prediction = SimpleRNN(5, return_sequences=False, batch_input_shape=(10, 2, 2))(input_embedding)
	output = Dense(1, activation='sigmoid')(color_prediction)


	predict_green = Model(input=[input_sentence], output=[output])
	predict_green.compile(optimizer='sgd', loss='binary_crossentropy')

	# fit the model to predict what color each person is
	predict_green.fit([sentences_array], [is_green], nb_epoch=5000, verbose=1)
	embeddings = predict_green.layers[1].W.get_value()

	# print out the embedding vector associated with each word
	for i in range(n_words):
		print('{}: {}'.format(idx2word[i], embeddings[i]))

if __name__ == "__main__":
	
	newslist = read_csv(sys.argv[1])
	# sentences = news_to_sentences(newslist)
	# print max_length, len(word_dictionary)
	# (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=30000)
	# print X_test
	# embed_news_array = word_embedding(newslist)
	# save_hdf5(embed_news_array)
	embed_news = load_hdf5('embednews.h5')
	print embed_news








	
