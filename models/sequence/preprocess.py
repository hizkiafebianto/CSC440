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
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.layers import Input, Embedding, merge, Flatten, Dense, SimpleRNN
from keras.models import Model
from keras.preprocessing import sequence
from keras.datasets import imdb
from nltk.corpus import stopwords





tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
max_news_length = 350
embedding_dim = 32
max_news = 1989


def read_csv(filepath):
	global n_news
	f = open(filepath)
	csv_f = list(csv.reader(f))
	newslist = list()
	for row in csv_f[1:]:
		newslist.append(list(row[2:]))
	return newslist


def string_clean(sentence):
	sentence = re.sub("[^a-zA-Z]"," ", sentence)
	sentence = re.sub("b"," ", sentence)
	words = tokenizer.tokenize(sentence.strip().lower())
	words = [word for word in words if len(word) > 2 and word not in stop_words]
	return words


def news_to_sentences(newslist):
	sentences = list()
	word_counter = collections.Counter()
	for news in newslist:
		sentence = list()
		for headline in news:
			sentence += string_clean(headline)
		sentences.append(sentence)
	return sentences

def sentences_to_nparray(sentences):
	words = set(itertools.chain(*sentences))
	word2idx = dict((v, i) for i, v in enumerate(words))
	idx2word = list(words)
	to_idx = lambda x: [word2idx[word] for word in x]
	sentences_idx = [to_idx(sentence) for sentence in sentences]
	sentences_array = np.asarray(sequence.pad_sequences(sentences_idx, maxlen = max_news_length), dtype = 'int32')
	return sentences_array


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
	sentences = news_to_sentences(newslist)
	sentences_array = sentences_to_nparray(sentences)
	print sentences_array



