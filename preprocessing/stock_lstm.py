# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:46:21 2016

@author: Hizkia

Testing an LSTM algorithm to predict DJIA value based on historical values

"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM


#Load the dataset
URL = "https://raw.githubusercontent.com/yuewang0319/CSC440/develop/preprocessing/DJIA.csv"
dataset = pd.read_csv(URL, delimiter=',', header=0)

dataset.head()

#