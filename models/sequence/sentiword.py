# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 18:04:28 2016

@author: Hizkia
"""

import pandas as pd
import numpy as np
import os 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import sentiwordnet as swn
from collections import defaultdict

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


