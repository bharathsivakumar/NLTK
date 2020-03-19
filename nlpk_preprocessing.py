import pandas as pd
import numpy as np
import nltk 
from nltk.corpus import stopwords
from nltk.text import Text
import re #regular expressions that we'll use to remove unwated strings
import string

"""
In this module, we create 3 functions to pre-process a movie review dataset
which we use to figure out how to perform sentiment analysis with the help 
of a Naive Bayes classifier. The 3 functions do the following respectively: 
convert the review to lower case, remove punctuations etc., take a review 
in tokenized form and remove stop words from it, and finally convert a 
tokenized review to a dictionary format with words as keys and
value 'true' for all of them. This is required for the Naive Bayes 
classifier since it takes features in this format
"""

def clean_text(text): 
    """
    Parameters
    ----------
    text : string
        A text in string form in particular for 
        our case the string would be a movie review

    Returns
    -------
    Returns the string (review) after it has been preprocessed
    by doing the following: 
    1) converted to lower case
    2) numbers removed
    3) all words between <> removed including the <>
    4) Punctuations removed 
    """
    text = text.lower() #converts to lower case
    text = re.sub(r'\d+', '', text) #removes number 
    text = re.sub('\<.*?>', '', text) #removes (inclusive) all the words between <> 
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) 
    #string.punctuation has all punctuations, in the above line, we remove them 
    return text

def clean_tokenized_text(tokenized_text):
    """
    Parameters
    ----------
    tokenized_text : list of words (string typr)
        A text in a list of string form with the 
        text split off at punctuation marks or spaces
        it is list of words and in our particular case,
        the words come from a review 

    Returns
    -------
    Returns the list of strings with all the stopwords in the english language
    removed from the list. Stop words in english include all, just, don't, being,
    can etc. 
    """
    #stop_words below will contain all the stop words in English language as a list
    stop_words = set(stopwords.words('english')) 
    filtered_sentence = [] #empty list to store the words that are not stop words
    
    for w in tokenized_text:
        if w not in stop_words:
            #Below, any word that isn't a stop word is appended to empty list 
            filtered_sentence.append(w) 

    return filtered_sentence

def create_dict_of_tokenized_text(tokenized_text):
    """
    tokenized_text : list of words (string typr)
        A text in a list of string form with the 
        text split off at punctuation marks or spaces
        it is list of words and in our particular case,
        the words come from a review 

    Returns
    -------
    Returns a dictionary where each word in the list of strings
    tokenized_text as key and the value as true. We do this because 
    we will be using a Naive Bayes classifier to perform sentiment 
    analysis of our movie review and that alogorithm requires its 
    features to be in this form. 
    """

    dict_tokenized_text = dict([(word, True) for word in tokenized_text])
    return dict_tokenized_text