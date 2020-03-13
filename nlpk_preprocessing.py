import pandas as pd
import numpy as np
import nltk 
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.text import Text
import string, re 
import matplotlib.pyplot as plt 
import re
import string

text_dataset = "IMDB_Dataset.csv" #located in same directory as the code 
data_df = pd.read_csv(text_dataset)
print( data_df.head() )

def clean_text(text): # cleaning up our review data by removing unwanted stuff
    text = text.lower() #converts to lower case
    text = re.sub(r'\d+','',text) #removes number 
    text = re.sub('\<.*?>', '', text) #removes (inclusive) all the words between <> 
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) 
    #above line removes anything that is present in string.punctuation which would be punctuations
    return text

def clean_tokenized_text(text):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    for w in text:
        if w not in stop_words:
            filtered_sentence.append(w)
    
    return filtered_sentence

text_cleaned = pd.Series(data_df.iloc[:,0].apply(clean_text)) 
text_tokenized = text_cleaned.apply(nltk.word_tokenize)
text_tokenized_cleaned = text_tokenized.apply(clean_tokenized_text)

data_cleaned_df = pd.concat([text_tokenized_cleaned, data_df.iloc[:,1]], axis = 1)
print( data_cleaned_df.head() )