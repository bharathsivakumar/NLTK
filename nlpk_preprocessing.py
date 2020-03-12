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

dataset = "IMDB_Dataset.csv" #located in same directory as the code 
data_df = pd.read_csv(dataset)
print( data_df.head() )

def clean_review(review): # cleaning up our review data by removing unwanted stuff
    review = review.lower() #converts to lower case
    review = re.sub(r'\d+','',review) #removes number 
    review = re.sub('\<.*?>', '', review) #removes (inclusive) all the words between <> 
    review = re.sub('[%s]' % re.escape(string.punctuation), '', review) 
    #above line removes anything that is present in string.punctuation which would be punctuations
    return review


first_cleaning = lambda  x: clean_review(x)

review_cleaned = pd.DataFrame(data_df.review.apply(clean_review)) 
data_cleaned = pd.concat([review_cleaned.review, data_df.sentiment], axis = 1)
print( data_cleaned.head() )