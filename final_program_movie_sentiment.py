import pandas as pd

import nltk

import nlpk_preprocessing #Module built by me to preprocess the dataset  
from nlpk_preprocessing import clean_text #function to preprocess the dataset after 
from nlpk_preprocessing import clean_tokenized_text
from nlpk_preprocessing import create_dict_of_tokenized_text

import predict_review_2
from predict_review_2 import estimate_learning_curves_Naive_Bayes
from predict_review_2 import plot_learning_curves
from predict_review_2 import confusion_matrix_Naive_Bayes 

text_dataset = "IMDB_Dataset.csv" #located in same directory as the code 

data_df = pd.read_csv(text_dataset) 
print("The first few entries in the dataset are:")
print( data_df.head() )

text_cleaned = pd.Series(data_df.iloc[:,0].apply(clean_text)) 
text_tokenized = text_cleaned.apply(nltk.word_tokenize)
text_tokenized_cleaned = text_tokenized.apply(clean_tokenized_text)
data_cleaned_df = pd.concat([text_tokenized_cleaned, data_df.iloc[:,1]], axis = 1)
data_cleaned_df.to_pickle('cleaned_data.pickle')

dict_review = data_cleaned_df.iloc[:,0].apply(create_dict_of_tokenized_text)
data_model_df = pd.concat([dict_review,data_cleaned_df.iloc[:,1] ], axis = 1)
print( data_model_df.head() )
data_model_df.to_pickle('model_data.pickle')

split = 70.0 #Size of training set as a percentage of entire dataset 

training_size = int( split/100 * len(data_model_df) )

data_model_df = data_model_df.sample(frac=1).reset_index(drop=True) #specifying drop=True prevents .reset_index from creating a column containing the old index entries.

training_data_df = data_model_df[:training_size]
testing_data_df  =  data_model_df[training_size:]

classifier, training_set_sizes, train_score, test_score,labels_testing_predicted_list, labels_testing_actual_list = estimate_learning_curves_Naive_Bayes(training_data_df, testing_data_df)

#We are now going to save our fully trained classifier into a pickle file so that we don't have to use re-train it
save_classifier = open("naive_bayes_sentiment_analysis.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

cm = confusion_matrix_Naive_Bayes(labels_testing_predicted_list, labels_testing_actual_list)
print("\nConfusion matrix on the testing dataset for our naive bayes classifier is:\n", cm)

plot_learning_curves(train_score, test_score, training_set_sizes)