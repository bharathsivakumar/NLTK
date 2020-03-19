import pandas as pd

import nltk

import nlpk_preprocessing #Module built to preprocess the dataset  
from nlpk_preprocessing import clean_text 
from nlpk_preprocessing import clean_tokenized_text
from nlpk_preprocessing import create_dict_of_tokenized_text

import predict_review_2
from predict_review_2 import estimate_learning_curves_Naive_Bayes
from predict_review_2 import plot_learning_curves
from predict_review_2 import confusion_matrix_Naive_Bayes 

import pickle 
"""
In this program, we shall use the two modules that we have created 
before: nlpk_preprocessing and predict_review_2 together to 
finally construct our Naive Bayes Classifier that we will use 
to perform sentiment analysis on a movie review. 
This program starts out with taking a csv file which contains over
40000 reviews and its corresponding sentiment as input. We shall
use this dataset to train our model and predict the sentiment of 
a new review that it hasn't seen before
"""

text_dataset = "IMDB_Dataset.csv" #input move review csv file, located in same directory as the code 

data_df = pd.read_csv(text_dataset) #reading the csv file and loading it into a dataframe
print("The first few entries in the dataset are:")
print( data_df.head() )

#using the clean_text function below in nlpk_preprocessing module to 
#remove unnecessary stuff like punctuations and numbers from our reviews:
text_cleaned = pd.Series(data_df.iloc[:, 0].apply(clean_text))  

text_tokenized = text_cleaned.apply(nltk.word_tokenize) #Tokenizing our preprocessed review
#using the clean_text function below in nlpk_preprocessing module to 
#remove stopwrods from our reviews:
text_tokenized_cleaned = text_tokenized.apply(clean_tokenized_text) 
#Creating a new dataframe below where put the cleaned tokeinzed reviews as our 
#first column and its corresponding sentiments in the second column 
data_cleaned_df = pd.concat([text_tokenized_cleaned, data_df.iloc[:, 1]], axis = 1)
data_cleaned_df.to_pickle('cleaned_data.pickle')#Saving new dataframe to a pickle file for future use

#We need our training features in the form of a dictionary with words as keys
#and true as values for Naive Bayes so we convert our tokenized reviews into
#the same dictionary type down below 
dict_review = data_cleaned_df.iloc[:, 0].apply(create_dict_of_tokenized_text)
#Again we create another dataframe below where put the dictionaries as our 
#first column and its corresponding sentiments in the second column 
data_model_df = pd.concat([dict_review,data_cleaned_df.iloc[:, 1] ], axis = 1)
print( data_model_df.head() )
data_model_df.to_pickle('model_data.pickle') #Saving new dataframe to a pickle file for future use

split = 70.0 #Size of training set as a percentage of entire dataset 

training_size = int(split / 100 * len(data_model_df)) #Size of required training set

#Before we start training on this model directly, we need to shuffle the dataset
#to prevent the model from picking up patterns that may not exist in the data 
#specifying drop=True above prevents .reset_index from creating a column 
#containing the old index entries. we specify frac 
data_model_df = data_model_df.sample(frac = 1).reset_index(drop = True) 

#We store the appropriate datasets in new variables  
training_data_df = data_model_df[:training_size]
testing_data_df  =  data_model_df[training_size:]

#The estimate_learning_curves_Naive_Bayes function in our predict_review_2 module
#is used to get our classifier, training set sizes, training and test scores as an
#array over the training set sizes as well as the predicted sentiments for our 
#test dataset 
classifier, training_set_sizes, train_score, test_score,labels_testing_predicted_list, labels_testing_actual_list = estimate_learning_curves_Naive_Bayes(training_data_df, testing_data_df)

#We are now going to save our fully trained classifier below into a pickle file 
#so that we don't have to use re-train it in the future 
save_classifier = open("naive_bayes_sentiment_analysis.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

#We are useing the plot_learning_curves function in predict_review_2
#module to plot the learning curves of our training and test scores 
#as a function of the training set sizes 
plot_learning_curves(train_score, test_score, training_set_sizes)

#We are going to use the confusion_matrix_Naive_Bayes function in predict_review_2
#module to get the confusion matrix for our classifier and use the same
#to get the accuract of our modle 
cm = confusion_matrix_Naive_Bayes(labels_testing_predicted_list, labels_testing_actual_list)
print("\nConfusion matrix on the testing dataset for our naive bayes classifier is:\n", cm)
total_correct_predictions = cm['negative','negative'] + cm['positive','positive']
total_wrong_predictions = cm['negative','positive'] + cm['positive','negative']
accuracy = total_correct_predictions/(total_wrong_predictions + total_correct_predictions) 
print("Accuracy of the model on the testing data set is ", accuracy)

#Now that we have our classifier and accuracy for our model, we can
#try to make some predictions on a test review that we have written
test_review = "The Room is an incredibly poorly made movie. Makes me believe that I can be a great director one day"
token_test_review = nltk.word_tokenize(test_review)
dict_test_review = dict([(word, True) for word in token_test_review])
print("\n Sentiment for input review is", classifier.classify(dict_test_review) )

#Getting the prediction is one thing but let's also have a look at 
#the probablities that our model has assigned for each label for the
#test review 

#extracts the probability distribution for our test review in dictionary form
#over all the possible labels
dist = classifier.prob_classify(dict_test_review) 
#dist.samples() extracts all the labels from our distribution and 
#we try to print the probability for each of the label using a loop
for label in dist.samples():
    print( "%s: %f" % (label, dist.prob(label)) )