import pandas as pd
import random 
from nltk import classify 
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.metrics import ConfusionMatrix
from sklearn.model_selection import learning_curve 
import matplotlib.pyplot as plt 

data_model_df = pd.read_pickle('model_data.pickle')
#print(data_model_df.head())
split = 70.0 #Size of training set as a percentage of entire dataset 

training_size = int( split/100 * len(data_model_df) )

data_model_df = data_model_df.sample(frac=1).reset_index(drop=True) #specifying drop=True prevents .reset_index from creating a column containing the old index entries.

training_data_df = data_model_df[:training_size]
testing_data_df  =  data_model_df[training_size:]

training_data_list = training_data_df.values.tolist()
testing_data_list = testing_data_df.values.tolist()

classifier = NaiveBayesClassifier.train(training_data_list)

labels_testing_predicted_list = list(testing_data_df.iloc[:,0].apply(classifier.classify).values)
labels_testing_acual_list = list(testing_data_df.iloc[:,1].values)

# You could use the Scikit learn confusion matrix here which directly gives
# an n x n matrix 
cm = ConfusionMatrix(labels_testing_predicted_list, labels_testing_acual_list)

print("\nConfusion matrix of the model is:\n", cm)

total_correct_predictions = cm['negative','negative'] + cm['positive','positive']
total_wrong_predictions = cm['negative','positive'] + cm['positive','negative']
accuracy = total_correct_predictions/(total_wrong_predictions + total_correct_predictions) 
print("\Accuracy of the model on the testing data set is ", accuracy)

test_review = "Children of Men is an incredible movie"
token_test_review = word_tokenize(test_review)
dict_test_review = dict([(word, True) for word in token_test_review])

print("\n Sentiment for input review is", classifier.classify(dict_test_review) )
dist = classifier.prob_classify(dict_test_review)

for label in dist.samples():
    print( "%s: %f" % (label, dist.prob(label)) )