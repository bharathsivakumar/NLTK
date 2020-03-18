import pandas as pd
import numpy as np 
import random 
from nltk import classify 
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.metrics import ConfusionMatrix
from sklearn.model_selection import learning_curve 
from sklearn import metrics 
import matplotlib.pyplot as plt 
from matplotlib.ticker import FuncFormatter

def plot_learning_curves_Naive_Bayes(training_data_df, testing_data_df,suptitle='Learning curve of a Naive Bayes Classifier', title='Estimating sentiment of a movie review', xlabel='Size of training dataset', ylabel='Accuracy'):
    """
    Plots learning curves for a Naive Bayes estimator
    Parameters
    ----------
    training_data_df : pd.DataFrame
        training set with labels where features are in 
    testing_data_df : pd.DataFrame
        training set (response)
    suptitle : str
        Chart suptitle
    title: str
        Chart title
    xlabel: str
        Label for the X axis
    ylabel: str
        Label for the y axis
    Returns
    -------
    Plot of learning curves
    """
    
    # create lists to store train and testing scores
    train_score = []
    test_score = []
    training_data_list = training_data_df.values.tolist()
    testing_data_list = testing_data_df.values.tolist()	
    # create ten incremental training set sizes
    training_set_sizes = np.linspace(5, len(training_data_list), num = 10, dtype='int')

    training_data_list = training_data_df.values.tolist()
    testing_data_list = testing_data_df.values.tolist()
    labels_testing_actual_list = list(testing_data_df.iloc[:,1].values)
    # for each one of those training set sizes

    for i in training_set_sizes:
        # fit the model only using that many training examples
        classifier = NaiveBayesClassifier.train(training_data_list[0:i])
        # calculate the training accuracy only using those training examples
        labels_training_predicted_list = list(training_data_df.iloc[0:i,0].apply(classifier.classify).values)
        labels_training_actual_list = list(training_data_df.iloc[0:i,1].values)
        labels_testing_predicted_list = list(testing_data_df.iloc[:,0].apply(classifier.classify).values)
        train_accuracy = metrics.accuracy_score(labels_training_actual_list,labels_training_predicted_list)
        # calculate the testing accuracy using the whole testing set
        
	
        test_accuracy = metrics.accuracy_score(labels_testing_predicted_list,labels_testing_actual_list)
        # store the scores in their respective lists
        train_score.append(train_accuracy)
        test_score.append(test_accuracy)

    # plot learning curves
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.plot(training_set_sizes, train_score, c='gold')
    ax.plot(training_set_sizes, test_score, c='steelblue')

    # format the chart to make it look nice
    fig.suptitle(suptitle, fontweight='bold', fontsize='20')
    ax.set_title(title, size=20)
    ax.set_xlabel(xlabel, size=16)
    ax.set_ylabel(ylabel, size=16)
    ax.legend(['training set', 'testing set'], fontsize=16)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylim(0, 1)

    def percentages(x, pos):
        """The two args are the value and tick position"""
        if x < 1:
            return '{:1.0f}'.format(x*100)
        return '{:1.0f}%'.format(x*100)

    def numbers(x, pos):
        """The two args are the value and tick position"""
        if x >= 1000:
            return '{:1,.0f}'.format(x)
        return '{:1.0f}'.format(x)

    y_formatter = FuncFormatter(percentages)
    ax.yaxis.set_major_formatter(y_formatter)

    x_formatter = FuncFormatter(numbers)
    ax.xaxis.set_major_formatter(x_formatter)
    plt.show()

data_model_df = pd.read_pickle('model_data.pickle')

split = 70.0 #Size of training set as a percentage of entire dataset 

training_size = int( split/100 * len(data_model_df) )

data_model_df = data_model_df.sample(frac=1).reset_index(drop=True) #specifying drop=True prevents .reset_index from creating a column containing the old index entries.

training_data_df = data_model_df[:training_size]
testing_data_df  =  data_model_df[training_size:]

plot_learning_curves_Naive_Bayes(training_data_df, testing_data_df)