import pandas as pd
import numpy as np 
from nltk import NaiveBayesClassifier
from nltk.metrics import ConfusionMatrix
from sklearn import metrics 
import matplotlib.pyplot as plt 
import pickle 

def estimate_learning_curves_Naive_Bayes(training_data_df, testing_data_df):
    """
    Returns: the final trained Naive Bayes estimator trained 
    on the entier training set,the accuracy of the model on 
    the training and test datasets with increasing training 
    set siz, the predicted labels for the training 
    dataset as a list and the actual labels of the training 
    dataset as a list 

    Parameters
    ----------
    training_data_df : pd.DataFrame
        training set with labels where features are in 
        dictionary form with the words as keys and true 
        as the value for each key 
    testing_data_df : pd.DataFrame
        testing set in the same format
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
    final trained Naive Bayes estimator 
    the accuracies of the model on the training set as a list  
    the accuracies of the model on the test set as a list  
    """
    
    # create lists to store train and testing scores
    train_score = []
    test_score = []

    #converting the training set data frame and testing set dataframe
    #into list since that is the format in which NLTK Naive Bayes 
    # classifier takes its input as 
    training_data_list = training_data_df.values.tolist()
    testing_data_list = testing_data_df.values.tolist()	

    # create ten incremental training set sizes
    training_set_sizes = np.linspace(5, len(training_data_list), num = 10, dtype = 'int')
    
    #converting the labels for the testing dataset into a list since 
    #we need in this form to evaluate the accuracy of our classifier 
    #on the testing dataset using the sklearn metric 
    labels_testing_actual_list = list(testing_data_df.iloc[:, 1].values)

    # for each one of those training set sizes
    for i in training_set_sizes:
        # fit the model only using that many training examples
        classifier = NaiveBayesClassifier.train(training_data_list[0 : i])

        # calculate the training accuracy only using those training examples
        labels_training_predicted_list = list(training_data_df.iloc[0 : i, 0].apply(classifier.classify).values)
        labels_training_actual_list = list(training_data_df.iloc[0 : i, 1].values)
        labels_testing_predicted_list = list(testing_data_df.iloc[:, 0].apply(classifier.classify).values)
        train_accuracy = metrics.accuracy_score(labels_training_actual_list, labels_training_predicted_list)

        # calculate the testing accuracy using the whole testing set 
        test_accuracy = metrics.accuracy_score(labels_testing_predicted_list, labels_testing_actual_list)

        # store the scores in their respective lists
        train_score.append(train_accuracy)
        test_score.append(test_accuracy)

    return classifier, training_set_sizes, train_score, test_score, labels_testing_predicted_list, labels_testing_actual_list

# plot learning curves
def plot_learning_curves(train_score, test_score, training_set_sizes, 
                         suptitle='Learning curve of a Naive Bayes Classifier',
                         title='Estimating sentiment of a movie review', 
                         xlabel='Size of training dataset', ylabel='Accuracy'):

    """
    Returns the plot of the learning curve 
    as training accuracy and test accuracy 
    with increasing size of the training set

    Parameters
    ----------
    training_score : list
        training accuracies as a list 
    testing_score : list
        testing accuracies as a list 
    training_set_sizes : np.array
        an array containg the set of training sizes 
        for which the accuracies will be plotted 
    Returns
    -------
    Returns the plot of the learning curve 
    as training accuracy and test accuracy 
    with increasing size of the training set
    """
    fig, ax = plt.subplots(figsize = (14, 9))
    ax.plot(training_set_sizes, train_score, c = 'gold')
    ax.plot(training_set_sizes, test_score, c = 'steelblue')

    # format the chart to make it look nice
    fig.suptitle(suptitle, fontweight = 'bold', fontsize = '20')
    ax.set_title(title, size = 20)
    ax.set_xlabel(xlabel, size = 16)
    ax.set_ylabel(ylabel, size = 16)
    ax.legend(['training set', 'testing set'], fontsize = 16)
    ax.tick_params(axis = 'both', labelsize = 12)
    ax.set_ylim(0, 1)

    plt.show()

def confusion_matrix_Naive_Bayes(labels_testing_predicted_list, labels_testing_actual_list):
    return ConfusionMatrix(labels_testing_predicted_list, labels_testing_actual_list)