import math
import numpy as np
from collections import Counter
import sys
#-------------------------------------------------------------------------
'''
    Problem 1: k nearest neighbor 
    In this problem, you will implement a classification method using k nearest neighbors. 
    The main goal of this problem is to get familiar with the basic settings of classification problems. 
    KNN is a simple method for classification problems.
    You could test the correctness of your code by typing `nosetests test1.py` in the terminal.
'''

#--------------------------
def Terms_and_Conditions():
    ''' 
        By submiting this homework or changing this function, you agree with the following terms:
       (1) Not sharing your code/solution with any student before and after the homework due. For example, sending your code segment to another student, putting your solution online or lending your laptop (if your laptop contains your solution or your dropbox automatically sychronize your solution between your home computer and your laptop) to another student to work on this homework will violate this term.
       (2) Not using anyone's code in this homework, build your own solution. For example, using some code segments from another student or online resources due to any reason (like too busy recently) will violate this term. Changing other people's code as your solution (such as changing the variable names) will also violate this term.
       (3) When discussing with any other student about this homework, only discuss high-level ideas or using psudo-code. Don't discuss about the solution at the code level. For example, discussing with another student about the solution of a function (which needs 5 lines of code to solve), and then working on the solution "independently", however the code of the two solutions are exactly the same, or only with minor differences  (like changing variable names) will violate this term.
      All violations of (1),(2) or (3) will be handled in accordance with the WPI Academic Honesty Policy.  For more details, please visit: https://www.wpi.edu/about/policies/academic-integrity/dishonesty
      Historical Data: in one year, we ended up finding 25% of the students in the class violating this term in their homework submissions and we handled ALL of these violations according to the WPI Academic Honesty Policy. 
    '''
    #########################################
    Read_and_Agree = True
    #########################################
    return Read_and_Agree
 
#--------------------------
def compute_distance(Xtrain, Xtest):
    '''
        compute the Euclidean distance between instances in a test set and a training set 
        Input:
            Xtrain: the feature matrix of the training dataset, a float python matrix of shape (n_train by p). Here n_train is the number of data instance in the training set, p is the number of features/dimensions.
            Xtest: the feature matrix of the test dataset, a float python matrix of shape (n_test by p). Here n_test is the number of data instance in the test set, p is the number of features/dimensions.
        Output:
            D: the distance between instances in Xtest and Xtrain, a float python matrix of shape (ntest, ntrain), the (i,j)-th element of D represents the Euclidean distance between the i-th instance in Xtest and j-th instance in Xtrain.
    '''
    #########################################
    D = np.zeros(shape=(np.shape(Xtest)[0], np.shape(Xtrain)[0]))

    i = 0
    for test in Xtest:
        j = 0
        for train in Xtrain:
            D[i][j] = np.linalg.norm(test - train)
            j += 1
        i += 1
    #########################################
    return D 



#--------------------------
def k_nearest_neighbor(Xtrain, Ytrain, Xtest, K = 3):
    '''
        compute the labels of test data using the K nearest neighbor classifier.
        Input:
            Xtrain: the feature matrix of the training dataset, a float numpy matrix of shape (n_train by p). Here n_train is the number of data instance in the training set, p is the number of features/dimensions.
            Ytrain: the label vector of the training dataset, an integer python list of length n_train. Each element in the list represents the label of the training instance. The values can be 0, ..., or num_class-1. num_class is the number of classes in the dataset.
            Xtest: the feature matrix of the test dataset, a float python matrix of shape (n_test by p). Here n_test is the number of data instance in the test set, p is the number of features/dimensions.
            K: the number of neighbors to consider for classification.
        Output:
            Ytest: the predicted labels of test data, an integer numpy vector of length ntest.
        Note: you cannot use any existing package for KNN classifier.
    '''
    #########################################
    # nubmer of testing instances
    N = np.shape(Xtest)[0]
    Ytest = np.ndarray(shape=(N,))

    # compute distances between test set and training set
    D = np.zeros(shape=(np.shape(Xtest)[0], np.shape(Xtrain)[0]))
    smallest = np.infty*np.ones(shape=(N, K))
    K_nn = np.zeros(shape=(N, K))
    i = 0
    for test in Xtest:
        j = 0
        for train in Xtrain:
            D[i][j] = np.linalg.norm(test - train)
            if np.any(D[i][j] < smallest[i]):
                smallest[i][np.argmax(smallest[i])] = D[i][j]
                K_nn[i][np.argmax(smallest[i])] = Ytrain[j]
            j += 1
        i += 1

    k = 0
    for nn in K_nn:
        counts = Counter(nn)
        Ytest[k] = counts.most_common(1)[0][0]
        k += 1

    #########################################
    return Ytest 

