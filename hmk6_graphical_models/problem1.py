import math
import numpy as np
from collections import Counter
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 1: Naive Bayes Classifier (with discrete attributes)
    In this problem, you will implement the naive Bayes classification method. 
    In the data1.csv file, we have a collection of email spam detection data. 
    The class label indicate whether or not an email is spam (1: spam; 0: not spam).
    Each email has many features where each feature represents whether or not a certain word has appeared in the email.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
    Note: please don't use any existing package for classification problems, implement your own version.
'''

#-----------------------------------------------
def prob_smooth(X,c=2,k=1):
    '''
        Estimate the probability distribution of a random variable with Laplace smoothing.
        Input:
            X: the observed values of training samples, an integer numpy vector of length n. 
                Here n is the number of training instances. Each X[i] = 0,1, ..., or c-1. 
            c: the number of possible values of the variable, an integer scalar.
            k: the parameter of Laplace smoothing, an integer, denoting the number of imagined instances observed for each possible value. 
        Output:
            P: the estimated probability distribution, a numpy vector of length c.
                Each P[i] is the estimated probability of the i-th value.
    '''
    #########################################
    counter = Counter(X)
    P = np.zeros(shape=(c,))
    for i in range(c):
        P[i] = (counter[i]+k)/float(len(list(counter.elements())) + c*k)
    #########################################
    return P
    


#--------------------------
def class_prior(Y):
    '''
        Estimate the prior probability of Class labels: P(Class=y).
        Here we assume this is binary classification problem.
        Input:
            Y : the labels of training instances, an integer numpy vector of length n. 
                Here n is the number of training instances. Each Y[i] = 0 or 1. 
        Output:
            PY: the prior probability of each class, a numpy vector of length c.
    '''
    #########################################
    counter = Counter(Y)
    # c = np.max(list(counter.elements()))+1 # assuming highest cluster has at least one sample
    # if c == 1:
    #     c = 2  # at least two clusters, y=0,1. override c value if 1
    c = 2
    PY = np.zeros(shape=(c,))
    for i in range(c):
        PY[i] = counter[i]/float(len(list(counter.elements())))
    #########################################
    return PY


#--------------------------
def conditional_prob(X,Y,k=1):
    '''
        Estimate the conditional probability of P(X=x|Class=y) for each value of attribute X given each class.
        Input:
            X : the values of one attribute for training instances, an integer numpy vector of length n. 
                n is the number of training instances. 
                Here we assume X is a binary variable, with 0 or 1 values. 
            Y : the labels of training instances, an integer numpy vector of length n. 
                Each Y[i] = 0,1
            k: the parameter of Laplace smoothing, an integer, denoting the number of imagined instances observed for each possible value of X given each value of Y.
        Output:
            PX_Y: the probability of P(X|Class), a numpy array (matrix) of shape 2 by 2.
                  PX_Y[i,j] represents the probability of X=j given the class label is i.
    '''
    #########################################
    c = 2
    n = len(X)

    # need joint prob first
    x0y0 = 0
    x1y0 = 0
    x0y1 = 0
    x1y1 = 0
    for i in range(n):
        if X[i]==0 and Y[i]==0:
            x0y0 += 1
        if X[i]==1 and Y[i]==0:
            x1y0 += 1
        if X[i]==0 and Y[i]==1:
            x0y1 += 1
        if X[i] == 1 and Y[i] == 1:
            x1y1 += 1

    cXY = np.zeros(shape=(c, c))
    cXY[0,0] = x0y0
    cXY[1,0] = x1y0
    cXY[0,1] = x0y1
    cXY[1,1] = x1y1

    # determine conditional prob
    PX_Y = np.zeros(shape=(c, c))
    counter_y = Counter(Y)
    for i in range(c): # prob of X=i
        for j in range(c): # given Y=j
            PX_Y[i,j] = (cXY[i,j] + k) / float(counter_y[j] + k*c)
    PX_Y = PX_Y.T
    #########################################
    return PX_Y


#--------------------------
def train(X,Y,k=1):
    '''
        Training the model parameters on a training dataset.
        Input:
            X : the values of attributes for training instances, an integer numpy matrix of shape p by n. 
                p is the number of attributes. 
                n is the number of training instances. 
                Here we assume X is binary-valued, with 0 or 1 elements. 
            Y : the labels of training instances, an integer numpy vector of length n. 
                Each Y[i] = 0,1
            k: the parameter of Laplace smoothing, an integer, denoting the number of imagined instances observed for each possible value. 
        Output:
            PX_Y: the estimated probability of P(X|Class), a numpy array of shape p by 2 by 2.
                  PX_Y[i,j,k] represents the probability of the i-th attribute to have value k given the class label is j.
            PY: the estimated prior probability distribution of the class labels, a numpy vector of length 2.
                Each PY[i] is the estimated probability of the i-th class.
    '''
    #########################################
    c=2
    p = len(X)
    PX_Y = np.zeros(shape=(p, c, c))
    for i in range(p):
        PX_Y[i,:,:] = conditional_prob(X[i,:],Y,k)
    PY = class_prior(Y)
    #########################################
    return PX_Y, PY


#--------------------------
def inference(X,PY, PX_Y):
    '''
        Given a trained model, predict the label of one test instance in the test dataset.
        Input:
            X : the values of attributes for one test instance, an integer numpy vector of length p. 
                p is the number of attributes. 
                Here we assume X is binary-valued, with 0 or 1 elements. 
            PX_Y: the estimated probability of P(X|Class), a numpy array of shape p by 2 by 2.
                  PX_Y[i,j,k] represents the probability of the i-th attribute to have value k given the class label is j.
            PY: the estimated prior probability distribution of the class labels, a numpy vector of length 2.
                Each PY[i] is the estimated probability of the i-th class.
        Output:
            Y: the predicted class label, an integer scalar of value 0 or 1.
            P: the probability P(class | X), a float array of length 2.
                P[i] is the probability of the instance X in the i-th class.
    '''
    #########################################
    c=2
    p = len(X)
    P = np.zeros(shape=(c,))

    P0 = []
    P1 = []
    for i in range(p):
        # product of x0=x[0], x1=x[1], etc for y=0
        P0.append((PX_Y[i,0,X[i]])) # product of Pr{xi=x[i] | y=0}
        P1.append((PX_Y[i,1,X[i]])) # product of Pr{xi=x[i] | y=1}

    P[0] = np.prod(P0) * PY[0]
    P[1] = np.prod(P1) * PY[1]

    # normalize into a proper posterior
    P = P / P.sum()

    # predict label on highest posterior conditional probability
    Y = np.argmax(P)
    #########################################
    return Y, P


#--------------------------
def predict(X,PY, PX_Y):
    '''
        Given a trained model, predict the labels of test instances in the test dataset.
        Input:
            X : the values of attributes for test instances, an integer numpy matrix of shape p by n. 
                p is the number of attributes. 
                n is the number of test instances. 
                Here we assume X is binary-valued, with 0 or 1 elements. 
            PX_Y: the estimated probability of P(X|Class), a numpy array of shape p by 2 by 2.
                  PX_Y[i,j,k] represents the probability of the i-th attribute to have value k given the class label is j.
            PY: the estimated prior probability distribution of the class labels, a numpy vector of length 2.
                Each PY[i] is the estimated probability of the i-th class.
        Output:
            Y: the predicted class labels, a numpy vector of length n.
               Each Y[i] is the predicted label of the i-th instance.
    '''
    #########################################
    n = X.shape[1]
    Y = np.zeros(shape=(n,))
    for i in range(n):
        Y[i], _ = inference(X[:,i].T,PY, PX_Y)
    #########################################
    return Y

