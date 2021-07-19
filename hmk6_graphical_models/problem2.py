import numpy as np
from scipy.stats import multivariate_normal
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 1: EM algorithm for Gaussian Mixture Model (GMM).
    In this problem, you will implement the expectation-maximization method for Gaussian Mixture Model.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''

#-------------------------------------------------------------------------
'''
    Let's first work on a simpler problem:    Maximum Likelihood Estimation (MLE) of one single Gaussian Distribution.
    Pleaese implement the maximium likelihood estimation of the parameters with a set of observed data points sampled from the distribution.
'''

#--------------------------
def compute_mu_mle(X):
    '''
        Compute the MLE estimation of mean of a Gaussian distribution. 
        Input:
            X: the feature matrix of a collection of observed samples from an unknown Gaussian distribution, a numpy matrix of shape p by n
                Here n is the number of samples, p is the number of features
        Output:
            mu: the estimated mean of the Gaussian Distribution, a numpy vector of shape (p by 1).
    '''
    #########################################
    mu = np.mean(X, axis=1)
    #########################################
    return mu 

#--------------------------
def compute_sigma_mle(X):
    '''
        Compute the MLE estimation of covariance of a Gaussian distribution. 
        Input:
            X: the feature matrix of a collection of observed samples from an unknown Gaussian distribution, a numpy matrix of shape p by n
                Here n is the number of samples, p is the number of features
        Output:
            sigma: the estimated covariance matrix of the Gaussian Distribution, a numpy matrix of shape (p by p).
    '''
    #########################################
    sigma = np.asmatrix(np.cov(X))
    #########################################
    return sigma 


#-------------------------------------------------------------------------
'''
   Now let's work on a mixture of multiple Gaussian Distributions. Estimate the parameters with expectation-maximization.
'''


#--------------------------
def E_step(X,mu,sigma,PY):
    '''
        E-step: Given the current estimate of model parameters, compute the expected mixture of components on each data point. 
        Input:
            X: the feature matrix of data samples, a numpy matrix of shape n by p
                Here n is the number of samples, p is the number of features
                X[i] is the i-th data sample.
            mu: the list of mean of each Gaussian component, a float matrix of k by p.
                k is the number of components in Gaussian mixture.
                p is the number dimensions in the feature space.
                mu[i] is the mean of the i-th component.
            sigma: the list of co-variance matrix of each Gaussian component, a float tensor of shape k by p by p.
                sigma[i] is the covariance matrix of the i-th component.
            PY: the probability of each component P(Y=i), a float vector of length k.
        Output:
            Y: the estimated label of each data point, a numpy matrix of shape n by k.
                Y[i,j] represents the probability of the i-th data point being generated from j-th Gaussian component.
        Hint: you could use multivariate_normal.pdf() to compute the density funciont of Gaussian ditribution.
    '''
    #########################################
    n = X.shape[0]
    k = mu.shape[0]
    Y = np.asmatrix(np.zeros(shape=(n,k)))

    # compute improper posteriors
    for i in range(n):
        for j in range(k):
            Y[i,j] =  multivariate_normal.pdf(X[i],mu[j],sigma[j])*PY[j]

    # normalize into a proper posterior
    Y = Y / Y.sum(axis=1)
    #########################################
    return Y 


#--------------------------
def M_step(X,Y):
    '''
        M-step: Given the current estimate of label distribution, update the parameters of GMM. 
        Input:
            X: the feature matrix of data samples, a numpy matrix of shape n by p
                Here n is the number of samples, p is the number of features
                X[i] is the i-th data sample.
            Y: the estimated label of each data point, a numpy matrix of shape n by k.
                Y[i,j] represents the probability of the i-th data point being generated from j-th Gaussian component.
        Output:
            mu: the list of mean of each Gaussian component, a float matrix of k by p.
                k is the number of components in Gaussian mixture.
                p is the number dimensions in the feature space.
                mu[i] is the mean of the i-th component.
            sigma: the list of co-variance matrix of each Gaussian component, a float tensor of shape k by p by p.
                sigma[i] is the covariance matrix of the i-th component.
            PY: the probability of each component P(Y=i), a float vector of length k.
    '''
    #########################################
    p = X.shape[1]
    k = Y.shape[1]
    N = X.shape[0]
    mu = np.zeros(shape=(k,p))
    sigma = np.zeros(shape=(k, p, p))

    # PY is mean probability of each class over each data point, since Y is soft decision and not actual labels
    PY = np.mean(Y, axis=0)
    PY = np.squeeze(np.asarray(PY))
    for i in range(k):
        for n in range(N):
            mu[i] += Y[n,i]*X[n,:]
        mu[i] = mu[i] /  np.sum(Y[:,i])
        for n in range(N):
            #print((np.reshape(X[n,:], newshape=(p,1)) - np.reshape(mu[i], newshape=(p,1)))*(X[n,:] - mu[i]))
            sigma[i] += Y[n,i]*(np.reshape(X[n,:], newshape=(p,1)) - np.reshape(mu[i], newshape=(p,1)))*(X[n,:] - mu[i])
        sigma[i] = sigma[i] / np.sum(Y[:,i])
    #########################################
    return mu,sigma,PY



#--------------------------
def EM(X,k=2,num_iter=10):
    '''
        EM: Given a set of data samples, estimate the parameters and label assignments of GMM. 
        Input:
            X: the feature matrix of data samples, a numpy matrix of shape n by p
                Here n is the number of samples, p is the number of features
                X[i] is the i-th data sample.
            k: the number of components in Gaussian mixture, an integer scalar.
            num_iter: the number EM iterations, an integer scalar.
        Output:
            Y: the estimated label of each data point, a numpy matrix of shape n by k.
                Y[i,j] represents the probability of the i-th data point being generated from j-th Gaussian component.
            mu: the list of mean of each Gaussian component, a float matrix of k by p.
                p is the number dimensions in the feature space.
                mu[i] is the mean of the i-th component.
            sigma: the list of co-variance matrix of each Gaussian component, a float tensor of shape k by p by p.
                sigma[i] is the covariance matrix of the i-th component.
            PY: the probability of each component P(Y=i), a float vector of length k.
    '''
    # initialization (for testing purpose, we use first k samples as the initial value of mu) 
    n,p = X.shape
    mu = X[:k]

    sigma = np.zeros((k,p,p))
    for i in range(k):
        sigma[i] = np.eye(p)
    PY = np.ones(k)/k

    #########################################
    for i in range(num_iter):
        Y = E_step(X,mu,sigma,PY)
        mu, sigma, PY = M_step(X,Y)
    #########################################
    return Y,mu,sigma,PY

