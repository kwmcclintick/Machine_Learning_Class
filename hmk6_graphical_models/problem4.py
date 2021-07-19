#-------------------------------------------------------------------------
# Note: please don't use any additional package except the following packages
import numpy as np
from collections import Counter
#-------------------------------------------------------------------------
'''
    Problem 4: LDA (Latent Dirichlet Allocation) using Gibbs sampling method
    You could test the correctness of your code by typing `nosetests -v test4.py` in the terminal.

    Notations:
            ---------- dimensions ------------------------
            m: the number of data documents, an integer scalar.
            n: the number of words in each document, an integer scalar.
            p: the number of all possible words (the size of the vocabulary), an integer scalar.
            k: the number of topics, an integer scalar
            ---------- model parameters ----------------------
            alpha: the parameter of the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
                    Here we are assuming the parameters in all dimensions to have the same value.
            beta: the parameters of word distribution on k topics, a numpy float matrix of shape k by p. 
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
            -----------------------------------------------
'''

#------------------------------------------------------------------------------
#  Sampling Methods (for Bayesian Networks)
#------------------------------------------------------------------------------

'''
    Let's first practise with a simpler case: a network with only 3 random variables.

        X1 --->  X2  --->  X3

    Suppose we are sampling from the above Bayesian network.
 
'''

#--------------------------
def prior_sampling(n,PX1,PX2,PX3):
    '''
        Use prior sampling to sample data from the above graphical model. 
        Input:
            n:  the number of samples, an integer scalar. 
            PX1: the probability distribution of random variable X1, a float vector of length 2.
                 PX1[i] represents the probability of X1=i.
            PX2: the conditional probability distribution of random variable X2 given X1, a float matrix of shape 2 by 2.
                 PX2[i,j] represents the probability of X2=j given that X1=i.
            PX3: the conditional probability distribution of random variable X3 given X2, a float matrix of shape 2 by 2.
                 PX3[i,j] represents the probability of X3=j given that X2=i.
        Output:
            S: a collection of samples from the graphical model, an integer matrix of shape n by 3.
                S[i] represents the i-th sample, where S[i,j] represents the value of X(j+1).
    '''
    #########################################
    S = np.zeros(shape=(n, 3) ,dtype=np.int)
    for i in range(n):
        # assign X1
        S[i,0] = np.random.choice(a=PX1.shape[0], p=PX1) # pick random X1 state

        # assign X2
        if S[i,0] == 0: # if X1 = 0
            S[i, 1] = np.random.choice(a=PX2.shape[1], p=PX2[0,:]) # pick random X1 state
        else: # if X1 = 1
            S[i, 1] = np.random.choice(a=PX2.shape[1], p=PX2[1,:]) # pick random X1 state

        # assign X3
        if S[i,1] == 0: # if X2 = 0
            S[i, 2] = np.random.choice(a=PX3.shape[1], p=PX3[0,:]) # pick random X1 state
        else: # if X2 = 1
            S[i, 2] = np.random.choice(a=PX3.shape[1], p=PX3[1,:]) # pick random X1 state
    #########################################
    return S 



'''
    Now let's assume we observe the value of X2 (evidence).

        X1 --->  X2(observed)  --->  X3
    Use different sampling methods to sample data from the above Bayesian network.
'''
#--------------------------
def rejection_sampling(n,PX1,PX2,PX3, ev):
    '''
        Use rejection sampling to sample data from the above graphical model. 
        Input:
            n:  the number of samples, an integer scalar. 
            PX1: the probability distribution of random variable X1, a float vector of length 2.
                 PX1[i] represents the probability of X1=i.
            PX2: the conditional probability distribution of random variable X2 given X1, a float matrix of shape 2 by 2.
                 PX2[i,j] represents the probability of X2=j given that X1=i.
            PX3: the conditional probability distribution of random variable X3 given X2, a float matrix of shape 2 by 2.
                 PX3[i,j] represents the probability of X3=j given that X2=i.
            ev: the observed value of X2, an integer scalar of value 0 or 1.
        Output:
            S: a collection of samples from the graphical model, an integer matrix of shape n by 3.
                S[i] represents the i-th sample, where S[i,j] represents the value of X(j+1).
    '''
    #########################################
    S = np.zeros(shape=(n, 3), dtype=np.int)
    for i in range(n):
        # assign X1
        S[i, 0] = np.random.choice(a=PX2.shape[1], p=PX2[:,ev] / np.sum(PX2[:,ev]))  # pick random X1 state

        # assign X2, always equal to ev
        S[:, 1] = ev

        # assign X3
        if S[i, 1] == 0:  # if X2 = 0
            S[i, 2] = np.random.choice(a=PX3.shape[1], p=PX3[0, :])  # pick random X1 state
        else:  # if X2 = 1
            S[i, 2] = np.random.choice(a=PX3.shape[1], p=PX3[1, :])  # pick random X1 state
    #########################################
    return S 

#--------------------------
def importance_sampling(n,PX1,PX2,PX3, ev):
    '''
        Use importance (likelihood) sampling to sample data from the above graphical model. 
        Input:
            n:  the number of samples, an integer scalar. 
            PX1: the probability distribution of random variable X1, a float vector of length 2.
                 PX1[i] represents the probability of X1=i.
            PX2: the conditional probability distribution of random variable X2 given X1, a float matrix of shape 2 by 2.
                 PX2[i,j] represents the probability of X2=j given that X1=i.
            PX3: the conditional probability distribution of random variable X3 given X2, a float matrix of shape 2 by 2.
                 PX3[i,j] represents the probability of X3=j given that X2=i.
            ev: the observed value of X2, an integer scalar of value 0 or 1.
        Output:
            S: a collection of samples from the graphical model, an integer matrix of shape n by 3.
                S[i] represents the i-th sample, where S[i,j] represents the value of X(j+1).
            w: the weights of samples, a float vector of length n.
                w[i] denotes the weight(likelihood) of the i-th sample in S.
    '''
    #########################################
    S = np.zeros(shape=(n, 3), dtype=np.int)
    w = np.zeros(shape=(n,))
    for i in range(n):
        # assign X1
        S[i, 0] = np.random.choice(a=PX1.shape[0], p=PX1)  # pick random X1 state

        # assign weight for sample
        if S[i, 0] == 0:  # if X1 = 0
            w[i] = PX2[0,0]  # set weight for X2=0 given X1=0
        else:  # if X1 = 1
            w[i] = PX2[1,0]  # set weight for X2=0 given X1=1
        # either way, assign X2, always equal to ev
        S[:, 1] = ev

        # assign X3
        if S[i, 1] == 0:  # if X2 = 0
            S[i, 2] = np.random.choice(a=PX3.shape[1], p=PX3[0, :])  # pick random X1 state
        else:  # if X2 = 1
            S[i, 2] = np.random.choice(a=PX3.shape[1], p=PX3[1, :])  # pick random X1 state
    #########################################
    return S, w 


#------------------------------------------------------------------------------
# Gibbs Sampling
#------------------------------------------------------------------------------
'''
    Gibbs sampling: Let's switch to the following network (X3 is observed).
        X1 --->  X2  --->  X3 (observed)
'''

#--------------------------
def sample_X1(X2,X3,PX1,PX2,PX3):
    '''
        re-sample the value of X1 given the values of X2 and X3 
        Input:
            X2: the current value of X2 , an integer of value 0 or 1.
            X3: the current value of X3 , an integer of value 0 or 1.
            PX1: the probability distribution of random variable X1, a float vector of length 2.
                 PX1[i] represents the probability of X1=i.
            PX2: the conditional probability distribution of random variable X2 given X1, a float matrix of shape 2 by 2.
                 PX2[i,j] represents the probability of X2=j given that X1=i.
            PX3: the conditional probability distribution of random variable X3 given X2, a float matrix of shape 2 by 2.
                 PX3[i,j] represents the probability of X3=j given that X2=i.
        Output:
            X1: the generative sample of X1, an integer scalar of value 0 or 1.
    '''
    #########################################
    prob = PX1 * PX2[:,X2] / np.sum(PX1 * PX2[:,X2])
    X1 = np.random.choice(a=PX1.shape[0], p=prob)
    #########################################
    return X1 

#--------------------------
def sample_X2(X1,X3,PX1,PX2,PX3):
    '''
        re-sample the value of X2 given the values of X1 and X3 
        Input:
            X1: the current value of X1 , an integer of value 0 or 1.
            X3: the current value of X3 , an integer of value 0 or 1.
            PX1: the probability distribution of random variable X1, a float vector of length 2.
                 PX1[i] represents the probability of X1=i.
            PX2: the conditional probability distribution of random variable X2 given X1, a float matrix of shape 2 by 2.
                 PX2[i,j] represents the probability of X2=j given that X1=i.
            PX3: the conditional probability distribution of random variable X3 given X2, a float matrix of shape 2 by 2.
                 PX3[i,j] represents the probability of X3=j given that X2=i.
        Output:
            X2: the generative sample of X2, an integer scalar of value 0 or 1.
    '''
    #########################################
    probx20 = PX2[X1,0] * PX3[0, X3]  # P(X2=0 | X1=x1) * P(X3=x3 | X2=0)
    probx21 = PX2[X1, 1] * PX3[1, X3]  # P(X2=1 | X1=x1) * P(X3=x3 | X2=1)
    prob = np.array([probx20, probx21])  # normalize
    prob = prob / np.sum(prob)
    X2 = np.random.choice(a=PX1.shape[0], p=prob)
    #########################################
    return X2 

#--------------------------
def gibbs_sampling(n,X1,X2,X3,PX1,PX2,PX3):
    '''
        Use Gibbs sampling to sample data from the above graphical model. 
        Input:
            n:  the number of samples, an integer scalar. 
            X1: the initial values of X1, an integer scalar.
            X2: the initial values of X2, an integer scalar.
            X3: the observed values of X3, an integer scalar.
            PX1: the probability distribution of random variable X1, a float vector of length 2.
                 PX1[i] represents the probability of X1=i.
            PX2: the conditional probability distribution of random variable X2 given X1, a float matrix of shape 2 by 2.
                 PX2[i,j] represents the probability of X2=j given that X1=i.
            PX3: the conditional probability distribution of random variable X3 given X2, a float matrix of shape 2 by 2.
                 PX3[i,j] represents the probability of X3=j given that X2=i.
        Output:
            S: a collection of samples from the graphical model, an integer matrix of shape n by 3.
                S[i] represents the i-th sample, where S[i,j] represents the value of X(j+1).
    '''
    #########################################
    S = np.zeros(shape=(n, 3), dtype=np.int)
    for i in range(n):
        # sample X1
        S[i,0] = sample_X1(X2, X3, PX1, PX2, PX3)
        X1 = S[i,0]  # update X1
        # sample x2
        S[i,1] = sample_X2(X1, X3, PX1, PX2, PX3)
        X2 = S[i,1]  # update X2
        # assign X3, observed/fixed
        S[i,2] = X3
    #########################################
    return S 



#------------------------------------------------------------------------------
#  Gibbs Sampling Methods for LDA
#------------------------------------------------------------------------------

#--------------------------
def resample_z(w,d,z,nz,nzw,ndz,alpha=1.,eta=1.):
    '''
        Use Gibbs sampling to re-sample the topic (z) of one word in a document, and update the statistics (nz, nzw,nmz) accordingly.
        Input:
            w:  the index of the current word (in the vocabulary), an integer scalar. 
                if w = i, it means that the current word in the document is the i-th word in the vocabulary. 
            d:  the id of the current document, an integer scalar (ranging from 0 to m)
            z:  the current topic assigned the word, an integer scalar (ranging from 0 to k-1).
            nz:  the frequency counts for each topic, an integer vector of length k.
                The (i)-th entry is the number of times topic i is assigned in the corpus
            nzw:  the word frequence count for each topic , an integer matrix of shape k by p.
                The (i,j) entry of is the number of times the j-th word in the vocabulary is assigned to topic i.
            ndz:  the topic frequence count for each document, an integer matrix of shape m by k.
                The (i,j) entry is the number of words in document i assigned to topic j
            alpha: the parameter for topic prior (Dirichlet distribution), a float scalar.
            eta: the parameter for word prior (Dirichlet distribution), a float scalar.
        Output:
            z: the resampled topic of the current word
            p: the vector of probability of generating each topic for the current word, a float vector of length k.
                p[i] is the probability of the current word to be assigned to the i-th topic.
    '''
    #########################################
    # remove the current word from the statistics in nz,nzw,nmz
    nz[z] -= 1  # how much each topic appears [T1, T2]
    nzw[z,w] -= 1  # how much each word appears for each topic T1:[W1, W2]T2:[W1, W2]
    ndz[d,z] -= 1  # how much each topic appears in each doc D1:[T1, T2]D2:[T1,T2]

    # compute the probability of generating each topic for the current word.
    # Pr{z11 = T1 | everything} prop to Pr{z11|theta1}Pr{z11|beta(z11)}Pr{W1|z11}
    P = nzw.shape[1] # number of possible words
    K = len(nz) # number of possible topics
    p = np.zeros(shape=(K,))
    for k in range(K):
        p[k] = (ndz[d,k] + alpha) * (nzw[k,w] + eta) / (nz[k] + P*eta)
    # normalize to make it a PMF for sampling
    p = p / np.sum(p)

    # sample z according to the probability p
    z = np.random.choice(a=K, p=p)

    # update statistics (nz, nzw,nmz) with the newly sampled topic, z
    nz[z] += 1
    nzw[z,w] += 1
    ndz[d,z] += 1
    #########################################
    return z, p 

#--------------------------
def resample_Z(W,Z,nz,nzw,ndz,alpha=1.,eta=1.):
    '''
        Use Gibbs sampling to re-sample the topics (Z) of all words in all documents (for one pass), and update the statistics (nz, nzw,nmz) accordingly.
        Input:
            W:  the document matrix, an integer numpy matrix of shape m by n. 
                W[i,j] represents the index (in the vocabulary) of the j-th word in the i-th document.
                Here m is the number of text documents, an integer scalar.
                n: the number of words in each document, an integer scalar.
            Z:  the current topics assigned to all words in all documents, an integer matrix of shape m by n (value ranging from 0 to k-1).
            nz:  the frequency counts for each topic, an integer vector of length k.
                The (i)-th entry is the number of times topic i is assigned in the corpus
            nzw:  the word frequence count for each topic , an integer matrix of shape k by p.
                The (i,j) entry of is the number of times the j-th word in the vocabulary is assigned to topic i.
            ndz:  the topic frequence count for each document, an integer matrix of shape m by k.
                The (i,j) entry is the number of words in document i assigned to topic j
            alpha: the parameter for topic prior (Dirichlet distribution), a float scalar.
            eta: the parameter for word prior (Dirichlet distribution), a float scalar.
        Output:
            Z: the resampled topics of all words in all documents
    '''
    #########################################

    # c_z = Counter(Z[d,:])  # count occurances of each topic in document 'd'
    # occurances = list(c_z.values())  # sep counts from keys
    # for i in range(len(occurances)):
    #     S_nz[i] += occurances[i]  # cycle through all topic occurances and add to count accross all documents
    #     S_ndz[d, i] = occurances[i]  # set frequencies that each topic occurs in this document
    M = W.shape[0]
    N = W.shape[1]
    for d in range(M):  # iterate over all text documents
        for n in range(N):  # iterate over all possible words
            Z[d,n], p = resample_z(W[d,n], d, Z[d,n], nz, nzw, ndz, alpha, eta)
    #########################################
    return Z




#--------------------------
def gibbs_sampling_LDA(W,k,p,Z,alpha=1., eta=1.,n_samples=10, n_burnin=100, sample_rate=10):
    '''
        Use Gibbs sampling to generate a collection of samples of z (topic of each word) from LDA model.
        Input:
            W:  the document matrix, a float numpy matrix of shape m by n. 
                W[i,j] represents the index (in the vocabulary) of the j-th word in the i-th document.
                Here m is the number of text documents, an integer scalar.
                n: the number of words in each document, an integer scalar.
            k: the number of topics, an integer scalar.
            p: the number of words in the vocabulary, an integer scalar.
            Z:  the initial topics assigned to all words in all documents, an integer matrix of shape m by n (value ranging from 0 to k-1).
            n_samples: the number of samples to be drawn, an integer scalar.
            n_burnin: the number of samples to skip at the begining of the gibbs sampling, an integer scalar.
            sample_rate: for how many passes to select one sample, an integer scalaer. 
                         When the iteration number i % sample_rate = 0, take the sample for output.
                        For example, if sample_rate=3, n_burnin = 5, n_samples=4, the samples chosen (highlighted in parenthesis) will be on pass numbers: 0,1,2,3,4,5,(6),7,8,(9),10,11,(12),13,14,(15)
        Output:
            S_nz: the sum of nz counts in all the samples, an integer vector of length k.
            S_nzw: the sum of nzw counts in all the samples, an integer matrix of shape k by p.
            S_ndz: the sum of ndz counts in all the samples, an integer matrix of shape m by k.
    '''
    #########################################
    M = W.shape[0]
    N = W.shape[1]
    S_nz = np.zeros(shape=(k,))
    S_nzw = np.zeros(shape=(k, p))
    S_ndz = np.zeros(shape=(M, k))

    # initialize nz, nzw, ndz based upon W and Z
    for d in range(M): # iterate through all documents
        for n in range(N):
            S_nz[Z[d,n]] += 1
            S_nzw[Z[d,n], W[d,n]] += 1  # count occurances of each word within each topic
            S_ndz[d, Z[d,n]] += 1

    # collect one sample every sample_rate iterations
    n_collected = 0  # keep track of how many samples collected, break if finished
    for d in range(M): # cycle through all documents
        for n in range(N): # cycle through all words
            if n > n_burnin: # check to see if past burned samples
                if n % sample_rate == 0: # downsample
                    if n_collected > n_samples:
                        break
                    n_collected += 1  # now we can finally sample, increase counter
                    # remove counts caused by old sample
                    # S_nz[Z[d, n]] -= 1
                    # S_nzw[Z[d, n], W[d, n]] -= 1
                    # S_ndz[d, Z[d, n]] -= 1
                    temp_resampled_Z = resample_Z(W,Z,S_nz,S_nzw,S_ndz,alpha,eta)
                    Z[d,n] = temp_resampled_Z[d,n]  # change the sample
                    # add counts from new sample
                    S_nz[Z[d, n]] += 1
                    S_nzw[Z[d, n], W[d, n]] += 1
                    S_ndz[d, Z[d, n]] += 1
    #########################################
    return S_nz,S_nzw,S_ndz 

#--------------------------
def compute_theta(ndz, alpha=1.):
    '''
        compute theta based upon the statistics of the samples from Gibbs sampling.
        Input:
            ndz:  the topic frequence count for each document, an integer matrix of shape m by k.
                The (i,j) entry is the number of words in document i that are assigned to topic j
            alpha: the parameter for topic prior (Dirichlet distribution), a float scalar.
        Output:
            theta: the updated estimation of parameters for topic mixture of each document, a numpy float matrix of shape m by k.
                Each element theta[i] represents the vector of topic mixture in the i-th document. 
    '''
    #########################################
    M = ndz.shape[0]
    theta = np.zeros(shape=(ndz.shape))
    for m in range(M):
        theta[m,:] = ndz[m,:] + alpha
        theta[m, :] = theta[m,:] / np.sum(theta[m,:])  # normalize

    #########################################
    return theta 

#--------------------------
def compute_beta(nzw,eta=1.):
    '''
        compute beta based upon the statistics of the samples from Gibbs sampling. 
        Input:
            nzw:  the word frequence count for each topic , an integer matrix of shape k by p.
                The (i,j) entry of is the number of times the j-th word in the vocabulary is assigned to topic i.
            eta: the parameter for word prior (Dirichlet distribution), a float scalar.
        Output:
            beta: the updated estimation of parameters for word distribution on k topics, a numpy float matrix of shape k by p. 
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
    '''
    #########################################
    K = nzw.shape[0]
    beta = np.zeros(shape=(nzw.shape))
    for i in range(K):
        beta[i, :] = nzw[i, :] + eta
        beta[i, :] = beta[i, :] / np.sum(beta[i, :])  # normalize
    #########################################
    return beta 


#--------------------------
def LDA(W,k=3,p=100,alpha=.1,eta=1.,n_samples=10, n_burnin=100, sample_rate=10):
    '''
        Variational EM algorithm for LDA. 
        Input:
            W:  the document matrix, a float numpy matrix of shape m by n. 
                W[i,j] represents the index (in the vocabulary) of the j-th word in the i-th document.
                Here m is the number of text documents, an integer scalar.
            k: the number of topics, an integer scalar
            p: the number of all possible words (the size of the vocabulary), an integer scalar.
            alpha: the alpha parameter of the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
            eta: the eta parameter of the Dirichlet distribution for generating word distribution for each topic, a float scalar (eta>0).
            n_samples: the number of samples to be drawn in Gibbs sampling, an integer scalar.
            n_burnin: the number of samples to skip at the begining of the Gibbs sampling, an integer scalar.
            sample_rate: sampling rate for Gibbs sampling, an integer scalaer. 
        Output:
            alpha: the updated estimation of parameters alpha, a float scalar
            beta: the updated estimation of parameters for word distribution on k topics, a numpy float matrix of shape k by p. 
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
            gamma:  the optimal value for gamma, a numpy float vector of length k. 
            phi:  the optimal values for phi, a numpy float matrix of shape n by k.
    '''
    #########################################
    # randomly initialize Z
    M = W.shape[0]
    N = W.shape[1]
    Z = np.random.randint(low=0, high=k-1, size=(M,N))

    # Gibbs sampling
    S_nz, S_nzw, S_ndz = gibbs_sampling_LDA(W, k, p, Z, alpha, eta, n_samples, n_burnin, sample_rate)

    # compute theta
    theta = compute_theta(S_ndz, alpha)

    # compute beta
    beta = compute_beta(S_nzw, eta)
    #########################################
    return beta,theta




