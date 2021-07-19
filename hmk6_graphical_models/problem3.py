#-------------------------------------------------------------------------
# Note: please don't use any additional package except the following packages
import numpy as np
from scipy.special import psi,polygamma
from scipy.linalg import inv
#-------------------------------------------------------------------------
'''
    Problem 3: LDA (Latent Dirichlet Allocation) using Variational EM method
    In this problem, you will implement the Latent Dirichlet Allocation to model text data.
    You could test the correctness of your code by typing `nosetests -v test3.py` in the terminal.

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

#--------------------------
def variational_inference(w,beta, alpha=1., n_iter=100):
    '''
        Given a document, find the optimal values of the variational parameters: gamma and phi, using mean-field variational inference.
        Input:
            w:  the vector word ids of a document, an integer numpy vector of length n. 
                n: the number of words in the document, an integer scalar.
            beta: the current estimation of parameters for word distribution on k topics, a numpy float matrix of shape k by p. 
                p: the number of all possible words (the size of the vocabulary), an integer scalar.
                k: the number of topics, an integer scalar
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
            alpha: the parameter of the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
                    Here we are assuming the parameters in all dimensions to have the same value.
            n_iter: the number of iterations for iteratively updating gamma and phi. 
        Output:
            gamma:  the optimal value for gamma, a numpy float vector of length k. 
            phi:  the optimal values for phi, a numpy float matrix of shape n by k.
                Here k is the number of topics.
        Hint: you could use the psi() in scipy package to compute digamma function
    '''
    #########################################
    # init shape scalaras and outputs
    N = w.shape[0]
    K = beta.shape[0]
    gamma = N/float(K) + alpha*np.ones(shape=(K,))
    phi = (1/float(K))*np.ones(shape=(N,K))

    for _ in range(n_iter): # repeat until convergence
        for n in range(N): # iterate through words in document
            for i in range(K): # iterate through topics in document
                phi[n,i] = beta[i,w[n]]*np.exp(psi(gamma[i]))
            # normalize phi
            phi[n] = phi[n]/ np.sum(phi[n])
        # update gamma
        for i in range(K):
            gamma[i] = alpha + np.sum(phi[:,i])
    #########################################
    return gamma, phi

#--------------------------
def E_step(W,beta, alpha=1., n_iter=100):
    '''
        Infer the optimal values for variational parameters on all documents: phi and gamma.
        Input:
            W:  the document matrix, an integer numpy matrix of shape m by n. 
                Here m is the number of text documents, an integer scalar.
                n: the number of words in each document, an integer scalar.
            beta: the current estimation of parameters for word distribution on k topics, a numpy float matrix of shape k by p. 
                p: the number of all possible words (the size of the vocabulary), an integer scalar.
                k: the number of topics, an integer scalar
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
            alpha: the parameter of the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
                    Here we are assuming the parameters in all dimensions to have the same value.
            n_iter: the number of iterations for iteratively updating gamma and phi. 
        Output:
            gamma:  the optimal gamma values for all documents, a numpy float matrix of shape m by k. 
            phi:  the optimal phi values for all documents, a numpy float tensor of shape m by n by k. 
    '''
    #########################################
    M = W.shape[0]
    N = W.shape[1]
    K = beta.shape[0]
    gamma = np.asmatrix(np.zeros(shape=(M,K)))
    phi = np.zeros(shape=(M,N,K))

    for m in range(M):
        gamma[m], phi[m] = variational_inference(W[m], beta, alpha, n_iter)

    #########################################
    return gamma, phi 



#--------------------------
def update_beta(W, phi,p):
    '''
        update beta based upon the new values of the variational parameters. 
        Input:
            W:  the document matrix, a float numpy matrix of shape m by n. 
                Here m is the number of text documents, an integer scalar.
            phi:  the optimal phi values for all documents, a numpy float tensor of shape m by n by k. 
            p: the number of all possible words in the vocabulary.
        Output:
            beta: the updated estimation of parameters for word distribution on k topics, a numpy float matrix of shape k by p. 
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
    '''
    #########################################
    K = phi.shape[2]
    M = W.shape[0]
    N = W.shape[1]
    beta = np.asmatrix(np.zeros(shape=(K,p)))

    # compute
    for i in range(K): # iterate over all topics
        for j in range(p): # iterate over all possible words
            for d in range(M): # iterate over all text documents
                for n in range(N): # iterate over all observed words
                    if W[d,n] == j: # if the nth word of the dth doc is j, w^j_{d,n} = 1
                        wjdn = 1
                    else:
                        wjdn = 0
                    beta[i,j] += phi[d,n,i] * wjdn
    # normalize into a proper posterior
    beta = beta / beta.sum(axis=1)
    #########################################
    return beta 


#------------------------------------------------------------------------------
#  Newton's Method (for computing Alpha)
#------------------------------------------------------------------------------

'''
    Let's first practise with a simpler case: Newton's method in one dimensional space.
'''

#--------------------------
def newtons_1d(f,df,x,tol=1e-2):
    '''
        compute the root of function f using Newton's method.
        Input:
            f: the function f(x), a python function that takes a scalar x as input.
            df: the first derivative of function f(x), a python function that takes a scalar x as input.
            x: the initial solution, a float scalar.
            tol: the tolerance of error, a float scalar. 
                If the function values |f(x)| < tol  then stop the iteration.
        Output:
            x: a root of f(x), a float scalar 
        Hint: you could solve this problem using 2 lines of code.
    '''
    #########################################
    while True:
        if np.abs(f(x)) < tol:
            break
        else:
            x = x - f(x) / df(x)

    #########################################
    return x

#--------------------------
def min_newtons_1d(f,df,d2f,x,tol=1e-2):
    '''
        compute the minimium value of function f using Newton's method (1 dimensional input)
        Input:
            f: the function f(x), a python function that takes a scalar x as input.
            df: the first derivative of function f(x), a python function that takes a scalar x as input.
            d2f: the second derivative of function f(x), a python function that takes a scalar x as input.
            x: the initial solution, a float scalar.
            tol: the tolerance of error, a float scalar. 
                If the function values |df(x)| < tol  then stop the iteration.
        Output:
            x: a local optimal solution, where f(x) has a minimium value, a float scalar 
            v: a local minimium value of f(x), a float scalar 
    '''
    #########################################
    while True:
        if np.abs(df(x)) < tol:
            break
        else:
            x = x - df(x) / d2f(x)
            v = f(x)
    #########################################
    return x, v 


#--------------------------
def newtons(f,df,x,tol=1e-2):
    '''
        compute the root of function f using Multi-variate Newton's method (p dimensional space).
        Input:
            f: the function f(x), a python function that takes a p-dimensional vector x as input and output a p-dimensional vector.
            df: the first derivative of function f(x), a python function that takes a p-dimensional vector x as input and output a pXp matrix (Jacobian matrix)
            x: the initial solution, a float p-dimensional vector.
            tol: the tolerance of error, a float scalar. 
                If the function values |f(x)| < tol  then stop the iteration. You could use np.allclose() function and set 'atol=tol'.
        Output:
            x: a root of f(x), a float p-dimensional vector 
        Hint: you could use np.linalg.inv() to compute the inverse of a matrix
    '''
    #########################################
    while True:
        if np.all(np.less(np.abs(f(x)), tol)):
            break
        else:
            x = x - np.dot(np.linalg.inv(df(x)), f(x))
    #########################################
    return x

#--------------------------
def newtons_linear(f,h,z,x,tol=1e-2):
    '''
        compute the root of function f using Linear-time Multivariate Newton's method (p dimensional space).
        Input:
            f: the function f(x), a python function that takes a p-dimensional vector x as input and output a p-dimensional vector.
            df: the first derivative of function f(x), a python function that takes a p-dimensional vector x as input and output a pXp matrix (Jacobian matrix)
                Here Jacobian matrix is represented as diag(h) + 1z1^T
                h: the diagonal elements of Jacobian matrix, a python function outputing a float vector of length p.
                z: the constant background of Jacobian matrix, a python function outputing  a float scalar.
            x: the initial solution, a float p-dimensional vector.
            tol: the tolerance of error, a float scalar. 
                If the function values |f(x)| < tol  then stop the iteration. You could use np.allclose() function and set 'atol=tol'.
        Output:
            x: a root of f(x), a float p-dimensional vector 
        Hint: you could use np.linalg.inv() to compute the inverse of a matrix
    '''
    #########################################
    # slide 136
    while True:
        if np.all(np.less(np.abs(f(x)), tol)):
            break
        else:
            c = np.sum(f(x)/h(x)) / (z(x)**-1 + np.sum(1./h(x)))
            x = x - (f(x) - c) / h(x)
    #########################################
    return x


#------------------------------------------------------------------------------
'''
    Let's use Newton's method to compute alpha 
    For simplicity, we assume alpha is a symmetric prior: alpha vector has the same value on all topics.
    So this problem can be solved by one-dimension Newton's method.
    (For assymetric priors, you will need to use linear-time Newton's method to compute alpha.)
'''

#--------------------------
def compute_df(alpha,gamma):
    '''
        compute the first derivative of ELBO function w.r.t. alpha 
        Here we assume alpha is a symmetric prior, and the output  is a sum of the first derivative values on all topics.
        Input:
            alpha: the current estimate of parameter alpha in the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
                    Here we assume alpha is a symmetric prior, where alpha in all topics have the same value.
            gamma:  the gamma values for all documents, a numpy float matrix of shape m by k. 
                    Here k is the number of topics,  m is the number of documents.
        Output:
            df: the first derivative of ELBO function w.r.t. alpha, a float scalar
    '''
    #########################################
    # slide 135
    # dL/da_i = M(digam(alpha_0) - digam(alpha_i))  +  sum^M_{d=1} [digam(gamma_{d,i}) - digam(gamma_0)]
    M = gamma.shape[0]
    K = gamma.shape[1]

    alpha0 = K*alpha # if alpha were a vector it would be K long with each element equal to alpha
    df_dai = np.zeros(shape=(K,))
    for i in range(K):
        df_dai[i] = M*(psi(alpha0) - psi(alpha)) # M(digam(alpha_0) - digam(alpha_i))
        for d in range(M):
            gamma0 = np.sum(gamma[d,:]) # gamma_0 = sum all gamma_i
            df_dai[i] += psi(gamma[d,i]) - psi(np.sum(gamma0)) # sum^M_{d=1} [digam(gamma_{d,i}) - digam(gamma_0)]

    df = np.sum(df_dai) # sum all dL/dai_i to get scalar
    #########################################
    return df 

#--------------------------
def compute_d2f(alpha,m,k):
    '''
        compute the second derivative of ELBO function w.r.t. alpha.
        Here we assume alpha is a symmetric prior, and the output  is a sum of all the values in Hessian matrix.
        Input:
            alpha: the current estimate of parameter alpha in the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
                    Here we assume alpha is a symmetric prior, where alpha in all topics have the same value.
            m: the number of documents
            k: the number of topics
        Output:
            d2f: the second derivative of ELBO function w.r.t. alpha, a float scalar 
        Hint: you could use the polygamma() in scipy package to compute first derivative of the digamma function
    '''
    #########################################
    # slide 135
    # d^2L/da_ida_j = dirac(i,j)*M*digam'(alpha_i) - digam'(alpha_0)
    alpha0 = k*alpha
    d2f = 0
    for i in range(k):
        d2f += m*polygamma(1, alpha) - polygamma(1, alpha0)
        print(polygamma(1, alpha0))
    #########################################
    return d2f 



#--------------------------
def update_alpha(alpha,gamma,tol=1e-2):
    '''
        update alpha using linear-time Newton method.
        Input:
            alpha: the current estimate of parameter alpha in the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
                    Here we assume alpha is a symmetric prior, where alpha in all topics have the same value.
            gamma:  the gamma values for all documents, a numpy float matrix of shape m by k. 
            tol: the tolerance of error, a float scalar. 
                If the function values |df(x)| < tol  then stop the iteration. 
        Output:
            alpha: the updated estimation of parameters alpha, a float scalar
        Hint: alpha is a variable with value >0, but log alpha is a variable that can take any real-value. We need to apply Newton's method on log alpha, instead of alpha.
    '''
    #########################################
    m = gamma.shape[0]
    k = gamma.shape[1]
    while True:
        # find df without x
        df = compute_df(alpha, gamma)
        # check for convergence
        if np.abs(df) < tol:
            break
        else:
            # convert alpha to log
            loga = np.log(alpha)
            # calculate first, second derivative
            df = compute_df(loga, gamma)
            d2f = compute_d2f(loga,m,k)
            # compute using newtons method
            loga = loga - df / (d2f * alpha + df)
            # convert back to alpha
            alpha = np.exp(loga)
            break
    #########################################
    return alpha


#--------------------------
def M_step(W,phi,p,alpha,gamma,tol_alpha=1e-2):
    '''
        M step of the EM algorithm. 
        Input:
            W:  the document matrix, a float numpy matrix of shape m by n. 
                Here m is the number of text documents, an integer scalar.
            phi:  the optimal phi values for all documents, a numpy float tensor of shape m by n by k. 
            p: the number of all possible words in the vocabulary.
            alpha: the current estimate of parameter alpha in the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
                    Here we assume alpha is a symmetric prior, where alpha in all topics have the same value.
            gamma:  the gamma values for all documents, a numpy float matrix of shape m by k. 
            tol_alpha: the tolerance of error for Newton's method, a float scalar. 
                If the function values |df(x)| < tol  then stop the iteration. 
        Output:
            alpha: the updated estimation of parameters alpha, a float scalar
            beta: the updated estimation of parameters for word distribution on k topics, a numpy float matrix of shape k by p. 
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
    '''
    alpha = update_alpha(alpha,gamma,tol_alpha)
    beta = update_beta(W,phi,p)
    return alpha,beta



#--------------------------
def LDA(W,k=3,p=100,alpha=.1,n_iter_var=100,n_iter_em=10,tol_alpha=1e-2):
    '''
        Variational EM algorithm for LDA. 
        Input:
            W:  the document matrix, a float numpy matrix of shape m by n. 
                Here m is the number of text documents, an integer scalar.
            p: the number of all possible words (the size of the vocabulary), an integer scalar.
            k: the number of topics, an integer scalar
            alpha: the initial value of the alpha parameter of the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
            n_iter_var: the number of iterations in variational inference (E-step). 
            n_iter_em: the number of iterations in EM
            tol_alpha: the tolerance of error for Newton's method, a float scalar. 
                If the function values |df(x)| < tol  then stop the iteration. 
        Output:
            alpha: the updated estimation of parameters alpha, a float scalar
            beta: the updated estimation of parameters for word distribution on k topics, a numpy float matrix of shape k by p. 
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
            gamma:  the optimal value for gamma, a numpy float vector of length k. 
            phi:  the optimal values for phi, a numpy float matrix of shape n by k.
    '''
    # initialize beta (for testing purpose)
    beta = np.arange(float(k*p)).reshape((k,p))+1.
    for i in range(k):
        beta[i] = beta[i] /sum(beta[i])

    for _ in range(n_iter_em):
        #########################################
        # expectation step
        gamma, phi = E_step(W, beta, alpha, n_iter_var)
        # maximization step
        alpha, beta = M_step(W,phi,p,alpha,gamma,tol_alpha)
        #########################################
    return alpha, beta, gamma, phi


