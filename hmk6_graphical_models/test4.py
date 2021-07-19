from problem4 import *
import numpy as np
import sys

'''
    Unit test 4:
    This file includes unit tests for problem4.py.
    You could test the correctness of your code by typing `nosetests -v test4.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 4 (30 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)

#-------------------------------------------------------------------------
def test_prior_sampling():
    '''(2 points) prior_sampling'''
    PX1 = np.array([.5,.5])
    PX2 = np.array([[.8,.2],
                    [.3,.7]])
    PX3 = np.array([[.1,.9],
                    [.6,.4]])
    S=prior_sampling(1000,PX1,PX2,PX3)
    assert S.dtype==int
    assert np.allclose(((S[:,0]==0).sum())/1000, .5, atol=0.1)
    idx = (S[:,0]==0)
    assert np.allclose(((S[idx,1]==0).sum())/(idx.sum()), .8, atol=0.1)
    idx = (S[:,0]==1)
    assert np.allclose(((S[idx,1]==0).sum())/(idx.sum()), .3, atol=0.1)
    idx = (S[:,1]==0)
    assert np.allclose(((S[idx,2]==0).sum())/(idx.sum()), .1, atol=0.1)
    idx = (S[:,1]==1)
    assert np.allclose(((S[idx,2]==0).sum())/(idx.sum()), .6, atol=0.1)
 

#-------------------------------------------------------------------------
def test_rejection_sampling():
    '''(2 points) rejection_sampling'''
    PX1 = np.array([.5,.5])
    PX2 = np.array([[.8,.2],
                    [.3,.7]])
    PX3 = np.array([[.1,.9],
                    [.6,.4]])
    ev = 0
    S=rejection_sampling(1000,PX1,PX2,PX3,ev)
    assert S.shape == (1000,3)
    assert S.dtype==int


    assert np.allclose(((S[:,0]==0).sum())/1000, .4/.55, atol=0.05)
    assert np.allclose((S[:,1]==0).sum(), 1000, atol=0.1)
    assert np.allclose((S[:,2]==0).sum()/1000, .1, atol=0.1)
 
    ev = 1
    S=rejection_sampling(1000,PX1,PX2,PX3,ev)
    assert S.dtype==int

    assert np.allclose(((S[:,0]==0).sum())/1000, 2/9, atol=0.05)
    assert np.allclose((S[:,1]==1).sum(), 1000, atol=0.1)
    assert np.allclose((S[:,2]==0).sum()/1000, .6, atol=0.1)
 

#-------------------------------------------------------------------------
def test_importance_sampling():
    '''(2 points) importance_sampling'''
    PX1 = np.array([.5,.5])
    PX2 = np.array([[.8,.2],
                    [.3,.7]])
    PX3 = np.array([[.1,.9],
                    [.6,.4]])
    ev = 0
    S,w = importance_sampling(1000,PX1,PX2,PX3,ev)
    assert S.dtype==int
    assert S.shape == (1000,3)
    assert w.shape == (1000,)
    assert (w<.9).all()
    sw = w.sum()
    print((((S[:,0]==0)*w).sum())/sw) 
    assert np.allclose((((S[:,0]==0)*w).sum())/sw, .4/.55, atol=0.05)
    assert np.allclose((S[:,1]==0).sum(), 1000, atol=0.1)
    assert np.allclose(((S[:,2]==0)*w).sum()/sw, .1, atol=0.1)
 
    ev = 1
    S=rejection_sampling(1000,PX1,PX2,PX3,ev)
    assert S.dtype==int
   
    print(((S[:,0]==0).sum())/1000)
    assert np.allclose((((S[:,0]==0)*w).sum())/sw, 2/9, atol=0.05)
    assert np.allclose(((S[:,1]==1)*w).sum()/sw, 1, atol=0.1)
    assert np.allclose(((S[:,2]==0)*w).sum()/sw, .6, atol=0.1)



#-------------------------------------------------------------------------
def test_sample_X1():
    '''(2 points) sample_X1'''
    PX1 = np.array([.5,.5])
    PX2 = np.array([[.8,.2],
                    [.3,.7]])
    PX3 = np.array([[.1,.9],
                    [.6,.4]])
    X2,X3 = 0,1
    c=0
    for _ in range(1000):
        X1=sample_X1(X2,X3,PX1,PX2,PX3)
        c+=X1
    print(c/1000)
    assert np.allclose(c/1000, 3/11, atol=0.05)

    X2,X3 = 1,0
    c=0
    for _ in range(1000):
        X1=sample_X1(X2,X3,PX1,PX2,PX3)
        c+=X1
    print(c/1000)
    assert np.allclose(c/1000, 7/9, atol=0.05) 

    X2,X3 = 0,0
    c=0
    for _ in range(1000):
        X1=sample_X1(X2,X3,PX1,PX2,PX3)
        c+=X1
    print(c/1000)
    assert np.allclose(c/1000, 3/11, atol=0.05)

    X2,X3 = 1,1
    c=0
    for _ in range(1000):
        X1=sample_X1(X2,X3,PX1,PX2,PX3)
        c+=X1
    print(c/1000)
    assert np.allclose(c/1000, 7/9, atol=0.05) 

#-------------------------------------------------------------------------
def test_sample_X2():
    '''(2 points) sample_X2'''
    PX1 = np.array([.5,.5])
    PX2 = np.array([[.8,.2],
                    [.3,.7]])
    PX3 = np.array([[.1,.9],
                    [.6,.4]])
    X1,X3 = 0,1
    c=0
    for _ in range(1000):
        X2=sample_X2(X1,X3,PX1,PX2,PX3)
        c+=X2
    assert np.allclose(c/1000, 0.1, atol=0.05)

    X1,X3 = 0,0
    c=0
    for _ in range(1000):
        X2=sample_X2(X1,X3,PX1,PX2,PX3)
        c+=X2
    assert np.allclose(c/1000, 0.6, atol=0.05)


    X1,X3 = 1,0
    c=0
    for _ in range(1000):
        X2=sample_X2(X1,X3,PX1,PX2,PX3)
        c+=X2
    assert np.allclose(c/1000, 0.9333, atol=0.05)

    X1,X3 = 1,1
    c=0
    for _ in range(1000):
        X2=sample_X2(X1,X3,PX1,PX2,PX3)
        c+=X2
    assert np.allclose(c/1000, 0.50909, atol=0.05)


#-------------------------------------------------------------------------
def test_gibbs_sampling():
    '''(2 points) gibbs_sampling'''
    PX1 = np.array([.5,.5])
    PX2 = np.array([[.8,.2],
                    [.3,.7]])
    PX3 = np.array([[.1,.9],
                    [.6,.4]])
    X1,X2,X3 = 0,0,1
    S = gibbs_sampling(1000,X1,X2,X3,PX1,PX2,PX3)
    assert S.dtype==int
    assert S.shape == (1000,3)
    assert np.allclose(S[:,2],np.ones(1000))

    idx = (S[:,0]==0)
    assert np.allclose(((S[idx,1]==0).sum())/(idx.sum()), .9, atol=0.08)
    idx = (S[:,0]==1)
    assert np.allclose(((S[idx,1]==0).sum())/(idx.sum()), .491, atol=0.08)

    X1,X2,X3 = 1,0,0
    S = gibbs_sampling(1000,X1,X2,X3,PX1,PX2,PX3)
    assert S.dtype==int
    assert S.shape == (1000,3)
    assert np.allclose(S[:,2],np.zeros(1000))

    idx = (S[:,0]==0)
    assert np.allclose(((S[idx,1]==0).sum())/(idx.sum()), .4, atol=0.08)
    idx = (S[:,0]==1)
    assert np.allclose(((S[idx,1]==0).sum())/(idx.sum()), 1/15, atol=0.05)


#-------------------------------------------------------------------------
def test_resample_z():
    '''(6 points) resample_z'''
    # 2 topics, 2 words in vocabulary, 2 documents, each document has 2 words
    # W = [[0,0],[1,1*]] here * indicate the current word
    # Z = [[0,0],[1,1*]]
    nz = np.array([2, 2])
    nzw = np.array([[2,0],
                    [0,2]])
    ndz = np.array([[2,0],
                    [0,2]])
    w = 1 # the word id = 2nd word in the vocabulary
    d = 1 # in the second document
    z = 1 # the first topic 

    z_new, p = resample_z(w,d,z,nz,nzw,ndz,0.,0.)
    assert np.allclose(p,[0,1])
    assert np.allclose(z_new,1)
    assert np.allclose(nz,[2, 2])
    assert np.allclose(nzw,[[2,0],[0,2]])
    assert np.allclose(ndz,[[2,0],[0,2]])

    # 2 topics, 2 words in vocabulary, 2 documents, each document has 4 words
    # W = [[0,0,0,1],[1,0,1,1*]] here * indicate the current word
    # Z = [[0,0,1,0],[0,1,1,1*]]
    c = 0
    for _ in range(100):
        nz = np.array([4, 4])
        nzw = np.array([[2,2],
                        [2,2]])
        ndz = np.array([[3,1],
                        [1,3]])
        w = 1 # the word id = 2nd word in the vocabulary
        d = 1 # in the second document
        z = 1 # the first topic 
        z_new, p = resample_z(w,d,z,nz,nzw,ndz,0.,0.)
        assert np.allclose(p,[0.43,0.57],atol=1e-2)
        if z_new==0:
            assert np.allclose(nz,[5, 3])
            assert np.allclose(nzw,[[2,3],[2,1]])
            assert np.allclose(ndz,[[3,1],[2,2]])
            c+=1
        else:
            assert np.allclose(nz,[4, 4])
            assert np.allclose(nzw,[[2,2],[2,2]])
            assert np.allclose(ndz,[[3,1],[1,3]])
    assert np.allclose(c/100,0.43,atol=0.1)



    # 2 topics, 2 words in vocabulary, 2 documents, each document has 2 words
    # W = [[0,0],[1,1*]] here * indicate the current word
    # Z = [[0,0],[1,1*]]
    nz = np.array([2, 2])
    nzw = np.array([[2,0],
                    [0,2]])
    ndz = np.array([[2,0],
                    [0,2]])
    w = 1 # the word id = 2nd word in the vocabulary
    d = 1 # in the second document
    z = 1 # the first topic 

    z_new, p = resample_z(w,d,z,nz,nzw,ndz,100.,100.)
    assert np.allclose(p,[.5,.5],atol=.1)

#-------------------------------------------------------------------------
def test_resample_Z():
    '''(2 points) resample_Z'''
    # 2 topics, 2 words in vocabulary, 2 documents, each document has 2 words
    # here we are adding 100 imagined samples to each topic
    c = 0
    cnz=np.zeros(2)
    cnzw=np.zeros((2,2))
    cndz=np.zeros((2,2))
    for _ in range(100):
        W = np.array([[1,1],[1,1]])
        Z = np.array([[1,1],[1,1]])
        nz = np.array([0, 4])
        nzw = np.array([[0,0],
                        [0,4]])
        ndz = np.array([[0,2],
                        [0,2]])
        
        Z = resample_Z(W,Z,nz,nzw,ndz,50.,50.)
        c+= (Z==0).sum() 
        cnz+=nz
        cndz+=ndz
        cnzw+=nzw
    assert np.allclose(c/400,0.5, atol =0.1)
    assert np.allclose(cnz/100,[2,2], atol =.5)
    assert np.allclose(cndz/100,[[1,1],[1,1]], atol =.5)
    assert np.allclose(cnzw/100,[[0,2],[0,2]], atol =.5)


    c = 0
    cnz=np.zeros(2)
    cnzw=np.zeros((2,2))
    cndz=np.zeros((2,2))
    for _ in range(100):
        W = np.array([[1,1],[1,1]])
        Z = np.array([[1,1],[1,1]])
        nz = np.array([0, 4])
        nzw = np.array([[0,0],
                        [0,4]])
        ndz = np.array([[0,2],
                        [0,2]])

        Z = resample_Z(W,Z,nz,nzw,ndz,1.,1.)
        c+= (Z==0).sum() 
        cnz+=nz
        cndz+=ndz
        cnzw+=nzw
    #print('c:',c/400)
    assert np.allclose(c/400,.27, atol =0.1)
    #print('cnz:',cnz/400)
    assert np.allclose(cnz/400,[0.27,0.73], atol =.2)
    #print('cndz:',cndz/200)
    assert np.allclose(cndz/200,[[0.3,0.6],[0.3,0.7]], atol =.2)
    print('cnzw:',cnzw/400)
    assert np.allclose(cnzw/400,[[0.,.3],[0.,.7]], atol =.2)
     

#-------------------------------------------------------------------------
def test_gibbs_sampling_LDA():
    '''(6 points) gibbs_sampling_LDA'''

    #          Word:  w0,w1,w2,w3,w4,w5
    beta = np.array([[.5,.4,.1,.0,.0,.0],  # topic 0: word distribution
                     [.0,.1,.4,.4,.1,.0],  # topic 1: word distribution
                     [.0,.0,.0,.1,.4,.5]]) # topic 2: word distribution

    m=3*5 # number of documents
    t = np.random.random(m)
    theta = np.ones((m,3))*.1 
    theta[0::3,0]+=0.7
    theta[1::3,1]+=0.7
    theta[2::3,2]+=0.7

    # sample words and documents
    n = 20  # number of words per document
    Z = np.zeros((m,n),dtype=int)
    W = np.zeros((m,n),dtype=int)
    for i in range(m):
        for j in range(n):
            z = np.random.choice(3,p=theta[i])  
            W[i,j] = np.random.choice(6,p=beta[z]) 
            # initialize Z with the correct topics
            Z[i,j]=z
     
    nz,nzw,ndz = gibbs_sampling_LDA(W,3,6,Z,alpha=1.,eta=1.,n_samples=10,n_burnin=0,sample_rate=1)
    assert np.allclose(nz.sum(),3000)
    assert np.allclose(nzw.sum(),3000)
    assert np.allclose(ndz.sum(),3000)

    assert np.allclose(nz/nz.sum(),np.ones(3)/3,atol=.1)
    assert np.allclose(nzw[0]/nz[0],beta[0],atol=.15) or np.allclose(nzw[0]/nz[0],beta[1],atol=.15) or np.allclose(nzw[0]/nz[0],beta[2],atol=.15)
    assert np.allclose(nzw[1]/nz[1],beta[0],atol=.15) or np.allclose(nzw[1]/nz[1],beta[1],atol=.15) or np.allclose(nzw[1]/nz[1],beta[2],atol=.15)
    assert np.allclose(nzw[2]/nz[2],beta[0],atol=.15) or np.allclose(nzw[2]/nz[2],beta[1],atol=.15) or np.allclose(nzw[2]/nz[2],beta[2],atol=.15)
   

    # randomly initialize Z
    Z = np.random.randint(3,size=(m,n))
    nz,nzw,ndz = gibbs_sampling_LDA(W,3,6,Z,alpha=1.,eta=1.,n_samples=20,n_burnin=10,sample_rate=2)
    assert np.allclose(nz.sum(),6000)
    assert np.allclose(nzw.sum(),6000)
    assert np.allclose(ndz.sum(),6000)

    assert np.allclose(nz/nz.sum(),np.ones(3)/3,atol=.1)
    assert np.allclose(nzw[0]/nz[0],beta[0],atol=.15) or np.allclose(nzw[0]/nz[0],beta[1],atol=.15) or np.allclose(nzw[0]/nz[0],beta[2],atol=.15)
    assert np.allclose(nzw[1]/nz[1],beta[0],atol=.15) or np.allclose(nzw[1]/nz[1],beta[1],atol=.15) or np.allclose(nzw[1]/nz[1],beta[2],atol=.15)
    assert np.allclose(nzw[2]/nz[2],beta[0],atol=.15) or np.allclose(nzw[2]/nz[2],beta[1],atol=.15) or np.allclose(nzw[2]/nz[2],beta[2],atol=.15)

 
#-------------------------------------------------------------------------
def test_compute_theta():
    '''(1 points) compute_theta'''

    ndz = np.array([[2,0],
                    [2,2]])
    theta = compute_theta(ndz,0.)
    assert np.allclose(theta,[[1,0],[0.5,0.5]],atol=.1)

    theta = compute_theta(ndz,1.)
    assert np.allclose(theta,[[.75,.25],[0.5,0.5]],atol=.1)



#-------------------------------------------------------------------------
def test_compute_beta():
    '''(1 points) compute_beta'''

    nzw = np.array([[2,0],
                    [2,2]])
    beta = compute_beta(nzw,0.)
    assert np.allclose(beta,[[1,0],[0.5,0.5]],atol=.1)

    beta = compute_beta(nzw,1.)
    assert np.allclose(beta,[[.75,.25],[0.5,0.5]],atol=.1)



#-------------------------------------------------------------------------
def test_LDA():
    '''(2 points) LDA'''
    #          Word:  w0,w1,w2,w3,w4,w5
    beta = np.array([[.5,.4,.1,.0,.0,.0],  # topic 0: word distribution
                     [.0,.1,.4,.4,.1,.0],  # topic 1: word distribution
                     [.0,.0,.0,.1,.4,.5]]) # topic 2: word distribution

    m=3*5 # number of documents
    t = np.random.random(m)
    theta = np.ones((m,3))*.1 
    theta[0::3,0]+=0.7
    theta[1::3,1]+=0.7
    theta[2::3,2]+=0.7

    # sample words and documents
    n = 20  # number of words per document
    W = np.zeros((m,n),dtype=int)
    for i in range(m):
        for j in range(n):
            z = np.random.choice(3,p=theta[i])  
            W[i,j] = np.random.choice(6,p=beta[z]) 
     
    # randomly initialize Z
    Z = np.random.randint(3,size=(m,n))
    beta_e,theta_e = LDA(W,3,6,alpha=1.,eta=1.,n_samples=20,n_burnin=10,sample_rate=2)
    a = 0.15 # atol
    assert np.allclose(beta_e[0],beta[0],atol=a) or np.allclose(beta_e[0],beta[1],atol=a) or np.allclose(beta_e[0],beta[2],atol=a)
    assert np.allclose(beta_e[1],beta[0],atol=a) or np.allclose(beta_e[1],beta[1],atol=a) or np.allclose(beta_e[1],beta[2],atol=a)
    assert np.allclose(beta_e[2],beta[0],atol=a) or np.allclose(beta_e[2],beta[1],atol=a) or np.allclose(beta_e[2],beta[2],atol=a)
    assert np.allclose(theta[:,0],theta[:,0],atol=a) or np.allclose(theta[:,0],theta[:,1],atol=a) or np.allclose(theta[:,0],theta[:,2],atol=a) 
    assert np.allclose(theta[:,1],theta[:,0],atol=a) or np.allclose(theta[:,1],theta[:,1],atol=a) or np.allclose(theta[:,1],theta[:,2],atol=a) 
    assert np.allclose(theta[:,2],theta[:,0],atol=a) or np.allclose(theta[:,2],theta[:,1],atol=a) or np.allclose(theta[:,2],theta[:,2],atol=a)


