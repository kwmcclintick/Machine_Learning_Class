from problem1 import SoftmaxRegression as sr
import torch as th
import torch.nn as nn
from torch.optim import SGD
import math
import numpy as np

#-------------------------------------------------------------------------
'''
    Problem 2: Convolutional Neural Network 
    In this problem, you will implement a convolutional neural network with a convolution layer and a max pooling layer.
    The goal of this problem is to learn the details of convolutional neural network. 
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
    Note: please do NOT use th.nn.functional.conv2d or th.nn.Conv2D, implement your own version of 2d convolution using only basic tensor operations.
'''

#--------------------------
def conv2d(x,W,b):
    '''
        Compute the 2D convolution with one filter on one image, (assuming stride=1).
        Input:
            x:  one training instance, a float torch Tensor of shape l by h by w. 
                h and w are height and width of an image. l is the number channels of the image, for example RGB color images have 3 channels.
            W: the weight matrix of a convolutional filter, a torch tensor of shape l by s by s. 
            b: the bias vector of the convolutional filter, a torch scalar tensor. 
        Output:
            z: the linear logit tensor after convolution, a float torch tensor of shape (h-s+1) by (w-s+1)
        Note: please do NOT use th.nn.functional.conv2d, implement your own version of 2d convolution,using basic tensor operation, such as matrix multiplication. 
    '''
    #########################################
    h = x.size(1)
    w = x.size(2)
    s = W.size(1)
    z = th.empty(h-s+1, w-s+1)
    for i in range(h-s+1):
        for j in range(w-s+1):
            # sum first across l using sum(,0), then convolve using sum()
            z[i, j] = th.sum(th.sum(x[:, i:i+s, j:j+s]*W[:], 0)) + b
    #########################################
    return z


#--------------------------
def Conv2D(x,W,b):
    '''
        Compute the 2D convolution with multiple filters on a batch of images, (assuming stride=1).
        Input:
            x:  a batch of training instances, a float torch Tensor of shape (n by l by h by w). n is the number instances in a batch.
                h and w are height and width of an image. l is the number channels of the image, for example RGB color images have 3 channels.
            W: the weight matrix of a convolutional filter, a torch tensor of shape (n_filters by l by s by s). 
            b: the bias vector of the convolutional filter, a torch vector tensor of length n_filters. 
        Output:
            z: the linear logit tensor after convolution, a float torch tensor of shape (n by n_filters by (h-s+1) by (w-s+1) )
        Note: please do NOT use th.nn.functional.conv2d, implement your own version of 2d convolution.
        Hint: you could re-use conv2d() function to build this function.
    '''
    #########################################
    n = x.size(0)
    n_filters = W.size(0)
    h = x.size(2)
    w = x.size(3)
    s = W.size(2)
    z = th.empty(n, n_filters, h-s+1, w-s+1)
    for i in range(n):
        for j in range(n_filters):
            z[i,j,:,:] = conv2d(x[i], W[j], b[j])
    #########################################
    return z 


#--------------------------
def ReLU(z):
    '''
        Compute ReLU activation. 
        Input:
            z: the linear logit tensor after convolution, a float torch tensor of shape (n by n_filters by h by w )
                h and w are the height and width of the image after convolution. 
        Output:
            a: the nonlinear activation tensor, a float torch tensor of shape (n by n_filters by h by w )
        Note: please do NOT use th.nn.functional.relu, implement your own version using only basic tensor operations. 
    '''
    #########################################
    a = th.clamp(z, min=0, max=float("Inf"))
    #########################################
    return a 


#--------------------------
def avgpooling(a):
    '''
        Compute the 2D average pooling (assuming shape of the pooling window is 2 by 2).
        Input:
            a:  the feature map of one instance, a float torch Tensor of shape (n by n_filter by h by w). n is the batch size, n_filter is the number of filters in Conv2D.
                h and w are height and width after ReLU. 
        Output:
            p: the tensor after pooling, a float torch tensor of shape n by n_filter by floor(h/2) by floor(w/2).
        Note: please do NOT use torch.nn.AvgPool2d or torch.nn.functional.avg_pool2d or avg_pool1d, implement your own version using only basic tensor operations.
    '''
    #########################################
    n = a.size(0)
    n_filter = a.size(1)
    h = a.size(2)
    w = a.size(3)

    p = th.empty(n, n_filter, math.floor(h/2), math.floor(w/2))
    ps = 2 # pool size
    for k in range(n):
        for l in range(n_filter):
            for i in range(math.floor(h/2)):
                for j in range(math.floor(w/2)):
                    p[k,l,i,j] = th.mean(a[k,l,ps*i:ps*i+ps,ps*j:ps*j+ps])
    #########################################
    return p 

#--------------------------
def maxpooling(a):
    '''
        Compute the 2D max pooling (assuming shape of the pooling window is 2 by 2).
        Input:
            a:  the feature map of one instance, a float torch Tensor of shape (n by n_filter by h by w). n is the batch size, n_filter is the number of filters in Conv2D.
                h and w are height and width after ReLU. 
        Output:
            p: the tensor after max pooling, a float torch tensor of shape n by n_filter by floor(h/2) by floor(w/2).
        Note: please do NOT use torch.nn.MaxPool2d or torch.nn.functional.max_pool2d or max_pool1d, implement your own version using only basic tensor operations.
        Note: if there are mulitple max values, select the one with the smallest index.
    '''
    #########################################
    n = a.size(0)
    n_filter = a.size(1)
    h = a.size(2)
    w = a.size(3)
    p = th.empty(n, n_filter, math.floor(h / 2), math.floor(w / 2))
    ps = 2  # pool size
    for k in range(n):
        for l in range(n_filter):
            for i in range(math.floor(h / 2)):
                for j in range(math.floor(w / 2)):
                    # th.max doesnt work with backpass so I made my own max function
                    window = a[k, l, ps * i:ps * i + ps, ps * j:ps * j + ps]
                    # init max to always be smaller than the first value
                    biggest = -np.infty
                    # window is always a 2D tensor of some shape
                    for row in window:
                        for value in row:
                            if value > biggest:
                                biggest = value
                    # finally we can set maxpooled value to largest val in the window
                    p[k, l, i, j] = biggest
    #########################################
    return p 


#--------------------------
def num_flat_features(h=28, w=28, s=3, n_filters=10):
    ''' Compute the number of flat features after convolution and pooling. Here we assume the stride of convolution is 1, the size of pooling kernel is 2 by 2, no padding. 
        Inputs:
            h: the hight of the input image, an integer scalar
            w: the width of the input image, an integer scalar
            s: the size of convolutional filter, an integer scalar. For example, a 3 by 3 filter has a size 3.
            n_filters: the number of convolutional filters, an integer scalar
        Outputs:
            p: the number of features we will have on each instance after convolution, pooling, and flattening, an integer scalar.
    '''
    #########################################
    ps = 2 # pool size
    new_height = (h-s+1)/ps
    new_width = (w-s+1)/ps
    p = int(new_height*new_width*n_filters)
    #########################################
    return p
 

#-------------------------------------------------------
class CNN(sr):
    '''CNN is a convolutional neural network with a convolution layer (with ReLU activation), a max pooling layer and a fully connected layer.
       In the convolutional layer, we will use ReLU as the activation function. 
       After the convolutional layer, we apply a 2 by 2 max pooling layer, before feeding into the fully connected layer.
    '''
    # ----------------------------------------------
    def __init__(self, l=1, h=28, w=28, s=5, n_filters=5, c=10):
        ''' Initialize the model. Create parameters of convolutional layer and fully connected layer. 
            Inputs:
                l: the number of channels in the input image, an integer scalar
                h: the hight of the input image, an integer scalar
                w: the width of the input image, an integer scalar
                s: the size of convolutional filter, an integer scalar. For example, a 3 by 3 filter has a size 3.
                n_filters: the number of convolutional filters, an integer scalar
                c: the number of output classes, an integer scalar
            Outputs:
                self.conv_W: the weight matrix of the convolutional filters, a torch tensor of shape n_filters by l by s by s, initialized as all-zeros. 
                self.conv_b: the bias vector of the convolutional filters, a torch vector tensor of length n_filters, initialized as all-ones, to avoid vanishing gradient.
                self.W: the weight matrix parameter in fully connected layer, a torch tensor of shape (p, c), initialized as all-zeros. 
                        Hint: CNN is a subclass of SoftmaxRegression, which already has a W parameter. p is the number of flat features after pooling layer.
                self.b: the bias vector parameter, a torch tensor of shape (c), initialized as all-zeros
                self.loss_fn: the loss function object for softmax regression. 
            Note: In this problem, the parameters are initialized as either all-zeros or all-ones for testing purpose only. In real-world cases, we usually initialize them with random values.
        '''
        #########################################
        # compute the number of flat features
        p = num_flat_features(h=h, w=w, s=s, n_filters=n_filters)
        # initialize fully connected layer 
        self.conv_W = th.zeros((n_filters, l, s, s), requires_grad=True)
        self.conv_b = th.ones(n_filters, requires_grad=True)
        sr.__init__(self, p=p, c=c)
        # the kernel matrix of convolutional layer
        #########################################


    # ----------------------------------------------
    def forward(self, x):
        '''
           Given a batch of training instances, compute the linear logits of the outputs. 
            Input:
                x:  a batch of training instance, a float torch Tensor of shape n by l by h by w. Here n is the batch size. l is the number of channels. h and w are height and width of an image. 
            Output:
                z: the logit values of the batch of training instances after the fully connected layer, a float matrix of shape n by c. Here c is the number of classes
        '''
        #########################################
        # convolutional layer
        conv = Conv2D(x,self.conv_W, self.conv_b)
        # ReLU activation 
        a = ReLU(conv)
        # maxpooling layer
        p = maxpooling(a)
        # flatten 
        flat = th.flatten(p, start_dim=1)
        # fully connected layer
        z = sr.forward(self, flat)
        #########################################
        return z

    # ----------------------------------------------
    def train(self, loader, n_steps=10,alpha=0.01):
        """train the model 
              Input:
                loader: dataset loader, which loads one batch of dataset at a time.
                n_steps: the number of batches of data to train, an integer scalar
                alpha: the learning rate for SGD(stochastic gradient descent), a float scalar
        """
        # create a SGD optimizer
        optimizer = SGD([self.conv_W,self.conv_b,self.W,self.b], lr=alpha)
        count = 0
        acc_train = []
        acc_test = []
        while True:
            # use loader to load one batch of training data
            for x,y in loader:
                #########################################
                # forward pass
                z = self.forward(x)
                # compute loss 
                Loss = self.compute_L(z, y)
                # backward pass: compute gradients
                Loss.backward()
                # update model parameters
                optimizer.step()
                # reset the gradients of W and b to zero
                optimizer.zero_grad()
                #########################################
                count+=1
                if count >=n_steps:
                    return acc_train, acc_test

