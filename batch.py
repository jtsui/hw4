from scipy import *
import scipy.io


def nll(xtrain, ytrain, beta):
    pass


def mu(vector, beta):
    pass


def update(xtrain, ytrain, beta, lambda, alpha):
    '''
    returns new beta
    alpha is learning rate
    lambda is regularization weight
    '''
    pass


def batch():
    beta = zeros(57)
    threshold = 
    prev_l = 0
    while 
        beta = update()
        prev_l = l
        l = nll(xtrain, ytrain, beta)
        if (l - prev_l) < threshold:
            break

