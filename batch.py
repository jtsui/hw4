from scipy import *
import scipy.io
import numpy
import math
import preprocess
from matplotlib import pyplot


def nll(xtrain, ytrain, beta):
    '''
    return negative log likelihood of training set given beta
    '''
    l = 0
    for i in range(len(xtrain)):
        x_i = xtrain[i]
        y_i = ytrain[i]
        l += -(y_i * ln_mu(x_i, beta) + (1 - y_i) * ln_one_minus_mu(x_i, beta))
    return l


def ln_mu(vector, beta):
    try:
        return -math.log(1 + math.exp(-beta.dot(vector)))
    except OverflowError:
        import pdb
        pdb.set_trace()


def ln_one_minus_mu(vector, beta):
    return -beta.dot(vector) - math.log(1 + math.exp(-beta.dot(vector)))


def mu(vector, beta):
    '''
    returns mu(x) = P(spam | x; beta)
    = 1 / (1 + math.exp(-1.0 * beta.dot(vector)))
    '''
    return 1.0 / (1.0 + math.exp(-beta.dot(vector)))


def gradient(xtrain, ytrain, beta):
    '''
    Calculates the gradient over entire training set
    '''
    grad = numpy.zeros(len(xtrain[0]))
    for i in range(len(xtrain)):
        term = (ytrain[i] - mu(xtrain[i], beta)) * xtrain[i]
        grad += term
    return grad


def update(xtrain, ytrain, beta, regularization_weight, learning_rate):
    '''
    updates and returns new beta
    alpha is learning rate
    lambda is regularization weight
    '''
    regularization_term = regularization_weight * numpy.linalg.norm(beta)
    update_term = gradient(xtrain, ytrain, beta) - regularization_term
    return beta + learning_rate * (update_term)


def batch(xtrain, ytrain, threshold, reg_weight, learning_rate, xplot, yplot):
    beta = zeros(len(xtrain[0]))
    prev_l = 0
    l = 0
    iteration = 0
    while 1:
        beta = update(xtrain, ytrain, beta, reg_weight, learning_rate)
        prev_l = l
        l = nll(xtrain, ytrain, beta)
        print 'Iteration %s\tNLL %s\tDiff %s' % (iteration, l, l - prev_l)
        if abs(l - prev_l) < threshold:
            print 'Done.'
            break
        xplot += [iteration]
        yplot += [l]
        iteration += 1
    return beta


def test_error(xtest, ytest, beta):
    error = 0.0
    for i in range(len(xtest)):
        p = mu(xtest[i], beta)
        if p > 0.5 and ytest[i] == 0:
            error += 1
        elif p < 0.5 and ytest[i] == 1:
            error += 1
    return error / len(ytest)


def main():
    data = scipy.io.loadmat('spamData.mat')
    xtrain = preprocess.log_transform(data['Xtrain'])
    # xtrain = data['Xtrain']
    ytrain = data['ytrain']
    xtest = preprocess.log_transform(data['Xtest'])
    ytest = data['ytest']
    threshold = 0.1
    reg_learn_pairs = [(0.1, 0.0001), (0.001, 0.001), (0.001, 0.0001), (0.0001, 0.001), (0.0001, 0.0001)]
    for regularization_weight, learning_rate in reg_learn_pairs:
        print 'Regularization_weight %s learning_rate %s' % (regularization_weight, learning_rate)
        xplot = []
        yplot = []
        beta = batch(xtrain, ytrain, threshold, regularization_weight, learning_rate, xplot, yplot)
        train = test_error(xtrain, ytrain, beta)
        test = test_error(xtest, ytest, beta)
        with open('res.txt', 'a') as f:
            f.write('%s\t%s\t%s\t%s\n' % (regularization_weight, learning_rate, train, test))
            f.flush()
        # plot xplot vs yplot
        pyplot.plot(xplot, yplot)
        pyplot.title('Training Loss vs Number of Iterations.\nregularization_weight %s learning_rate %s' % (
            regularization_weight, learning_rate))
        pyplot.xlabel("Number of Iterations")
        pyplot.ylabel("Negative Log Likelihood")
        pyplot.show()


if __name__ == '__main__':
    main()
