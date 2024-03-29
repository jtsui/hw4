from scipy import *
import scipy.io
import numpy
import math
import preprocess
import random
import sys


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
    Calculates the gradient for a random training point
    '''
    i = random.randint(0, len(xtrain) - 1)
    return (ytrain[i] - mu(xtrain[i], beta)) * xtrain[i]


def update(xtrain, ytrain, beta, regularization_weight, learning_rate):
    '''
    updates and returns new beta
    alpha is learning rate
    lambda is regularization weight
    '''
    regularization_term = regularization_weight * numpy.linalg.norm(beta)
    update_term = gradient(xtrain, ytrain, beta) - regularization_term
    return beta + learning_rate * (update_term)


def batch(xtrain, ytrain, threshold, reg_weight, learning_rate):
    xplot, yplot = [], []
    beta = zeros(len(xtrain[0]))
    prev_l = 0
    l = 0
    iteration = 0
    while 1:
        beta = update(xtrain, ytrain, beta, reg_weight, learning_rate)
        prev_l = l
        l = nll(xtrain, ytrain, beta)
        print 'Iteration %s\tNLL %s\tDiff %s' % (iteration, l, abs(l - prev_l))
        if abs(l - prev_l) < threshold:
            print 'Done.'
            break
        xplot += [iteration]
        yplot += [l]
        iteration += 1
    return beta, xplot, yplot


def test_error(xtest, ytest, beta):
    error = 0.0
    for i in range(len(xtest)):
        p = mu(xtest[i], beta)
        if p > 0.5 and ytest[i] == 0:
            error += 1
        elif p < 0.5 and ytest[i] == 1:
            error += 1
    return error / len(ytest)


def shuffle(xtrain, ytrain):
    '''
    Shuffles xtrain and ytrain and returns the tuple
    '''
    shuffle_indices = numpy.random.permutation(range(3065))
    shuffled_xtrain = xtrain[shuffle_indices]
    shuffled_ytrain = ytrain[shuffle_indices]
    return shuffled_xtrain, shuffled_ytrain


def partition(xtrain, ytrain, i):
    '''
    Returns a tuple of the form (test_x, test_y, train_x, train_y)
    for the ith iteration of 5-fold validation
    '''
    start = 3065 / 5 * i
    end = 3065 / 5 * (i + 1)
    test_x = xtrain[start:end]
    test_y = ytrain[start:end]
    train_x = numpy.vstack((xtrain[:start], xtrain[end:]))
    train_y = numpy.vstack((ytrain[:start], ytrain[end:]))
    return (test_x, test_y, train_x, train_y)


def main():
    if len(sys.argv) != 2:
        print 'Missing args. Usage: python crossval.py [regularization weight]'
        return
    the_file, regularization_weight = sys.argv
    regularization_weight = float(regularization_weight)
    data = scipy.io.loadmat('spamData.mat')
    xtrain = preprocess.log_transform(data['Xtrain'])
    ytrain = data['ytrain']
    shuffled_xtrain, shuffled_ytrain = shuffle(xtrain, ytrain)

    threshold = 0.0001
    learning_rate = 0.0001
    print 'Regularization_weight %s learning_rate %s' % (regularization_weight, learning_rate)
    train = 0
    test = 0
    for i in range(5):
        xtest, ytest, xtrain, ytrain = partition(shuffled_xtrain, shuffled_ytrain, i)
        beta, xp, yp = batch(xtrain, ytrain, threshold, regularization_weight, learning_rate)
        train += test_error(xtrain, ytrain, beta)
        test += test_error(xtest, ytest, beta)
    train = train / 5
    test = test / 5
    print '%s\t%s\t%s\t%s' % (regularization_weight, learning_rate, train, test)
    with open('res%s.txt' % regularization_weight, 'a') as f:
        f.write('%s\t%s\t%s\t%s\n' % (regularization_weight, learning_rate, train, test))
        f.flush()
        # # plot xplot vs yplot
        # pyplot.plot(xplot, yplot)
        # pyplot.title('Training Loss vs Number of Iterations.\nregularization_weight %s learning_rate %s' % (
        #     regularization_weight, learning_rate))
        # pyplot.xlabel("Number of Iterations")
        # pyplot.ylabel("Negative Log Likelihood")
        # pyplot.show()


if __name__ == '__main__':
    main()
