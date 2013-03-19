Our implementation is in Python and requires scipy, numpy, and matplotlib.


Problem 1
Run batch.py. Tunable parameters are in main() and include the type of preprocessing, regularization weights, and learning parameters. This will write the training and test error rates to a file res.txt and display the graphs.


Problem 2
Run stochastic.py. Tunable parameters are in main() similar to problem 1.


Problem 3
Run stochastic_variable_alpha.py. Tunable parameters are in main() similar to problem 1.


Problem 4.
Run crossval.py. Tunable parameters include threshold, learning rate, and a list of regularization weights to try. For each value in regularization weight, will run 5 fold cross validation and output the average error.