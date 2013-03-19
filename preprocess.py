import numpy
import scipy
import math

def standardize(matrix):
  def standardize_column(col):
    mean = numpy.mean(col)
    std = numpy.std(col)
    col = col - mean
    col = col / std
    return col
  return numpy.apply_along_axis(standardize_column, 0, matrix)

def log_transform(matrix):
  return numpy.log(matrix + 0.1)

def binarize(matrix):
  matrix2 = matrix
  matrix2[matrix > 0] = 1
  return matrix2


