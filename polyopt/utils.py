#!/usr/bin/python3

from numpy import *
from numpy.linalg import *
from numpy.random import uniform
from mpmath import norm

class Utils:
  """
  Some usefull math utilities.

  by Pavel Trutman, pavel.trutman@fel.cvut.cz
  """


  def LocalNorm(u, hessian):
    """
    Nesterov, p.181
    """

    return sqrt(dot((dot(hessian, u)).T, u))[0,0]


  def LocalNormA(u, hessian):
    """
    Nesterov, p.181
    """

    return sqrt(dot((solve(hessian, u)).T, u))[0,0]


  def gradientHessian(AAll, x):
    """
    Compute gradient and hessian analytically.
    """
    
    # get the dimension of the problem
    dim = len(x)

    Fd = zeros((dim, 1))
    Fdd = zeros((dim, dim))
    A = [None]*len(AAll)

    for a in range(0, len(AAll)):

      A[a] = copy(AAll[a][0])
      for i in range(1, len(AAll[a])):
        A[a] += AAll[a][i]*x[i - 1, 0]

      Ainv = inv(A[a])
      AAllinv = [None]*dim
      for i in range(0, dim):
        AAllinv[i] = dot(Ainv, AAll[a][i + 1])

      # gradient
      for i in range(0, dim):
        Fd[i, 0] -= einsum('ii->', AAllinv[i])

      # hessian
      for i in range(0, dim):
        Fdd[i, i] += einsum('ij,ji->', AAllinv[i], AAllinv[i])/2
        for j in range(i + 1, dim):
          Fdd[i, j] += einsum('ij,ji->', AAllinv[i], AAllinv[j])

    Fdd = Fdd + Fdd.T

    return Fd, Fdd, A


  def randomSymetric(dim):
    """
    Generate random symetric matrix with uniform distribution.

    Args:
      dim (int): dimension of the resulting matrix

    Returns:
      Matrix: random symetric matrix
    """


    M = zeros((dim, dim))
    for i in range(0, dim):
      for j in range(i, dim):
        r = uniform(-1, 1)
        M[i, j] = r
        M[j, i] = r
    return M
