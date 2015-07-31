#!/usr/bin/python3

from sympy import *

class Utils:
  """
  Some usefull math utilities.

  by Pavel Trutman, pavel.trutman@fel.cvut.cz
  """


  def LocalNorm(u, hessian):
    """
    Nesterov, p.181
    """

    return N(sqrt((hessian*u).transpose()*u)[0,0])


  def LocalNormA(u, hessian):
    """
    Nesterov, p.181
    """

    return N(sqrt((hessian.inv()*u).transpose()*u)[0,0])


  def gradientHessian(AAll, x):
    """
    Compute gradient and hessian analytically.
    """
    
    dim = len(x)
    A = AAll[0]
    for i in range(1, len(AAll)):
      A += AAll[i]*x[i - 1, 0]
    Ainv = A.inv()
    AAllinv = []
    for i in range(0, dim):
      AAllinv.append(Ainv*AAll[i + 1])

    # gradient
    Fd = zeros(dim, 1)
    for i in range(0, dim):
      Fd[i, 0] = -trace(AAllinv[i])

    # hessian
    Fdd = zeros(dim, dim)
    for i in range(0, dim):
      for j in range(i, dim):
        Aij = trace(AAllinv[i]*AAllinv[j])
        Fdd[i, j] = Aij
        Fdd[j, i] = Aij

    return Fd, Fdd, A
