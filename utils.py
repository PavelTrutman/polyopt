#!/usr/bin/python3

from sympy import *
from numpy.random import uniform
from sympy.mpmath import norm

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


  def gradientHessian(AAll, x, R = None):
    """
    Compute gradient and hessian analytically.
    """
    
    # get the dimension of the problem
    dim = len(x)

    A = AAll[0]  
    for i in range(1, len(AAll)):
      A += AAll[i]*x[i - 1, 0]

    if R != None:
      dist = R**2 - norm(x)**2

    Ainv = A.inv()
    AAllinv = []
    for i in range(0, dim):
      AAllinv.append(Ainv*AAll[i + 1])

    # gradient
    Fd = zeros(dim, 1)
    for i in range(0, dim):
      Fd[i, 0] = -trace(AAllinv[i])
    if R != None:
      Fd += 2*x/dist

    # hessian
    Fdd = zeros(dim, dim)
    for i in range(0, dim):
      for j in range(i, dim):
        Aij = trace(AAllinv[i]*AAllinv[j])
        if R != None:
          if i == j:
            Aij += 2*(dist + 2*x[i, 0]**2)/(dist**2)
          else:
            Aij += (4*x[i, 0]*x[j, 0])/(dist**2)
        Fdd[i, j] = Aij
        Fdd[j, i] = Aij

    if R != None:
      appendix = eye(dim + 1)
      appendix[0, 0] = R**2
      appendix[1:dim + 1, 0] = x
      appendix[0, 1:dim + 1] = x.T
      A = Utils.concatenateDiagonal(A, appendix)

    return Fd, Fdd, A


  def concatenateDiagonal(A, B):
    """
    Concatenate two matrix along the diagonal.

        | A  0 |
    M = |      |
        | 0  B |

    Args:
      A (Matrix): first matrix to cocatenate
      B (Matrix): second matrix to concatenate
    
    Returns:
      Matrix: result of the concatenation
    """

    return A.row_join(zeros(A.rows, B.cols)).col_join(zeros(B.rows, A.cols).row_join(B))


  def randomSymetric(dim):
    """
    Generate random symetric matrix with uniform distribution.

    Args:
      dim (int): dimension of the resulting matrix

    Returns:
      Matrix: random symetric matrix
    """


    M = zeros(dim)
    for i in range(0, dim):
      for j in range(i, dim):
        r = uniform(-1, 1)
        M[i, j] = r
        M[j, i] = r
    return M
