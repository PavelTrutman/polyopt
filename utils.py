#!/usr/bin/python3

from sympy import *

class Utils:

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

  def gradientHessian(A0, A1, x0, x1):
    """
    Compute gradient and hessian analyticly.
    """

    A = eye(3) + A0*x0 + A1*x1
    Ainv = A.inv()
    Ainv0 = Ainv*A0
    Ainv1 = Ainv*A1
    Fd = Matrix([[-trace(Ainv0)], [-trace(Ainv1)]])
    Fdd = Matrix([[trace(Ainv0**2), trace(Ainv0*Ainv1)], [trace(Ainv0*Ainv1), trace(Ainv1**2)]])
    return Fd, Fdd
