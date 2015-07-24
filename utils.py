#!/usr/bin/python3

from sympy import *

class Utils:

  def LocalNorm(u, hessian):
    """
    Nesterov, p.181
    """
    return sqrt((hessian*u).transpose()*u)[0,0]

  def LocalNormA(u, hessian):
    """
    Nesterov, p.181
    """
    return sqrt((hessian.inv()*u).transpose()*u)[0,0]
