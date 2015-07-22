#!/usr/bin/python3

from numpy import *

class Utils:

  def LocalNorm(u, hessian):
    """
    Nesterov, p.181
    """
    return float(sqrt((hessian*u).T*u))

  def LocalNormA(u, hessian):
    """
    Nesterov, p.181
    """
    return float(sqrt((hessian.I*u).T*u))
