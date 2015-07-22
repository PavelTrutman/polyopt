#!/usr/bin/python3

from numpy import *

class Utils:

  def LocalNorm(u, hessian):
    return float(sqrt((hessian*u).T*u))
