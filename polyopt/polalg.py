#!/usr/bin/python3

import numpy as np
import scipy

class Polalg:
  """
  Polynomial algebra utilities.

  by Pavel Trutman, pavel.trutman@fel.cvut.cz
  """


  def generateVariablesDegree(d, n):
    """
    Generates whole set of variables of given degree.

    Args:
      d (int): degree of the variables
      n (int): number of unknowns

    Returns:
      list: list of all variables
    """

    # generate zero degree variables
    if d == 0:
      return [(0,)*n]

    # generrate one degree variables
    elif d == 1:
      variables = []
      for i in range(0, n):
        t = [0]*n
        t[i] = 1
        variables.append(tuple(t))
      return variables

    # there is only one unkown with the degree d
    elif n == 1:
      return [(d,)]

    # generate variables in general case
    else:
      variables = []
      for i in range(0, d + 1):
        innerVariables = Polalg.generateVariablesDegree(d - i, n - 1)
        variables.extend([v + (i,) for v in innerVariables])
      return variables
  

  def generateVariablesUpDegree(d, n, reverse=False):
    """
    Generates whole set of variables up to given degree.

    Args:
      d (int): maximal degree of the variables
      n (int): number of unknowns

    Returns:
      list: list of variables
    """

    variables = []
    r = range(0, d + 1) if not reverse else reversed(range(0, d + 1))
    for i in r:
      variables.extend(Polalg.generateVariablesDegree(i, n))
    return variables


  def numVariablesDegree(d, n):
    """
    Computes number of monomials of degree d in n variables.

    Args:
      d (int): degree of monomials
      n (int): number of variables

    Returns:
      int: number of monomials
    """

    return scipy.special.comb(d + n - 1, n - 1, exact = True)


  def numVariablesUpDegree(d, n):
    """
    Computes number of monomials of degrees up to d in n variables.

    Args:
      d (int): maximal degree of monomials
      n (int): number of variables

    Returns:
      int: number of monomials
    """

    return scipy.special.comb(d + n, n, exact = True)
