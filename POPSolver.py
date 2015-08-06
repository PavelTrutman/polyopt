#!/usr/bin/python3

from sympy import *
from utils import Utils
from SDPSolver import SDPSolver
from math import ceil
from scipy.misc import comb
from numpy.random import uniform
from numpy.linalg import norm
from numpy.linalg import eig
import numpy as np
import logging

class POPSolver:
  """
  Class providing POP (Polynomial Optimization Problem) Solver.

  by Pavel Trutman, pavel.tutman@fel.cvut.cz
  """


  def __init__(self, f, g, d):
    """
    Initialization of the POP problem.

    Args:
      f (dictionary: tuple => int): representation of the objective function f(x)
      g (dictionary: tuple => int): representation of the constraining function g(x)
      d (int): degree of the relaxation
    """

    # get number of variables
    key = list(f.keys())[0]
    self.n = len(key)
    self.d = d

    # generate all variables up to degree 2*d
    allVar = self.generateVariablesUpDegree(2*self.d)

    # collect all variables used
    varUsed = allVar

    # generate moment matrix and localizing matrix
    self.MM = self.momentMatrix(self.d, varUsed)
    self.LM = self.localizingMatrix(self.d - 1, varUsed, g)

    # generate objective function for SDP
    self.c = zeros(len(varUsed) - 1, 1)
    for variable in range(1, len(varUsed)):
      self.c[variable - 1, 0] = f.get(varUsed[variable], 0)


  def solve(self, startPoint):
    print(self.c)
    print(self.MM)
    print(self.LM)
    SDP = SDPSolver(self.c, [self.MM, self.LM])
    SDP.setDrawPlot(True)
    SDP.setPrintOutput(True)
    x = SDP.solve(startPoint)


  def momentMatrix(self, d, varUsed):
    """
    Constructs moment matrix.

    Args:
      d (int): degree of the relaxation
      varUsed (list of tuples): all variables that are used

    Returns:
      list: list of moment matrices
    """

    varUpD = self.generateVariablesUpDegree(d)
    varUsedNum = len(varUsed)
    dimM = len(varUpD)

    MM = [zeros(dimM) for i in range(0, varUsedNum)]

    for i in range(0, dimM):
      for j in range(i, dimM):
        # sum up the degrees
        varCur = tuple(sum(t) for t in zip(varUpD[i], varUpD[j]))
        # find this variable amongs used vars
        index = [k for k in range(0, varUsedNum) if varUsed[k] == varCur]
        if len(index) > 0:
          pos = index[0]
          MM[pos][i, j] = 1
          MM[pos][j, i] = 1
    return MM


  def localizingMatrix(self, d, varUsed, g):
    """
    Constructs localizing matrix.

    Args:
      d (int): degree of the relaxation
      varUsed (list of tuples): all variables that are used
      g (dictionary: tuple => int): representation of the constraining function g(x)

    Returns:
      list: list of localizing matrices
    """

    varUpD = self.generateVariablesUpDegree(d)
    varUsedNum = len(varUsed)
    dimM = len(varUpD)
    LM = [zeros(dimM) for i in range(0, varUsedNum)]

    for mon, coef in g.items():
      for i in range(0, dimM):
        for j in range(i, dimM):
          # sum up the degrees
          varCur = tuple(sum(t) for t in zip(varUpD[i], varUpD[j], mon))
          # find this variable amongs used vars
          index = [k for k in range(0, varUsedNum) if varUsed[k] == varCur]
          if len(index) > 0:
            pos = index[0]
            LM[pos][i, j] += coef
            if i != j:
              LM[pos][j, i] += coef
    return LM


  def generateVariablesDegree(self, d, n):
    if d == 0:
      return [(0,)*n]
    elif d == 1:
      variables = []
      for i in range(0, n):
        t = [0]*n
        t[i] = 1
        variables.append(tuple(t))
      return variables
    elif n == 1:
      return [(d,)]
    else:
      variables = []
      for i in range(0, d + 1):
        innerVariables = self.generateVariablesDegree(d - i, n - 1)
        variables.extend([v + (i,) for v in innerVariables])
      return variables
  

  def generateVariablesUpDegree(self, d):
    variables = []
    for i in range(0, d + 1):
      variables.extend(self.generateVariablesDegree(i, self.n))
    return variables


  def getFeasiblePoint(self, R):

    N = comb(self.n + self.d, self.n)
    N = ceil(N*1.5 + 1)

    # generate all variable
    usedVars = self.generateVariablesUpDegree(2*self.d)[1:]

    y = zeros(len(usedVars), 1)
    for alpha in range(0, len(usedVars)):
      i = 0
      s = 0
      while i < N:
        x = uniform(-R, R, (self.n, 1))
        if norm(np.array(x)) < R**2:
          yTemp = 1
          for j in range(0, self.n):
            yTemp *= x[j, 0]**usedVars[alpha][j]
          s += yTemp
          i += 1
      y[alpha, 0] = s/N

    return y
