#!/usr/bin/python3

from utils import Utils
from SDPSolver import SDPSolver
from math import ceil
from scipy.misc import comb
from numpy.random import uniform
from numpy.linalg import *
from numpy import *
import logging
import sys

class POPSolver:
  """
  Class providing POP (Polynomial Optimization Problem) Solver.

  Solves problem in this form:
    min f(x)
    s.t. g(x) >= 0

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

    # disable output
    logging.basicConfig(stream = sys.stdout, format = '%(message)s')
    self.logStdout = logging.getLogger()

    # generate all variables up to degree 2*d
    allVar = self.generateVariablesUpDegree(2*self.d)

    # collect all variables used
    varUsed = allVar

    # generate moment matrix and localizing matrix
    self.MM = self.momentMatrix(self.d, varUsed)
    self.LM = self.localizingMatrix(self.d - 1, varUsed, g)

    # generate objective function for SDP
    self.c = zeros((len(varUsed) - 1, 1))
    for variable in range(1, len(varUsed)):
      self.c[variable - 1, 0] = f.get(varUsed[variable], 0)

    # initialize SDP Solver
    self.SDP = SDPSolver(self.c, [self.MM, self.LM])


  def solve(self, startPoint):
    """
    Solves a POP problem.

    Args:
      startPoint (Matrix): some feasible point of the SDP problem

    Returns:
      Matrix: solution of the problem
    """

    # solve the SDP porblem
    y = self.SDP.solve(startPoint, self.SDP.dampedNewton)

    # extract solutions of the POP problem
    x = y[0:self.n, :]
    return x


  def setPrintOutput(self, printOutput):
    """
    Enables or disables printing of the computation state.

    Args:
      printOuput (bool): True - enables the output, False - disables the output

    Returns:
      None
    """

    self.SDP.setPrintOutput(printOutput)
    if printOutput:
      self.logStdout.setLevel(logging.INFO)
    else:
      self.logStdout.setLevel(logging.WARNING)


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

    MM = [zeros((dimM, dimM)) for i in range(0, varUsedNum)]

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
    LM = [zeros((dimM, dimM)) for i in range(0, varUsedNum)]

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
        innerVariables = self.generateVariablesDegree(d - i, n - 1)
        variables.extend([v + (i,) for v in innerVariables])
      return variables
  

  def generateVariablesUpDegree(self, d):
    """
    Generates whole set of variables up to given degree.

    Args:
      d (int): maximal degree of the variables

    Returns:
      list: list of variables
    """

    variables = []
    for i in range(0, d + 1):
      variables.extend(self.generateVariablesDegree(i, self.n))
    return variables


  def getFeasiblePoint(self, R):
    """Finds feasible point for SDP problem arisen from the POP problem.

    Args:
      R (int): a radius of a ball from which x points are choosen

    Returns:
      Matrix: feasible point for the SDP problem
    """

    N = comb(self.n + self.d, self.n)
    N = ceil(N*1.5 + 1)

    # generate all variable
    usedVars = self.generateVariablesUpDegree(2*self.d)[1:]

    y = zeros((len(usedVars), 1))
    i = 0

    # choose many points x to moment matrix have full rank
    while i < N:

      # select x from the ball with given radius
      x = uniform(-R, R, (self.n, 1))
      if norm(x) < R**2:
        
        # generate points y from it
        for alpha in range(0, len(usedVars)):
          yTemp = 1
          for j in range(0, self.n):
            yTemp *= x[j, 0]**usedVars[alpha][j]
          y[alpha, 0] += yTemp
        i += 1
    y = y / N

    return y
