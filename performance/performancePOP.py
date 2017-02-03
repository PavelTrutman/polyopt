#!/usr/bin/python3

"""
Preformance measuring and saves results into file.

by Pavel Trutman, pavel.trutman@fel.cvut.cz
"""

from numpy import *
from numpy.random import uniform
from POPSolver import POPSolver
import timeit


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
      innerVariables = generateVariablesDegree(d - i, n - 1)
      variables.extend([v + (i,) for v in innerVariables])
    return variables


def generateVariablesUpDegree(d, n):
  """
  Generates whole set of variables up to given degree.

  Args:
    d (int): maximal degree of the variables

  Returns:
    list: list of variables
  """

  variables = []
  for i in range(0, d + 1):
    variables.extend(generateVariablesDegree(i, n))
  return variables



# dimension: 1, 2, 3
dim = range(1, 4)

# degree of generated polynomials: 2, 3, 4
degF = range(2, 5)

# relaxation orders: ceil(degF/2), ceil(degF/2) + 1, ceil(degF/2) + 2, ceil(degF/2) + 3
order = range(0, 4)

# number of feasible point generations: 10
N = range(0, 10)

# initialize the array with result times
results = empty((max(dim) + 1, max(degF) + 1, len(order)))
results.fill(nan)

# for dimension
for n in dim:

  #for degree of polynomials
  for dF in degF:
    variables = generateVariablesUpDegree(dF, n)

    # generate objective function
    f = {}
    for v in variables:
      f[v] = uniform(-1, 1)

    #generate constraining function
    t = [0]*n
    g = {tuple(t): 1}
    for i in range(0, n):
      t = [0]*n
      t[i] = 2
      g[tuple(t)] = -1

    minD = int(ceil(dF/2))

    # for relaxation order
    for dR in order:
      d = dR + minD

      # clear measured time
      t = 0

      # for different feasible points
      for i in N:
        # startup time
        timeStart = timeit.default_timer()

        # initialize the solver
        problem = POPSolver(f, g, d)

        # obtain some feasible point for the SDP problem
        y0 = problem.getFeasiblePoint(1)

        #solve the problem
        problem.solve(y0)

        # final time
        t += timeit.default_timer() - timeStart

      print((n, dF, dR))
      results[n, dF, dR] = t/len(N)

save('performanceResults/performanceResults.npy', results)
print(results)
