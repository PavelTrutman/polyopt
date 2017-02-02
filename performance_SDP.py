#!/usr/bin/python3

"""
Preformance measuring for SDP solvers, saves results into file.

by Pavel Trutman, pavel.trutman@fel.cvut.cz
"""

from numpy import *
from SDPSolver import SDPSolver
import timeit
from utils import Utils
import time
import subprocess


# dimension: 1, ..., 10
dim = [1, 2, 3, 5, 7, 10, 13, 16, 20, 25]

# number of trials
N = range(0, 20)

# initialize the array with result times
results = empty([max(dim) + 1, 1])
results.fill(nan)

# for dimension
for n in dim:

  # clear measured time
  t = 0

  # for different feasible points
  for i in N:

    # starting point
    startPoint = zeros((n, 1));

    # objective function
    c = ones((n, 1))

    # get LMI matrices
    A = [identity(n)];
    for i in range(0, n):
      A.append(Utils.randomSymetric(n))

    # init SDP program
    problem = SDPSolver(c, [A])

    # bound the problem
    problem.bound(1)

    # startup time
    timeStart = timeit.default_timer()

    #solve the problem
    problem.solve(startPoint, problem.dampedNewton)

    # final time
    t += timeit.default_timer() - timeStart

  print('dimension:', n)
  results[n] = t/len(N)

date = time.strftime('%Y%m%d_%H%M')
gitVersion = subprocess.check_output(['git', 'describe', '--always'])
filename = 'performanceResults/SDP_' + date + '_' + gitVersion.decode('utf-8').strip() + '.npy'
save(filename, results)
print(results)
print(filename)
