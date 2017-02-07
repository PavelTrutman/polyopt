#!/usr/bin/python3

import pycallgraph
from numpy import *
from numpy.linalg import *
from polyopt import SDPSolver
import pickle
import re

def parseDot(regexp, content):
  m = regexp.search(content)
  return float(m.group(2))/float(m.group(1).replace(' ', '')) # removing &nbsp;


# number of iterations
N = 5

# function for which get the timings
functionsToMeasure = ['polyopt.utils.gradientHessian', 'numpy.core.fromnumeric.trace']
regexps = list(map((lambda x: re.compile(r".*" + x.replace('.', r"\.") + r".*calls:\s*([0-9 ]*).n.*time:\s*([0-9.]*)s")), functionsToMeasure))

# specify dimensions
dim = 50

# starting point
startPoint = zeros((dim, 1))

# objective function
c = ones((dim, 1))

# get LMI matrices
with open('matrices.pickle', 'rb') as f:
  A = pickle.load(f)

times = empty((len(functionsToMeasure), N))
for i in range(0, N):
  # init SDP program
  problem = SDPSolver(c, [A])
  
  # bound the problem
  problem.bound(1)
  
  # pycallgraph config
  config = pycallgraph.Config()
  config.include_stdlib = True
  dot = pycallgraph.output.GraphvizOutput(output_file='profile.dot', output_type='dot')
  graphviz = pycallgraph.output.GraphvizOutput(output_file='profile.png')
  
  # profile
  with pycallgraph.PyCallGraph(output=[graphviz, dot], config=config) as graph:
    # solve
    problem.solve(startPoint, problem.dampedNewton)

  with open('profile.dot', 'r') as f:
    content = f.read()
  
  times[:, i] = array(list(map((lambda r: parseDot(r, content)), regexps)))
  print('.', end='', flush=True)

print()
times = nanmedian(times, axis=1)
for i in range(0, len(functionsToMeasure)):
  print('{:>30s}: {:4.3g} ms'.format(functionsToMeasure[i], times[i]*1000))
