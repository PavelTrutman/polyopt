#!/usr/bin/python3

import pycallgraph
from numpy import *
from numpy.linalg import *
from polyopt import SDPSolver
from polyopt.utils import Utils


def filtercalls(call_stack, modul, clas, func, full):
    mod_ignore = ['shutil','scipy.optimize','re','os','sys','json']
    func_ignore = ['CustomFunctionName','pdbcall']
    clas_ignore = ['pdb']
    return modul not in mod_ignore and func not in func_ignore and clas not in clas_ignore


# specify dimensions
dim = 37

# starting point
startPoint = zeros((dim, 1));

# objective function
c = ones((dim, 1))

# get LMI matrices
A = [identity(dim)];
for i in range(0, dim):
  A.append(Utils.randomSymetric(dim))

# init SDP program
problem = SDPSolver(c, [A])

# bound the problem
problem.bound(1)

# pycallgraph config
config = pycallgraph.Config()
config.include_stdlib = True
graphviz = pycallgraph.output.GraphvizOutput(output_file='profile.png')

# profile
with pycallgraph.PyCallGraph(output=graphviz, config=config):
  # solve
  problem.solve(startPoint, problem.dampedNewton)
