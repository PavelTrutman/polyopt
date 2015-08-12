#!/usr/bin/python3

"""
Demo, how to use the SDP Solver.

by Pavel Trutman, pavel.trutman@fel.cvut.cz
"""

from numpy import *
from SDPSolver import SDPSolver

# Problem statement
# min c0*x0 + c1*x1
# s. t. I_3 + A0*x0 + A1*x1 >= 0

c = array([[1], [1]])
A0 = array([[1,  0,  0],
             [0, -1,  0],
             [0,  0, -1]])
A1 = array([[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]])
# starting point 
startPoint = array([[0], [0]])

# create the solver object
problem = SDPSolver(c, [[eye(3), A0, A1]])

# enable graphs
problem.setDrawPlot(True)

# enable informative output
problem.setPrintOutput(True)

# enable bounding into ball with radius 1
#problem.bound(1)

# solve!
x = problem.solve(startPoint, problem.dampedNewton)
print(x)

# print eigenvalues
#print(problem.eigenvalues())
