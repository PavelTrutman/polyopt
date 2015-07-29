#!/usr/bin/python3

"""
Demo, how to use the SDP Solver.

by Pavel Trutman, pavel.trutman@fel.cvut.cz
"""

from sympy import *
from SDPSolver import SDPSolver

# Problem statement
# min c0*x0 + c1*x1
# s. t. I_3 + A0*x0 + A1*x1 >= 0

c = Matrix([[1], [1]])
A0 = Matrix([[1,  0,  0],
             [0, -1,  0],
             [0,  0, -1]])
A1 = Matrix([[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]])

# starting point 
startPoint = Matrix([[0], [0]])

# create the solver object
problem = SDPSolver(c, A0, A1)

# enable graphs
problem.setDrawPlot(True)

# enable informative output
problem.setPrintOutput(True)

# solve!
x = problem.solve(startPoint)

print(x)
