#!/usr/bin/python3

from sympy import *
from SDPSolver import SDPSolver

c = Matrix([[1], [1]])
A0 = Matrix([[1,  0,  0],
             [0, -1,  0],
             [0,  0, -1]])
A1 = Matrix([[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]])
startPoint = Matrix([[0], [0]])


problem = SDPSolver(c, A0, A1)
problem.setDrawPlot(True)
problem.setPrintOutput(True)
x = problem.solve(startPoint)
print(x)
