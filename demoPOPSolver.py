#!/usr/bin/python3

"""
Demo, how to use the POP Solver.

by Pavel Trutman, pavel.trutman@fel.cvut.cz
"""

from sympy import *
from POPSolver import POPSolver

# objective function
# f(x, y) = (x - 1)^2 + (y - 2)^2
#         = x^2 -2*x + y^2 - 4*y + 5
# global minima at (1, 2)
f = {(0, 0): 5, (1, 0): -2, (2, 0): 1, (0, 1): -4, (0, 2): 1}

# constraint function
# g(x, y) = 9 - x^2 - y^2
g = {(0, 0): 3**3, (2, 0): -1, (0, 2): -1}

# degree of the relaxation
d = 2

# initialize the solver
POP = POPSolver(f, g, d)

# obtain some feasible point for the SDP problem (within ball with radius 3)
y0 = POP.getFeasiblePoint(3)

# enable outputs
POP.setPrintOutput(True)

#solve the problem
x = POP.solve(y0)
print(x)

