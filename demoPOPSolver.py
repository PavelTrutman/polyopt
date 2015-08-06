#!/usr/bin/python3

"""
Demo, how to use the POP Solver.

by Pavel Trutman, pavel.trutman@fel.cvut.cz
"""

from sympy import *
from POPSolver import POPSolver


# f(x, y) = (x - 1)^2 + (y - 2)^2
#         = x^2 -2*x + y^2 - 4*y + 5
# global minima at (1, 2)
f = {(0, 0): 5, (0, 1): -2, (0, 2): 1, (1, 0): -4, (2, 0): 1}

# g(x, y) = 1 - x^2 - y^2
g = {(0, 0): 1, (0, 2): -1, (2, 0): -1}

# degree of the relaxation
d = 2

POP = POPSolver(f, g, d)
y0 = POP.getFeasiblePoint(1)
print(y0)
POP.solve(y0)

