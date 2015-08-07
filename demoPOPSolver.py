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

#f = {(2, ): 1, (1, ): 1, (0, ): -1}

# g(x, y) = 3^3 - x^2 - y^2
g = {(0, 0): 3**3, (0, 2): -1, (2, 0): -1}
#g = {(0,): 9, (1,): 0, (2,): -1}

# degree of the relaxation
d = 2

POP = POPSolver(f, g, d)
y0 = POP.getFeasiblePoint(3)
POP.setPrintOutput(True)
y = POP.solve(y0)
print(y)

