#!/usr/bin/python3

"""
Demo, how to use the PS Solver.

by Pavel Trutman, pavel.trutman@cvut.cz
"""

from numpy import *
import polyopt

# the polynomials f1, f2, f3
# f1(x) = x2^4*x1 + 3*x1^3 - x2^4 - 3*x1^2
# f2(x) = x1^2*x2 -2*x1^2
# f3(x) = 2*x2^4*x1 - x1^3 - 2*x2^4 + x1^2
f1 = {(2, 0): -3, (3, 0): 3, (0, 4): -1, (1, 4): 1}
f2 = {(2, 0): -2, (2, 1): 1}
f3 = {(2, 0): 1, (3, 0): -1, (0, 4): -2, (1, 4): 2}

# ideal I
I = [f1, f2, f3]

# initialize the solver
PS = polyopt.PSSolver(I)

sol = PS.solve()
print()
print('Solution:')
print(sol)
