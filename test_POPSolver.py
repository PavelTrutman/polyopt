#!/usr/bin/python3

import unittest
from POPSolver import POPSolver
from sympy import *
from sympy.mpmath import norm
from utils import Utils
from time import process_time

class TestPOPSolver(unittest.TestCase):
  """
  Unit test for POPSolver.py.

  by Pavel Trutman, pavel.trutman@fel.cvut.cz
  """


  def testSpecificProblemDimOne(self):
    """
    A specific problem of dimension one has been choosen to be tested.
    """

    # prepare set of testing cases
    f = {(2, ): 1, (1, ): 1, (0, ): -1}
    g = {(0,): 9, (1,): 0, (2,): -1}

    # test all cases
    for degree in range(1, 4):
      with self.subTest(i = degree):
        # init prolem
        problem = POPSolver(f, g, degree)

        # solve and compare the results
        problem.solve(problem.getFeasiblePoint(3))


if __name__ == '__main__':
  unittest.main()
