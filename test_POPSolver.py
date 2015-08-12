#!/usr/bin/python3

import unittest
from numpy import *
from numpy.linalg import *
from POPSolver import POPSolver
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
    g = {(0,): 3**3, (1,): 0, (2,): -1}
    solution = matrix([[-0.5]])

    # test all cases
    for degree in range(1, 4):
      with self.subTest(i = degree):
        # init prolem
        problem = POPSolver(f, g, degree)

        # solve and compare the results
        x = problem.solve(problem.getFeasiblePoint(3))
        self.assertLessEqual(norm(x - solution), 10**(-3))


  def testSpecificProblemDimTwo(self):
    """
    A specific problem of dimension two has been choosen to be tested.
    """

    # prepare set of testing cases
    f = {(0, 0): 5, (1, 0): -2, (2, 0): 1, (0, 1): -4, (0, 2): 1}
    g = {(0, 0): 3**3, (0, 2): -1, (2, 0): -1}
    solution = matrix([[1], [2]])

    # test all cases
    for degree in range(1, 3):
      with self.subTest(i = degree):
        # init prolem
        problem = POPSolver(f, g, degree)

        # solve and compare the results
        x = problem.solve(problem.getFeasiblePoint(3))
        self.assertLessEqual(norm(x - solution), 10**(-3))


if __name__ == '__main__':
  unittest.main()
