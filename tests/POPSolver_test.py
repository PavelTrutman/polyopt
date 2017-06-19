#!/usr/bin/python3

import unittest
from numpy import *
from numpy.linalg import *
from polyopt import POPSolver
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
    g = [{(0,): 3**2, (1,): 0, (2,): -1}]
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
    g = [{(0, 0): 3**2, (0, 2): -1, (2, 0): -1}]
    solution = matrix([[1], [2]])

    # test all cases
    for degree in range(1, 3):
      with self.subTest(i = degree):
        # init prolem
        problem = POPSolver(f, g, degree)

        # solve and compare the results
        x = problem.solve(problem.getFeasiblePoint(3))
        self.assertLessEqual(norm(x - solution), 10**(-3))


  def testSpecificProblemDimTwoMoreCnstrns(self):
    """
    A specific problem of dimension two and two constraints has been choosen to be tested.
    """

    # prepare set of testing cases
    f = {(1, 0): -1, (0, 1): -3/2}
    g = [{(2, 0): -20, (1, 1): 1, (0, 2): -12, (1, 0): -16, (0, 1): -1, (0, 0): 48}, {(2, 0): 12, (1, 1): -58, (0, 2): 3, (1, 0): 46, (0, 1): -47, (0, 0): 44}]
    solution = matrix([[0.25], [1.5]])

    # test all cases
    for degree in range(2, 4):
      with self.subTest(i = degree):
        # init prolem
        problem = POPSolver(f, g, degree)

        # solve and compare the results
        x = problem.solve(problem.getFeasiblePoint(1))
        self.assertLessEqual(norm(x - solution), 10**(-3))


if __name__ == '__main__':
  unittest.main()
