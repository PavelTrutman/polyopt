#!/usr/bin/python3

import unittest
from numpy import *
from numpy.linalg import *
from polyopt import PSSolver

class TestPSSolver(unittest.TestCase):
  """
  Unit test for PSSolver.py.

  by Pavel Trutman, pavel.trutman@cvut.cz
  """


  def testSpecificProblemDimOne(self):
    """
    A specific problem of dimension one has been choosen to be tested.
    """

    # prepare set of testing cases
    f1 = {(2, ): 1, (1, ): -1, (0, ): -6}
    solution = [array([[-2]]), array([[3]])]

    # init prolem
    problem = PSSolver([f1])

    # solve and compare the results
    x = problem.solve()
    for s in solution:
      self.assertTrue((norm(x - s, axis=1) < 1e-3).any())


  def testSpecificProblemDimTwo(self):
    """
    A specific problem of dimension two has been choosen to be tested.
    """

    # prepare set of testing cases
    f1 = {(0, 0): 48, (1, 0): -16, (2, 0): -20, (1, 1): 1, (0, 1): -1, (0, 2): -12}
    f2 = {(0, 0): 44, (1, 0): 46, (2, 0): 12, (1, 1): -58, (0, 1): -47, (0, 2): 3}
    solution = [array([[1, 1]]), array([[-2, 0]]), array([[-0.5, 2]]), array([[-1, -2]])]

    # init prolem
    problem = PSSolver([f1, f2])

    # solve and compare the results
    good = True
    x = problem.solve()
    for s in solution:
      self.assertTrue((norm(x - s, axis=1) < 1e-3).any())


if __name__ == '__main__':
  unittest.main()
