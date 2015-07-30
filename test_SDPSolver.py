#!/usr/bin/python3

import unittest
from SDPSolver import SDPSolver
from sympy import *
from sympy.mpmath import *

class SDPSolverTest(unittest.TestCase):
  """
  Unit test for SDPSolver.py.

  by Pavel Trutman, pavel.trutman@fel.cvut.cz
  """


  def testSpecificProblemBoundedSet(self):
    """
    Some specific problems have been choosen to be tested.
    """


    # prepare set of testing cases
    data0 = {
      'c': Matrix([[1], [1]]),
      'A0': Matrix([[1,  0,  0],
                    [0, -1,  0],
                    [0,  0, -1]]),
      'A1': Matrix([[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]]),
      'startPoint': Matrix([[0], [0]]),
      'result': Matrix([[-0.777673169427983], [-0.592418253409468]])
    }

    data1 = {
      'c': Matrix([[1], [1]]),
      'A0': Matrix([[1,  0,  -1],
                    [0, -1,  0],
                    [-1,  0, -1]]),
      'A1': Matrix([[0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0]]),
      'startPoint': Matrix([[0], [0]]),
      'result': Matrix([[-0.541113957864176], [-0.833869642997048]])
    }
    parameters = [data0, data1]

    # test all cases
    for i in range(0, len(parameters)):
      with self.subTest(i = i):
        # init prolem
        problem = SDPSolver(parameters[i]['c'], parameters[i]['A0'], parameters[i]['A1'])

        # solve and compare the results
        self.assertLessEqual(norm(problem.solve(parameters[i]['startPoint']) - parameters[i]['result']), 10**(-5))

        # the matrix have to be semidefinite positive (eigenvalues >= 0)
        eigs = problem.eigenvalues()
        for eig in eigs:
          self.assertGreaterEqual(eig, 0)

        # the smallest eigenvalue has to be near zero
        self.assertLessEqual(eigs[0], 10**(-3))



  def testSpecificProblemUnboundedSet(self):
    """
    Some specific problems have been choosen to be tested. The set is unbounded, so it fails in the auxiliary path-follow part.
    """


    # prepare set of testing cases
    data0 = {
      'c': Matrix([[1], [1]]),
      'A0': Matrix([[1, 1, 0],
                    [1, 1, 0],
                    [0, 0, 0]]),
      'A1': Matrix([[1, 0, 1],
                    [0, 0, 1],
                    [1, 1, 1]]),
      'startPoint': Matrix([[0], [0]])
    }
    parameters = [data0]

    # test all cases
    for i in range(0, len(parameters)):
      with self.subTest(i = i):
        problem = SDPSolver(parameters[i]['c'], parameters[i]['A0'], parameters[i]['A1'])
        with self.assertRaises(ValueError):
          problem.solve(parameters[i]['startPoint'])


if __name__ == '__main__':
  unittest.main()
