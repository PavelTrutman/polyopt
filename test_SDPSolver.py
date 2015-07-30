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


  def testSpecificProblem(self):
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
        problem = SDPSolver(parameters[i]['c'], parameters[i]['A0'], parameters[i]['A1'])
        self.assertLessEqual(norm(problem.solve(parameters[i]['startPoint']) - parameters[i]['result']), 10**(-5))


if __name__ == '__main__':
  unittest.main()
