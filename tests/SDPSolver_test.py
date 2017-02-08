#!/usr/bin/python3

import unittest
from polyopt import SDPSolver
from numpy import *
from numpy.linalg import *
from polyopt.utils import Utils
from time import process_time

class TestSDPSolver(unittest.TestCase):
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
      'c': array([[1], [1]]),
      'A': [[identity(3),
            array([[1,  0,  0],
                    [0, -1,  0],
                    [0,  0, -1]]),
            array([[0,  1,  0],
                    [1,  0,  1],
                    [0,  1,  0]])]],
      'startPoint': array([[0], [0]]),
      'result': array([[-0.777673169427983], [-0.592418253409468]])
    }

    data1 = {
      'c': array([[1], [1]]),
      'A': [[identity(3),
            array([[ 1,  0, -1],
                    [ 0, -1,  0],
                    [-1,  0, -1]]),
            array([[ 0,  1,  0],
                    [ 1,  0,  1],
                    [ 0,  1,  0]])]],
      'startPoint': array([[0], [0]]),
      'result': array([[-0.541113957864176], [-0.833869642997048]])
    }

    data2 = {
      'c': array([[1], [1], [1]]),
      'A': [[identity(3),
            array([[1,  0,  0],
                    [0, -1,  0],
                    [0,  0, -1]]),
            array([[0,  1,  0],
                    [1,  0,  1],
                    [0,  1,  0]]),
            array([[0,  0,  0],
                    [0,  0, -1],
                    [0, -1,  0]])]],
      'startPoint': array([[0], [0], [0]]),
      'result': array([[-0.987675582117481], [-0.0243458354034874], [-1.98752220767823]])
    }

    # unbounded example manualy bounded
    data3 = {
      'c': array([[1], [1]]),
      'A': [[identity(3),
            array([[1, 1, 0],
                    [1, 1, 0],
                    [0, 0, 0]]),
            array([[1, 0, 1],
                    [0, 0, 1],
                    [1, 1, 1]])],
            [identity(3),
            array([[0, 1, 0],
                    [1, 0, 0],
                    [0, 0, 0]]),
            array([[0, 0, 1],
                    [0, 0, 0],
                    [1, 0, 0]])],
           ],
      'startPoint': array([[0], [0]]),
      'result': array([[-0.367456763013021], [-0.228720901608749]])
    }
    parameters = [data0, data1, data2, data3]

    # test all cases
    for i in range(0, len(parameters)):
      with self.subTest(i = i):
        # init prolem
        problem = SDPSolver(parameters[i]['c'], parameters[i]['A'])

        # solve and compare the results
        self.assertLessEqual(norm(problem.solve(parameters[i]['startPoint'], problem.auxFollow) - parameters[i]['result']), 10**(-5))

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
      'c': array([[1], [1]]),
      'A': [identity(3), 
            array([[1, 1, 0],
                    [1, 1, 0],
                    [0, 0, 0]]),
            array([[1, 0, 1],
                    [0, 0, 1],
                    [1, 1, 1]])],
      'startPoint': array([[0], [0]])
    }

    data1 = {
      'c': array([[1], [1], [1]]),
      'A': [identity(3), 
            array([[1,  0,  0],
                    [0, -1,  0],
                    [0,  0, -1]]),
            array([[0,  1,  0],
                    [1,  0,  1],
                    [0,  1,  0]]),
            array([[1,  1,  0],
                    [1,  0,  1],
                    [0,  1,  1]])],
      'startPoint': array([[0], [0], [0]])
    }
    parameters = [data0, data1]

    # test all cases
    for i in range(0, len(parameters)):
      with self.subTest(i = i):
        problem = SDPSolver(parameters[i]['c'], [parameters[i]['A']])
        with self.assertRaises(LinAlgError):
          problem.solve(parameters[i]['startPoint'], problem.auxFollow)


  def _testRandomProblemBoundedSetAuxFollow(self):
    """
    Test some random generated problems, which should be always bounded.

    Test one problem for dimensions specified below.
    """
    
    
    # specify dimensions
    dims = [1, 2, 3, 4, 5, 6, 7, 10, 25]
    
    # test all of them
    for n in dims:
      with self.subTest(i = n):
        # starting point
        startPoint = zeros((n, 1));

        # objective function
        c = ones((n, 1))

        # get LMI matrices
        A = [eye(n)];
        for i in range(0, n):
          A.append(Utils.randomSymetric(n))

        # init SDP program
        problem = SDPSolver(c, [A])

        # bound the problem
        problem.bound(1)

        # solve
        timeBefore = process_time();
        problem.solve(startPoint, problem.auxFollow)
        elapsedTime = process_time() - timeBefore
        #print(elapsedTime)

        # the matrix have to be semidefinite positive (eigenvalues >= 0)
        eigs = problem.eigenvalues()
        for eig in eigs:
          self.assertGreaterEqual(eig, 0)

        # the smallest eigenvalue has to be near zero
        self.assertLessEqual(eigs[0], 10**(-3))


  def testRandomProblemBoundedSetDampedNewton(self):
    """
    Test some random generated problems, which should be always bounded.

    Test one problem for dimensions specified below.
    """
    
    
    # specify dimensions
    dims = [1, 2, 3, 4, 5, 6, 7, 10, 25]
    
    # test all of them
    for n in dims:
      with self.subTest(i = n):
        # starting point
        startPoint = zeros((n, 1));

        # objective function
        c = ones((n, 1))

        # get LMI matrices
        A = [identity(n)];
        for i in range(0, n):
          A.append(Utils.randomSymetric(n))

        # init SDP program
        problem = SDPSolver(c, [A])

        # bound the problem
        problem.bound(1)

        # solve
        timeBefore = process_time();
        problem.solve(startPoint, problem.dampedNewton)
        elapsedTime = process_time() - timeBefore
        #print(elapsedTime)

        # the matrix have to be semidefinite positive (eigenvalues >= 0)
        eigs = problem.eigenvalues()
        for eig in eigs:
          self.assertGreaterEqual(eig, 0)

        # the smallest eigenvalue has to be near zero
        self.assertLessEqual(eigs[0], 10**(-3))


if __name__ == '__main__':
  unittest.main()
