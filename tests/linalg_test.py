#!/usr/bin/python3

import unittest
import numpy as np
import scipy.linalg
import polyopt

class TestLinalg(unittest.TestCase):
  """
  Unit test for linalg.py.

  by Pavel Trutman, pavel.trutman@fel.cvut.cz
  """


  def testQRSpecificMatrix(self):
    """
    Choosen matrix to dempose testing.
    """

    A = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
    Q, R, P = polyopt.linalg.qr(A)
    self.assertLessEqual(np.linalg.norm(np.dot(Q, R) - A[:, P]), 1e-3)


  def testQRSpecificMatrixWide(self):
    """
    Choosen matrix to dempose testing.
    """

    A = np.array([[12, -51, 4, 43], [6, 167, -68, -68]])
    Q, R, P = polyopt.linalg.qr(A)
    self.assertLessEqual(np.linalg.norm(np.dot(Q, R) - A[:, P]), 1e-3)


  def testQRRandomMatrices(self):
    """
    Random matrices with enlarging dimensions to dempose testing.
    """

    dims = [5, 10, 15, 25, 50, 100]
    for i in range(len(dims)):
      with self.subTest(i = i):
        A = np.random.rand(dims[i], dims[i])
        Q, R, P = polyopt.linalg.qr(A)
        self.assertLessEqual(np.linalg.norm(np.dot(Q, R) - A[:, P]), 1e-3)


  def testQRSpecificMatrixScipy(self):
    """
    Choosen matrix to dempose testing agains scipy QR decompistion.
    """

    A = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
    Q, R, P = polyopt.linalg.qr(A)
    Qs, Rs, Ps = scipy.linalg.qr(A, pivoting=True)
    self.assertLessEqual(np.linalg.norm(Q - Qs), 1e-3)
    self.assertLessEqual(np.linalg.norm(R - Rs), 1e-3)
    self.assertEqual(P, Ps.tolist())


  def testQRRandomMatricesScipy(self):
    """
    Random matrices with enlarging dimensions to dempose testing agains scipy QR decompistion.
    """

    dims = [5, 10, 15, 25, 50, 100]
    for i in range(len(dims)):
      with self.subTest(i = i):
        A = np.random.rand(dims[i], dims[i])
        Q, R, P = polyopt.linalg.qr(A)
        Qs, Rs, Ps = scipy.linalg.qr(A, pivoting=True)
        self.assertLessEqual(np.linalg.norm(Q - Qs), 1e-3)
        self.assertLessEqual(np.linalg.norm(R - Rs), 1e-3)
        self.assertEqual(P, Ps.tolist())


  def testQRSpecificMatrixUseLast(self):
    """
    Choosen matrix to dempose testing.
    """

    A = np.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
    Q, R, P = polyopt.linalg.qr(A, useLast=True)
    self.assertLessEqual(np.linalg.norm(np.dot(Q, R) - A[:, P]), 1e-3)
    self.assertEqual(P[0], 2)


  def testQRRandomMatricesUseLast(self):
    """
    Random matrices with enlarging dimensions to dempose testing.
    """

    dims = [5, 10, 15, 25, 50, 100]
    for i in range(len(dims)):
      with self.subTest(i = i):
        A = np.random.rand(dims[i], dims[i])
        Q, R, P = polyopt.linalg.qr(A, useLast=True)
        self.assertLessEqual(np.linalg.norm(np.dot(Q, R) - A[:, P]), 1e-3)
        self.assertEqual(P[0], dims[i] - 1)


if __name__ == '__main__':
  unittest.main()
