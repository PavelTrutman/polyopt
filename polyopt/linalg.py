#!/usr/bin/python3

import numpy as np

class Linalg:
  """
  Linear algebra utilities.

  by Pavel Trutman, pavel.trutman@fel.cvut.cz
  """


  def qr(A, useLast=False):
    """
    Rank revealing QR decomposition.

    Args:
      A (array): matrix to be decomposed
      useLast (bool): whether use last column in the first iteration

    Returns:
      array: matrix Q
      array: matrix R
      list: pivots
    """

    A = np.matrix.copy(A)
    m, n = A.shape
    Q = np.eye(m)
    p = list(range(n))
    if min(m, n) <= 1 and useLast:
      return np.eye(1), A, [p[-1]] + p[:-1]
    for i, _ in zip(range(n - 1), range(m - 1)):
      H = np.eye(m)

      # find the most nondependent column
      if (i == 0) and useLast:
        kMax = n - 1
      else:
        vMax = 0
        kMax = i
        for k in range(i, n):
          v = np.linalg.norm(A[i:, k])
          if v > vMax:
            vMax = v
            kMax = k
      A[:, [i, kMax]] = A[:, [kMax, i]]
      p[i], p[kMax] = p[kMax], p[i]

      H[i:, i:] = Linalg.make_householder(A[i:, i])
      Q = np.dot(Q, H)
      A = np.dot(H, A)
    return Q, A, p


  def make_householder(a):
    """
    Helper function for QR decomposition.
    """

    v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    v[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :])
    return H
