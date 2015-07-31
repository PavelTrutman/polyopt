#!/usr/bin/python3

from sympy import *
from utils import Utils
import pyGnuplot as gp
import logging
import sys

# some constants
beta = 1/9
gamma = 5/36


class SDPSolver:
  """
  Class providing SDP Solver.

  Solves problem in a form:
    min sum_i(c_i*x_i)
    s.t. A0 + sum_i(A_i*x_i) >= 0
  where A_i are symetric matrices.

  by Pavel Trutman, pavel.tutman@fel.cvut.cz
  """


  def __init__(self, c, AAll):
    """
    Initialization of the problem:
      min sum_i(c_i*x_i)
      s.t. A0 + sum_i(A_i*x_i) >= 0

    Args:
      c (Matrix)
      AAll (list of Matrix)

    Returns:
      None
    """
    
    self.solved = False

    # initialization of the problem
    self.c = c
    self.dim = c.rows
    self.AAll = AAll
    self.nu = AAll[0].rows

    # disable plotting
    self.drawPlot = False

    # disable output
    logging.basicConfig(stream = sys.stdout, format = '%(message)s')
    self.logStdout = logging.getLogger()


  def setDrawPlot(self, drawPlot):
    """
    Enables or disables the drawing of the graph.

    Args:
      drawPlot (bool): True - draw graph, False - do not draw graph
    
    Returns:
      None

    Throws:
      ValueError: the graph con not be plotted if the dimension is not 2
    """

    if drawPlot == True & self.dim != 2:
      raise ValueError('The graph can not be plotted. Dimension of the problem differ from 2.')

    self.drawPlot = drawPlot

    if self.drawPlot:
      # symbolic variables
      x = Symbol('x')
      y = Symbol('y')

      # self-concordant barrier
      X = self.AAll[0] + self.AAll[1]*x + self.AAll[2]*y

      # plot and save the set into file
      self.gnuplot = gp.gnuplot()
      self.gnuplot.set('view map')
      self.gnuplot('set contour')
      self.gnuplot('set cntrparam levels discrete 0')
      self.gnuplot('unset surface')
      self.gnuplot('set isosamples 2000, 2000')
      self.gnuplot('set table "setPlot.dat"')
      setPlot = self.gnuplot.splot(str(X.det()))
      self.gnuplot.newXterm()
      self.gnuplot.show(setPlot)
      self.gnuplot('unset table')
      self.gnuplot('reset')

      # plot the set
      self.plot = self.gnuplot.plot('"setPlot.dat" using 1:2', w = 'lines', title = 'Set boundary')
      self.gnuplot.show(self.plot)


  def setPrintOutput(self, printOutput):
    """
    Enables or disables printing of the computation state.

    Args:
      printOuput (bool): True - enables the output, False - disables the output

    Returns:
      None
    """

    if printOutput:
      self.logStdout.setLevel(logging.INFO)
    else:
      self.logStdout.setLevel(logging.WARNING)


  def solve(self, start):
    """
    Solve the problem from the starting point.

    Args:
      start (Matrix): the starting point of the algorithm

    Returns:
      Matrix: found optimal solution
    """

    x0 = self.auxFollow(start)
    return self.mainFollow(x0)


  def auxFollow(self, start):
    """
    Auxiliary path-following algorithm [Nesterov, p. 205]

    Args:
      start (Matrix): the starting point of the algorithm

    Returns:
      Matrix: approximation of the analytic center
    """

    t = 1
    k = 0

    # starting point
    y = start
    if self.drawPlot:
      y0All = [y[0, 0]]
      y1All = [y[1, 0]]

    # gradient and hessian
    Fd, Fdd, _ = Utils.gradientHessian(self.AAll, y)
    Fd0 = Fd

    self.logStdout.info('AUXILIARY PATH-FOLLOWING')

    while True:
      k += 1
      self.logStdout.info('\nk = ' + str(k))

      # iteration step
      t = t - gamma/Utils.LocalNormA(Fd0, Fdd)
      y = y - Fdd.inv()*(t*Fd0 + Fd)
      #self.logStdout.info('t = ' + str(t))
      self.logStdout.info('y = ' + str(y))

      if self.drawPlot:
        y0All.append(y[0, 0])
        y1All.append(y[1, 0])

      # gradient and hessian
      Fd, Fdd, A = Utils.gradientHessian(self.AAll, y)

      # print eigenvalues
      if self.logStdout.isEnabledFor(logging.INFO):
        eigs = list(A.eigenvals())
        eigs = [ re(N(eig)) for eig in eigs ]
        eigs.sort()
        self.logStdout.info('EIG = ' + str(eigs))

      # breaking condition
      if Utils.LocalNormA(Fd, Fdd) <= sqrt(beta)/(1 + sqrt(beta)):
        break

    # prepare x
    x = y - Fdd.inv()*Fd

    # plot auxiliary path
    if self.drawPlot:
      self.plot.add(y1All, xvals = y0All, title = 'Auxiliary path', w = 'points', pt = 1)
      self.gnuplot.show(self.plot)
    return x


  def mainFollow(self, x):
    """
    Main following algorithm [Nesterov, p. 202]

    Args:
      x (Matrix): good approximation of the analytic center, used as the starting point of the algorithm

    Returns:
      Matrix: found optimal solution of the problem
    """

    if self.drawPlot:
      x0All = [x[0, 0]]
      x1All = [x[1, 0]]

    # Main path-following scheme [Nesterov, p. 202]
    self.logStdout.info('\nMAIN PATH-FOLLOWING')

    # initialization of the iteration process
    t = 0
    eps = 10**(-3)
    k = 0

    # print the input condition to verify that is satisfied
    Fd, Fdd, _ = Utils.gradientHessian(self.AAll, x)
    self.logStdout.info('Input condition = ' + str(Utils.LocalNormA(Fd, Fdd)))

    while True:
      k += 1
      self.logStdout.info('\nk = ' + str(k))

      # gradient and hessian
      Fd, Fdd, A = Utils.gradientHessian(self.AAll, x)

      # iteration step
      t = t + gamma/Utils.LocalNormA(self.c, Fdd)
      x = x - Fdd.inv()*(t*self.c+Fd)

      if self.drawPlot:
        x0All.append(x[0, 0])
        x1All.append(x[1, 0])

      self.logStdout.info('t = ' + str(t))
      self.logStdout.info('x = ' + str(x))

      if self.logStdout.isEnabledFor(logging.INFO):
        # print eigenvalues
        eigs = list(A.eigenvals())
        eigs = [ re(N(eig)) for eig in eigs ]
        eigs.sort()
        self.logStdout.info('EIG = ' + str(eigs))

      # breaking condition
      self.logStdout.info('Breaking condition = ' + str(eps*t))
      if eps*t >= self.nu + (beta + sqrt(self.nu))*beta/(1 - beta):
        break

    self.solved = True
    self.result = x
    self.resultA = A

    # plot main path
    if self.drawPlot:
      self.plot.add(x1All, xvals = x0All, title = 'Main path', w = 'points', pt = 1)
      self.gnuplot.show(self.plot)
      print('\nPress enter to continue')
      input()

    return x


  def eigenvalues(self):
    """
    Returns eigenvalues of the barrier matrix at the optimal point.

    Returns:
      list: eigenvalues

    Throws:
      ValueError: when the problem has not been solved yet
    """

    if self.solved:
      eigs = list(self.resultA.eigenvals())
      eigs = [ re(N(eig)) for eig in eigs ]
      eigs.sort()
      return eigs
    else:
      raise ValueError('The problem has not been solved yet so the eignevalues can not be evaluated.')
    


