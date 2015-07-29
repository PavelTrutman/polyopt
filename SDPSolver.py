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
    min c0*x0 + c1*x1
    s.t. I_3 + A0*x0 + A1*x1 >= 0
  where A0 and A1 are symetric matrices.

  by Pavel Trutman, pavel.tutman@fel.cvut.cz
  """


  def __init__(self, c, A0, A1):
    """
    Initialization of the problem:
      min c0*x0 + c1*x1
      s.t. I_3 + A0*x0 + A1*x1 >= 0

    Args:
      c (Matrix)
      A0 (Matrix)
      A1 (Matrix)

    Returns:
      None
    """
    
    # initialization of the problem
    self.c = c
    self.A0 = A0
    self.A1 = A1
    self.nu = 3

    # symbolic variables
    self.x0 = Symbol('x0')
    self.x1 = Symbol('x1')

    # self-concordant barrier
    self.X = eye(3) + self.A0*self.x0 + self.A1*self.x1
    F = -log(self.X.det())

    # first symbolic derivation
    Fdx0 = diff(F, self.x0)
    Fdx1 = diff(F, self.x1)
    self.Fd = Matrix([[Fdx0], [Fdx1]])

    # symbolic hessian
    Fddx0x0 = diff(Fdx0, self.x0)
    Fddx1x1 = diff(Fdx1, self.x1)
    Fddx0x1 = diff(Fdx0, self.x1)
    self.Fdd = Matrix([[Fddx0x0, Fddx0x1], [Fddx0x1, Fddx1x1]])

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
    """

    self.drawPlot = drawPlot

    if self.drawPlot:
      # plot and save the set into file
      self.gnuplot = gp.gnuplot()
      self.gnuplot.set('view map')
      self.gnuplot('set contour')
      self.gnuplot('set cntrparam levels discrete 0')
      self.gnuplot('unset surface')
      self.gnuplot('set isosamples 2000, 2000')
      self.gnuplot('set table "setPlot.dat"')
      setPlot = self.gnuplot.splot(str(self.X.det().subs([(self.x0, Symbol('x')), (self.x1, Symbol('y'))])))
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
    self.y0All = [y[0, 0]]
    self.y1All = [y[1, 0]]

    FdS0 = self.Fd.subs([(self.x0, y[0, 0]), (self.x1, y[1, 0])])

    self.logStdout.info('AUXILIARY PATH-FOLLOWING')

    FdS = self.Fd.subs([(self.x0, y[0, 0]), (self.x1, y[1, 0])])
    FddS = self.Fdd.subs([(self.x0, y[0, 0]), (self.x1, y[1, 0])])
    while True:
      k += 1
      self.logStdout.info('\nk = ' + str(k))

      # iteration step
      t = t - gamma/Utils.LocalNormA(FdS0, FddS)
      y = y - FddS.inv()*(t*FdS0 + FdS)
      #self.logStdout.info('t = ' + str(t))
      self.logStdout.info('y = ' + str(y))

      self.y0All.append(y[0, 0])
      self.y1All.append(y[1, 0])

      # substitute to find gradient and hessian
      XS = self.X.subs([(self.x0, y[0, 0]), (self.x1, y[1, 0])])
      FdS = self.Fd.subs([(self.x0, y[0, 0]), (self.x1, y[1, 0])])
      FddS = self.Fdd.subs([(self.x0, y[0, 0]), (self.x1, y[1, 0])])

      # print eigenvalues
      if self.logStdout.isEnabledFor(logging.INFO):
        eigs = list(XS.eigenvals())
        eigs = [ re(N(eig)) for eig in eigs ]
        eigs.sort()
        self.logStdout.info('EIG = ' + str(eigs))

      # breaking condition
      if Utils.LocalNormA(FdS, FddS) <= sqrt(beta)/(1 + sqrt(beta)):
        break

    # prepare x
    x = y - FddS.inv()*FdS

    # plot auxiliary path
    if self.drawPlot:
      self.plot.add(self.y1All, xvals = self.y0All, title = 'Auxiliary path', w = 'points', pt = 1)
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

    x0All = [x[0, 0]]
    x1All = [x[1, 0]]

    # Main path-following scheme [Nesterov, p. 202]
    self.logStdout.info('\nMAIN PATH-FOLLOWING')

    # initialization of the iteration process
    t = 0
    eps = 10**(-3)
    k = 0

    # print the input condition to verify that is satisfied
    self.logStdout.info('Input condition = ' + str(Utils.LocalNormA(self.Fd.subs([(self.x0, x[0, 0]), (self.x1, x[1, 0])]), self.Fdd.subs([(self.x0, x[0, 0]), (self.x1, x[1, 0])]))))

    while True:
      k += 1
      self.logStdout.info('\nk = ' + str(k))

      # substitute to find gradient and hessian
      FdS = self.Fd.subs([(self.x0, x[0, 0]), (self.x1, x[1, 0])])
      FddS = self.Fdd.subs([(self.x0, x[0, 0]), (self.x1, x[1, 0])])

      # iteration step
      t = t + gamma/Utils.LocalNormA(self.c, FddS)
      x = x - FddS.inv()*(t*self.c+FdS)

      x0All.append(x[0, 0])
      x1All.append(x[1, 0])

      self.logStdout.info('t = ' + str(t))
      self.logStdout.info('x = ' + str(x))

      if self.logStdout.isEnabledFor(logging.INFO):
        # print eigenvalues
        XS = self.X.subs([(self.x0, x[0, 0]), (self.x1, x[1, 0])])
        eigs = list(XS.eigenvals())
        eigs = [ re(N(eig)) for eig in eigs ]
        eigs.sort()
        self.logStdout.info('EIG = ' + str(eigs))

      # breaking condition
      self.logStdout.info('Breaking condition = ' + str(eps*t))
      if eps*t >= self.nu + (beta + sqrt(self.nu))*beta/(1 - beta):
        break

    # plot main path
    if self.drawPlot:
      self.plot.add(x1All, xvals = x0All, title = 'Main path', w = 'points', pt = 1)
      self.gnuplot.show(self.plot)
      print('\nPress enter to continue')
      input()

    return x
