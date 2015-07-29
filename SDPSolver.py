#!/usr/bin/python3

from sympy import *
from utils import Utils
import pyGnuplot as gp

# some constants
beta = 1/9
gamma = 5/36

class SDPSolver:

  # min c_0*x_0 + c_1*x_1
  # s.t. I_3 + A_0*x_0 + A_1*x_1 >= 0

  def __init__(self, c, A0, A1):
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

    # enable plotting
    self.plotEnable = False



  def enablePlot(self, plotEnable):
    self.plotEnable = plotEnable

    if self.plotEnable:
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



  def solve(self, start):
    x0 = self.auxFollow(start)
    return self.mainFollow(x0)



  def auxFollow(self, start):

    # Auxiliary path-following scheme [Nesterov, p. 205]
    t = 1
    k = 0

    # starting point
    y = start
    self.y0All = [y[0, 0]]
    self.y1All = [y[1, 0]]

    FdS0 = self.Fd.subs([(self.x0, y[0, 0]), (self.x1, y[1, 0])])

    print('AUXILIARY PATH-FOLLOWING')
    FdS = self.Fd.subs([(self.x0, y[0, 0]), (self.x1, y[1, 0])])
    FddS = self.Fdd.subs([(self.x0, y[0, 0]), (self.x1, y[1, 0])])
    while True:
      k += 1
      print('\nk = ' + str(k))

      # iteration step
      t = t - gamma/Utils.LocalNormA(FdS0, FddS)
      y = y - FddS.inv()*(t*FdS0 + FdS)
      #print('t = ' + str(t))
      print('y = ' + str(y))

      self.y0All.append(y[0, 0])
      self.y1All.append(y[1, 0])

      # substitute to find gradient and hessian
      XS = self.X.subs([(self.x0, y[0, 0]), (self.x1, y[1, 0])])
      FdS = self.Fd.subs([(self.x0, y[0, 0]), (self.x1, y[1, 0])])
      FddS = self.Fdd.subs([(self.x0, y[0, 0]), (self.x1, y[1, 0])])

      # print eigenvalues
      eigs = list(XS.eigenvals())
      eigs = [ re(N(eig)) for eig in eigs ]
      eigs.sort()
      print('EIG = ' + str(eigs))

      # breaking condition
      #print('Breaking condition = ' + str(Utils.LocalNormA(FdS, FddS)))
      if Utils.LocalNormA(FdS, FddS) <= sqrt(beta)/(1 + sqrt(beta)):
        break

    # prepare x
    x = y - FddS.inv()*FdS

    # plot auxiliary path
    if self.plotEnable:
      self.plot.add(self.y1All, xvals = self.y0All, title = 'Auxiliary path', w = 'points', pt = 1)
      self.gnuplot.show(self.plot)
    return x



  def mainFollow(self, x):
    x0All = [x[0, 0]]
    x1All = [x[1, 0]]

    # Main path-following scheme [Nesterov, p. 202]
    print('\nMAIN PATH-FOLLOWING')

    # initialization of the iteration process
    t = 0
    eps = 10**(-3)
    k = 0

    # print the input condition to verify that is satisfied
    print('Input condition = ' + str(Utils.LocalNormA(self.Fd.subs([(self.x0, x[0, 0]), (self.x1, x[1, 0])]), self.Fdd.subs([(self.x0, x[0, 0]), (self.x1, x[1, 0])]))))

    while True:
      k += 1
      print('\nk = ' + str(k))

      # substitute to find gradient and hessian
      FdS = self.Fd.subs([(self.x0, x[0, 0]), (self.x1, x[1, 0])])
      FddS = self.Fdd.subs([(self.x0, x[0, 0]), (self.x1, x[1, 0])])

      # iteration step
      t = t + gamma/Utils.LocalNormA(self.c, FddS)
      x = x - FddS.inv()*(t*self.c+FdS)

      x0All.append(x[0, 0])
      x1All.append(x[1, 0])

      #print('t = ' + str(t))
      print('x = ' + str(x))

      # print eigenvalues
      XS = self.X.subs([(self.x0, x[0, 0]), (self.x1, x[1, 0])])
      eigs = list(XS.eigenvals())
      eigs = [ re(N(eig)) for eig in eigs ]
      eigs.sort()
      print('EIG = ' + str(eigs))

      # breaking condition
      print('Breaking condition = ' + str(eps*t))
      if eps*t >= self.nu + (beta + sqrt(self.nu))*beta/(1 - beta):
        break

    # plot main path
    if self.plotEnable:
      self.plot.add(x1All, xvals = x0All, title = 'Main path', w = 'points', pt = 1)
      self.gnuplot.show(self.plot)

    print('\nPress enter to continue')
    input()

    return x
