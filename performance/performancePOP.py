#!/usr/bin/python3

"""
Preformance measuring for POP solvers, saves results into file and plots them.

by Pavel Trutman, pavel.trutman@fel.cvut.cz
"""

from numpy import *
from numpy.random import uniform
from scipy.misc import comb
from polyopt import POPSolver
import timeit
import gnuplot as gp
import tempfile
import argparse

def generateVariablesDegree(d, n):
  """
  Generates whole set of variables of given degree.

  Args:
    d (int): degree of the variables
    n (int): number of unknowns

  Returns:
    list: list of all variables
  """

  # generate zero degree variables
  if d == 0:
    return [(0,)*n]

  # generrate one degree variables
  elif d == 1:
    variables = []
    for i in range(0, n):
      t = [0]*n
      t[i] = 1
      variables.append(tuple(t))
    return variables

  # there is only one unkown with the degree d
  elif n == 1:
    return [(d,)]

  # generate variables in general case
  else:
    variables = []
    for i in range(0, d + 1):
      innerVariables = generateVariablesDegree(d - i, n - 1)
      variables.extend([v + (i,) for v in innerVariables])
    return variables


def generateVariablesUpDegree(d, n):
  """
  Generates whole set of variables up to given degree.

  Args:
    d (int): maximal degree of the variables

  Returns:
    list: list of variables
  """

  variables = []
  for i in range(0, d + 1):
    variables.extend(generateVariablesDegree(i, n))
  return variables


# command line argument parser
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--help', '-h', action='help', help='show this help message and exit')
parser.add_argument('--bench', '-b', action='store_true', help='run benchmark')
parser.add_argument('--plot', '-p', action='store_true', help='plot results')
plotGroup = parser.add_argument_group('plot options', 'List of available graphs when --plot argument active')
plotGroup.add_argument('--relax-order', action='store_true', help='graph based on the relaxation order', dest='relaxOrder')
plotGroup.add_argument('--poly-degree', action='store_true', help='graph based on the degree of polynomials', dest='polynomialDegree')
plotGroup.add_argument('--num-variables', action='store_true', help='graph based on the number of variables', dest='dimension')
args = parser.parse_args()


# benchmarking
if args.bench:
  # dimension: 1, 2, 3
  dim = range(1, 4)
  
  # degree of generated polynomials: 2, 3, 4
  degF = [2, 4, 6]
  
  # relaxation orders: ceil(degF/2), ceil(degF/2) + 1, ceil(degF/2) + 2, ceil(degF/2) + 3
  order = range(0, 2)
  
  # number of feasible point generations: 10
  N = range(0, 20)
  
  # initialize the array with result times
  results = empty((max(dim) + 1, max(degF) + 1, len(order)))
  results.fill(nan)
  
  # for dimension
  for n in dim:
  
    #for degree of polynomials
    for dF in degF:
      variables = generateVariablesUpDegree(dF, n)
  
      # generate objective function
      f = {}
      for v in variables:
        f[v] = uniform(-1, 1)
  
      #generate constraining function
      t = [0]*n
      g = {tuple(t): 1}
      for i in range(0, n):
        t = [0]*n
        t[i] = 2
        g[tuple(t)] = -1
  
      minD = int(ceil(dF/2))
  
      # for relaxation order
      for dR in order:
        d = dR + minD
        print((n, dF, dR), ', SDP dim: {:3d} '.format(int(comb(n+2*d, n))), sep='', end='', flush=True)
  
        # clear measured time
        t = []
  
        # for different feasible points
        for i in N:
          # initialize the solver
          problem = POPSolver(f, g, d)
  
          # obtain some feasible point for the SDP problem
          y0 = problem.getFeasiblePoint(1)
  
          # startup time
          timeStart = timeit.default_timer()
  
          #solve the problem
          problem.solve(y0)
  
          # final time
          t.append(timeit.default_timer() - timeStart)

          print('.', end='', flush=True)
  
        print(' {:3.2f} s'.format(nanmedian(t)), sep='')
        results[n, dF, dR] = nanmedian(t)
  
  save('data/POP.npy', results)


# plotting
if args.plot:
  # load data form file
  data = load('data/POP.npy')
  print(data)
  shape = data.shape
  
  # based on dimension
  if args.dimension:
    # start GnuPlot
    gnuplot = gp.Gnuplot()
    gnuplot('set xlabel "Dimension"')
    gnuplot('set ylabel "Time [s]"')
    gnuplot('set title "Performance of the POP solver based on the number of variables."')
    
    # plot data
    tempFiles = []
    for df in range(0, shape[1]):
      for dr in range(0, shape[2]):
        dataSliced = data[:, df, dr]
        noNan = nonzero([~isnan(dataSliced)])[1]
        a = dataSliced[noNan]
        if a.shape[0] != 0:
          plotFile = tempfile.NamedTemporaryFile()
          tempFiles.append(plotFile)
          plot = gnuplot.replot(gp.Data(noNan, a, title = 'df = ' + str(df) + ', relaxOrder = ' + str(dr), with_ = 'lines lw 1.5', filename=plotFile.name))
    print('\nPress enter to continue')
    input()
  
    # close temp file
    for tempFile in tempFiles:
      tempFile.close()
          
  # based on relaxation order
  if args.relaxOrder:
    # start GnuPlot
    gnuplot = gp.Gnuplot()
    gnuplot('set xlabel "Relaxation order"')
    gnuplot('set ylabel "Time [s]"')
    gnuplot('set title "Performance of the POP solver based on the relaxation order."')
  
    # plot data
    tempFiles = []
    for n in range(0, shape[0]):
      for df in range(0, shape[1]):
        if isnan(data[n, df, :]).any():
          continue
        plotFile = tempfile.NamedTemporaryFile()
        tempFiles.append(plotFile)
        plot = gnuplot.replot(gp.Data(range(0, data.shape[2]), data[n, df, :], title = 'dim = ' + str(n) + ', df = ' + str(df), with_ = 'lines lw 1.5', filename=plotFile.name))
    print('\nPress enter to continue')
    input()
  
    # close temp file
    for tempFile in tempFiles:
      tempFile.close()
  
  # based on polynomial degree
  if args.polynomialDegree:
    # start GnuPlot
    gnuplot = gp.Gnuplot()
    gnuplot('set xlabel "Degree of the polynomial"')
    gnuplot('set ylabel "Time [s]"')
    gnuplot('set title "Performance of the POP solver based on the degree of polynomials."')
  
    # plot data
    tempFiles = []
    for n in range(0, shape[0]):
      dataSliced = data[n, :, :]
      noNan = nonzero([~isnan(dataSliced).any(axis=1)])[1]
      a = dataSliced[noNan, :]
      if a.shape[0] != 0:
        for relaxOrder in range(0, shape[2]):
          
          plotFile = tempfile.NamedTemporaryFile()
          tempFiles.append(plotFile)
          plot = gnuplot.replot(gp.Data(noNan, a[:, relaxOrder], title = 'dim = ' + str(n) + ', relaxOrder = ' + str(relaxOrder), with_ = 'lines lw 1.5', filename=plotFile.name))
    print('\nPress enter to continue')
    input()
  
    # close temp file
    for tempFile in tempFiles:
      tempFile.close()
