#!/usr/bin/python3

"""
Preformance measuring for SDP solvers, saves results into file and plots them.

by Pavel Trutman, pavel.trutman@fel.cvut.cz
"""

from numpy import *
from polyopt import SDPSolver
import timeit
from polyopt.utils import Utils
import time
import subprocess
import warnings
import argparse
import gnuplot as gp
import glob
import os


# command line argument parser
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--help', '-h', action='help', help='show this help message and exit')
parser.add_argument('--bench', '-b', action='store_true', help='run benchmark')
parser.add_argument('--plot', '-p', action='store_true', help='plot results')
args = parser.parse_args()


# benchmarking
if args.bench:
  # dimension: 1, ..., 10
  dim = [1, 2, 3, 5, 7, 10, 13, 16, 20, 25]
  
  # number of trials
  N = range(0, 20)
  
  # initialize the array with result times
  results = empty([max(dim) + 1, len(N)])
  results.fill(nan)
  
  # for dimension
  for n in dim:
  
    # clear measured time
    #t = 0
  
    # for different feasible points
    for i in N:
  
      # starting point
      startPoint = zeros((n, 1));
  
      # objective function
      c = ones((n, 1))
  
      # get LMI matrices
      A = [identity(n)];
      for _ in range(0, n):
        A.append(Utils.randomSymetric(n))
  
      # init SDP program
      problem = SDPSolver(c, [A])
  
      # bound the problem
      problem.bound(1)
  
      # startup time
      timeStart = timeit.default_timer()
  
      #solve the problem
      problem.solve(startPoint, problem.dampedNewton)
  
      # final time
      results[n, i] = timeit.default_timer() - timeStart
  
    print('dimension:', n)
    #results[n] = t/len(N)
  
  # median computation
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=RuntimeWarning)
    results = nanmedian(results, axis=1)
  
  date = time.strftime('%Y%m%d_%H%M')
  gitVersion = subprocess.check_output(['git', 'describe', '--always'])
  filename = 'data/SDP_' + date + '_' + gitVersion.decode('utf-8').strip() + '.npy'
  save(filename, results)
  print(results)
  print(filename)


# plotting
if args.plot:
  # list data files
  os.chdir('data')
  files = glob.glob('SDP_*.npy')
  files.sort()
  
  # start GnuPlot
  gnuplot = gp.Gnuplot()
  gnuplot('set xlabel "Dimension"')
  gnuplot('set ylabel "Time [s]"')
  gnuplot('set title "Performance of SDP solver on dimension of the problem."')
  
  for path in files:
    # load data from file
    data = load(path)
    shape = data.shape
  
    parts = path[:-4].split('_')
    print(parts)
    date = parts[1][0:4] + '.' + parts[1][4:6] + '.' + parts[1][6:8] + ' ' + parts[2][0:2] + ':' + parts[2][2:4]
    
    # plot them
    dims = []
    plotData = []
    for n in range(0, shape[0]):
      if not isnan(data[n]):
        dims.append(n)
        plotData.append(data[n])
    
    plot = gnuplot.replot(gp.Data(dims, plotData, title = date + ' ' + parts[3], with_ = 'lines'))

  print('\nPress enter to continue')
  input()
