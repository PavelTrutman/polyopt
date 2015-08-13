#!/usr/bin/python3

"""
Plots performance graphs.

by Pavel Trutman, pavel.trutman@fel.cvut.cz
"""

from numpy import *
import pyGnuplot as gp
from time import sleep

# what do you want to view?
relaxOrder = True
polynomialDegree = True

# load data form file
data = load('performanceResults.npy')
shape = data.shape

if relaxOrder:
  # start GnuPlot
  gnuplot = gp.gnuplot()
  gnuplot('set xlabel "Relaxation order"')
  gnuplot('set ylabel "Time [s]"')
  for n in range(0, shape[0]):
    gnuplot('set title "Dimension ' + str(n) + '"')
    first = True
    for df in range(0, shape[1]):
      if isnan(data[n, df, :]).any():
        continue
      if first:
        plot = gnuplot.plot(data[n, df, :], title = 'dim = ' + str(n) + ', df = ' + str(df), w = 'lines')
        first = False
      else:
        plot.add(data[n, df, :], title = 'dim = ' + str(n) + ', df = ' + str(df), w = 'lines')
    if not first:
      gnuplot.newXterm()
      gnuplot.show(plot)
      sleep(0.5)
  input()

if polynomialDegree:
  # start GnuPlot
  gnuplot = gp.gnuplot()
  gnuplot('set xlabel "Degree of the polynomial"')
  gnuplot('set ylabel "Time [s]"')
  for n in range(0, shape[0]):
    gnuplot('set title "Dimension ' + str(n) + '"')
    first = True
    a = data[n, :, :][~isnan(data[n, :, :]).any(axis=1)]
    plot = None
    if a.shape[0] != 0:
      for relaxOrder in range(0, shape[2]):
        if first:
          plot = gnuplot.plot(a[:, relaxOrder], xvals = range(shape[2] - a.shape[0] + 1, shape[2] + 1), title = 'dim = ' + str(n) + ', relaxOrder = ' + str(relaxOrder), w = 'lines')
          first = False
        else:
          plot.add(a[:, relaxOrder], xvals = range(shape[2] - a.shape[0] + 1, shape[2] + 1), title = 'dim = ' + str(n) + ', relaxOrder = ' + str(relaxOrder), w = 'lines')
      if plot != None:
        gnuplot.newXterm()
        gnuplot.show(plot)
        sleep(0.5)
  input()
