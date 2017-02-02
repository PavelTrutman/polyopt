#!/usr/bin/python3

"""
Plots performance graphs.

by Pavel Trutman, pavel.trutman@fel.cvut.cz
"""

from numpy import *
import gnuplot as gp
import glob
import os

# list data files
os.chdir('performanceResults')
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
    if not isnan(data[n, 0]):
      dims.append(n)
      plotData.append(data[n, 0])
  
  plot = gnuplot.replot(gp.Data(dims, plotData, title = date + ' ' + parts[3], with_ = 'lines'))
input()
