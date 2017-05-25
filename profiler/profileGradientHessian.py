#!/usr/bin/python3

import timeit

repeat = 35;
count = 35;

setup = '''
import polyopt
import pickle
with open('AAll.pickle', 'rb') as f:
  AAll = pickle.load(f)
with open('x.pickle', 'rb') as f:
  x = pickle.load(f)
'''

t = timeit.Timer('polyopt.utils.gradientHessian(AAll, x)', setup=setup)
times = t.repeat(repeat, count)
print(times)
print('{:4.3g} ms'.format(min(times)/count*1000))
