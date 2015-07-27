#!/usr/bin/python3

#from numpy import *
from sympy import *
from utils import Utils

# min c_0*x_0 + c_1*x_1
# s.t. I_3 + A_0*x_0 + A_1*x_1 >= 0

# initialization of the problem
c = Matrix([[3], [-2]])
#A0 = Matrix([[3, 5, 13], [5, 8, 7], [13, 7, 9]])
#A1 = Matrix([[11, 2, 3], [2, 3, 4], [3, 4, 5]])
A0 = Matrix([[1, 1, 0],
             [1, 1, 0],
             [0, 0, 0]])
A1 = Matrix([[1, 1, 1],
             [1, 0, 0],
             [1, 0, 1]])

# symbolic variables
x0 = Symbol('x0')
x1 = Symbol('x1')

# self-concordant barrier
#X = simplify(eye(3) + A0*x0 + A1*x1)
#print('X = ' + str(X))
#F = -log(X.det())
#print('F = ' + str(F))
nu = 3

# first symbolic derivation
#Fdx0 = diff(F, x0)
#Fdx1 = diff(F, x1)
#Fd = Matrix([[Fdx0], [Fdx1]])
#print('Fd = ' + str(simplify(Fd)))

# symbolic hessian
#Fddx0x0 = diff(Fdx0, x0)
#Fddx1x1 = diff(Fdx1, x1)
#Fddx0x1 = diff(Fdx0, x1)
#Fdd = Matrix([[Fddx0x0, Fddx0x1], [Fddx0x1, Fddx1x1]])
#print('Fdd = ' + str(simplify(Fdd)))

# some constants
beta = 1/9
gamma = 5/36

# Auxiliary path-following scheme [Nesterov, p. 205]
t = 1
k = 0
# starting point
y = Matrix([[0], [0]])

#FdS0 = Fd.subs([(x0, y[0, 0]), (x1, y[1, 0])])
F = eye(3) + A0*y[0, 0] + A1*y[1, 0]
Fi = F.inv()
Fi0 = A0*Fi
Fi1 = A1*Fi
g0 = -trace(Fi0)
g1 = -trace(Fi1)
g = Matrix([[g0], [g1]])
h00 = trace(Fi0**2)
h11 = trace(Fi1**2)
h01 = trace(Fi0*Fi1)
H = Matrix([[h00, h01], [h01, h11]])

gy0 = g

print('\n\ngy0 = ' + str(gy0))

print('AUXILIARY PATH-FOLLOWING')
#FdS = Fd.subs([(x0, y[0, 0]), (x1, y[1, 0])])
#FddS = Fdd.subs([(x0, y[0, 0]), (x1, y[1, 0])])
while True:
  k += 1
  print('\nk = ' + str(k))
  print('g = ' + str(g))
  print('H = ' + str(H))

  t = t - gamma/Utils.LocalNormA(gy0, H)
  y = y - H.inv()*(t*gy0 + g)
  print('t = ' + str(t))
  print('y = ' + str(y))

  #FdS = Fd.subs([(x0, y[0, 0]), (x1, y[1, 0])])
  #FddS = Fdd.subs([(x0, y[0, 0]), (x1, y[1, 0])])
  F = eye(3) + A0*y[0, 0] + A1*y[1, 0]
  Fi = F.inv()
  Fi0 = A0*Fi
  Fi1 = A1*Fi
  g0 = -trace(Fi0)
  g1 = -trace(Fi1)
  g = Matrix([[g0], [g1]])
  h00 = trace(Fi0**2)
  h11 = trace(Fi1**2)
  h01 = trace(Fi0*Fi1)
  H = Matrix([[h00, h01], [h01, h11]])

  print('Breaking condition = ' + str(Utils.LocalNorm(g, H)))
  if Utils.LocalNorm(g, H) <= sqrt(beta)/(1 + sqrt(beta)):
    break

# prepare x
x = y - H.inv()*g

# Main path-following scheme [Nesterov, p. 202]
print('\nMAIN PATH-FOLLOWING')

# initialization of the iteration process
t = 0
eps = 10**(-3)
k = 0

F = eye(3) + A0*x[0, 0] + A1*x[1, 0]
Fi = F.inv()
Fi0 = A0*Fi
Fi1 = A1*Fi
g0 = -trace(Fi0)
g1 = -trace(Fi1)
g = Matrix([[g0], [g1]])
h00 = trace(Fi0**2)
h11 = trace(Fi1**2)
h01 = trace(Fi0*Fi1)
H = Matrix([[h00, h01], [h01, h11]])

print('Input condition = ' + str(Utils.LocalNormA(g, H)))

#print('\nPress enter to continue')
#input()

#while True:
#  k += 1
#  print('\nk = ' + str(k))
#  FdS = Fd.subs([(x0, x[0, 0]), (x1, x[1, 0])])
#  FddS = Fdd.subs([(x0, x[0, 0]), (x1, x[1, 0])])
#  t = t + gamma/Utils.LocalNormA(c, FddS)
#  x = x - FddS.inv()*(t*c+FdS)
#  print('t = ' + str(t))
#  print('x = ' + str(x))
#  print('Breaking condition = ' + str(eps*t))
#  if eps*t >= nu + (beta + sqrt(nu))*beta/(1 - beta):
#    break

