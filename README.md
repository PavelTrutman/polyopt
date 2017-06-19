# Polynomial optimization problem solver and Semidefinite programming solver

This Python package enables you to solve semidefinite programming problems. Also provides a tool to convert a polynomial optimization problem into semidefinite programme.

## Installation

To install all required packages and setup this package, run the following code.
```bash
git clone https://github.com/PavelTrutman/polyopt.git
cd polyopt
pip3 install -r requirements.txt
python3 setup.py install
```

To run the tests, execute:
```bash
python3 setup.py test
```

## Usage

For demo, please refer to the files [demoSDPSolver.py](https://github.com/PavelTrutman/polyopt/blob/master/demoSDPSolver.py) and [demoPOPSolver.py](https://github.com/PavelTrutman/polyopt/blob/master/demoPOPSolver.py). These files are self-explanatory.

### SDP solver

```python
from numpy import *
import polyopt

# Problem statement
# min c0*x0 + c1*x1
# s. t. I_3 + A0*x0 + A1*x1 >= 0

c = array([[1], [1]])
A0 = array([[1,  0,  0],
             [0, -1,  0],
             [0,  0, -1]])
A1 = array([[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]])
# starting point 
startPoint = array([[0], [0]])

# create the solver object
problem = polyopt.SDPSolver(c, [[eye(3), A0, A1]])

# enable graphs
problem.setDrawPlot(True)

# enable informative output
problem.setPrintOutput(True)

# enable bounding into ball with radius 1
#problem.bound(1)

# solve!
x = problem.solve(startPoint, problem.dampedNewton)
print(x) 
```

### POP solver

```python
from numpy import *
import polyopt

# objective function
# f(x, y) = (x - 1)^2 + (y - 2)^2
#         = x^2 -2*x + y^2 - 4*y + 5
# global minima at (1, 2)
f = {(0, 0): 5, (1, 0): -2, (2, 0): 1, (0, 1): -4, (0, 2): 1}

# constraint function
# g(x, y) = 9 - x^2 - y^2
g = [{(0, 0): 3**2, (2, 0): -1, (0, 2): -1}]

# degree of the relaxation
d = 2

# initialize the solver
POP = polyopt.POPSolver(f, g, d)

# obtain some feasible point for the SDP problem (within ball with radius 3)
y0 = POP.getFeasiblePointFromRadius(3)
# or select some feasible points of the polynomial optimization problem
y0 = POP.getFeasiblePoint([array([[1],[1]]), array([[2],[2]]), array([[-1], [-1]]), array([[-2], [1]]), array([[1], [2]]), array([[0], [2]])])

# enable outputs
POP.setPrintOutput(True)

#solve the problem
x = POP.solve(y0)
print(x)

print('Rank of the moment matrix: ', POP.momentMatrixRank())
```

### Graph plotting

To enable plotting of the steps of the algorithm, use command 
```python
SDPSolver.setDrawPlot(True)
```
The package is using tool `gnuplot` to produce the graphs, so please install this tool first. Then a Python package `gnuplot-py` has to be installed. For example, you can use [gnuplot-py](https://github.com/PavelTrutman/gnuplot.py-py3k).

## Acknowledge

I would like to thanks to Didier Henrion for introducing me into polynomial optimization.
