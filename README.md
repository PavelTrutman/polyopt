# Polynomial optimization problem solver and Semidefinite programming solver

This Python package enables you to solve semidefinite programming problems. Also provides a tool to convert a polynomial optimization problem into semidefinite programme.

## Usage

For demo, please refer to the files [demoSDPSolver.py](https://github.com/PavelTrutman/POP-SDP/blob/master/demoSDPSolver.py) and [demoPOPSolver.py](https://github.com/PavelTrutman/POP-SDP/blob/master/demoPOPSolver.py). These files are self-explanatory.

### Graph plotting

To enable plotting of the steps of the algorithm, use command 
```python
SDPSolver.setDrawPlot(True)
```
The package is using tool `gnuplot` to produce the graphs, so please install this tool first. Then a Python package `gnuplot-py` has to be installed. For example, you can use [gnuplot-py](https://github.com/PavelTrutman/gnuplot.py-py3k).

## Acknowledge

I would like to thanks to Didier Henrion for introducing me into polynomial optimization.
