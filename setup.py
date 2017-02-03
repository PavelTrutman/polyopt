#!/usr/bin/python3

from setuptools import setup

"""
Setup script for polynomial optimization module distribution.

by Pavel Trutman, pavel.tutman@fel.cvut.cz
"""

setup (
  # Distribution meta-data
  name='polyopt',
  version='2.1',
  description='Polynomial optimization problem solver.',
  long_description='This Python package enables you to solve semidefinite programming problems. Also provides a tool to convert a polynomial optimization problem into semidefinite programme.',
  author='Pavel Trutman',
  author_email='pavel.trutman@fel.cvut.cz',
  url='https://github.com/PavelTrutman/POP-SDP',
  package_dir={'PolyOpt' : '.'},
  packages = ['polyopt'],
  test_suite='tests',
)
