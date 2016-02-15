#! /usr/bin/env python

# setup.py
#
# Setup and Install of llstring
# 
# Copyright 2016 Massachusetts Institute of Technology, Lincoln Laboratory
# version 0.1
#
# author: Charlie Dagli
# dagli@ll.mit.edu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Imports 
#
from setuptools import setup

#
# Head-off numpy/scipy install issues
#
np_check = True; sp_check = True

try: import numpy
except ImportError: np_check = False

try: import scipy
except ImportError: sp_check = False

if (not np_check) or (not sp_check):
    msg =  "llstring requires numpy and scipy dependencies to be pre-installed on " \
           "the host environment before installation. This is because optimized " \
           "installations of numpy and scipy depend on non-python system build " \
           "dependencies (i.e. gcc, gfortran, BLAS & LAPACK headers) which make " \
           "building numpy and scipy from source with python's setuptools " \
           "un-tractable. (This is a known python issue.) We recommend either " \
           "pre-installation system-wide or in a python environment manager such " \
           "as Anaconda.\n"
           #"dependencies (i.e. gcc, gfortran, BLAS & LAPACK headers) which make " \ 

    if np_check: tag = "(numpy: Installed, "
    else: tag = "(numpy: Not-Installed, "

    if sp_check: tag += " scipy: Installed)"
    else: tag += " scipy: Not-Installed)"

    msg += tag

    raise ImportError(msg)


#
# Do Setup
#
setup(name='llstring',
      version='0.1',
      description='MIT-LL String Processing and Matching Tools',
      url='https://g62code.llan.ll.mit.edu/cdagli/mitll-string-match',
      author='Charlie Dagli',
      author_email='dagli@ll.mit.edu',
      license='APLv2',
      packages=['llstring'],
      install_requires=[
          'jellyfish',
          'sklearn', #sklearn depends on both numpy and scipy packages.
                     #numpy and scipy are not included in this list b/c
                     #doing so would build numpy and scipy from source
                     #and depending on your setup could result in a really
                     #non-optimized set of linear algebra libraries for
                     #your particular system. 
      ],
      zip_safe=False)
