# -*- coding: utf-8 -*-
"""
Created on Thu Nov 01 16:51:01 2012

@author: slab
"""

import subprocess
import sys
try:
    subprocess.call(["python", "C:\_Lib\python\slab\plotting\hdfiview.py", sys.argv[1]])
except IndexError:
    subprocess.call(["python", "C:\_Lib\python\slab\plotting\hdfiview.py"])    