# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 03:23:49 2018

@author: slab
"""

import time
import numpy as np

def func(x, *args):
    print(args)
    
list = [1, 2, 3]

func(0, *list)