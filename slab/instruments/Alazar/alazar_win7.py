# -*- coding: utf-8 -*-
"""
Created on Tue Jun 07 15:01:07 2011

@author: Phil
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:14:11 2011

@author: Phil
"""
import ctypes as C
import numpy as np
import sys
# from .slablayout import *
# try:
#     from guiqwt.pyplot import *
# except:
#     print "Warning unable to import guiqwt.pyplot"
from scipy.fftpack import fft,rfft
from slab.dataanalysis import heterodyne
from numpy import sin,cos,pi
#import matplotlib.pyplot as mplt
#import operator
import time
#import fftw3
import os

U8 = C.c_uint8
U8P = C.POINTER(U8)
U32 = C.c_uint
U32P = C.POINTER(U32)
U32PP = C.POINTER(U32P)
aa = C.CDLL(r'C:\Windows\SysWow64\ATSApi64.dll')

U8 = C.c_uint8
U8P = C.POINTER(U8)
U32 = C.c_uint
U32P = C.POINTER(U32)
U32PP = C.POINTER(U32P)
aa = C.CDLL(r'C:\Windows\SysWow64\ATSApi64.dll')
MEM_COMMIT = 0x1000
PAGE_READWRITE = 0x4
size_bytes = 98304
C.windll.kernel32.VirtualAlloc.argtypes = [C.c_void_p, C.c_long, C.c_long, C.c_long]
C.windll.kernel32.VirtualAlloc.restype = C.c_int
bb = C.windll.kernel32.VirtualAlloc(0, C.c_long(size_bytes), MEM_COMMIT, PAGE_READWRITE)
C.windll.kernel32.VirtualAlloc.argtypes = [C.c_void_p, C.c_long, C.c_long, C.c_long]
C.windll.kernel32.VirtualAlloc.restype = C.c_void_p
cc = C.windll.kernel32.VirtualAlloc(0, C.c_long(size_bytes), MEM_COMMIT, PAGE_READWRITE)
print(hex(bb), hex(cc))

bufstr = C.create_string_buffer(32)
print(hex(C.addressof(bufstr)))
