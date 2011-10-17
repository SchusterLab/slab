# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 14:04:03 2011

@author: Dave
"""

import numpy as np
from guiqwt.pyplot import *
import glob
import os.path
from numpy import linspace,argmin,argmax, argsort, genfromtxt,loadtxt,array,transpose
import dateutil
import time
###################################################

#### General

def dBm_to_W(dBm):
    """convert dBm to Watts"""
    return 10**(dBm/10.)

dBmtoW=dBm_to_W

########################

def find_closest_index(a,v):
    return np.argsort(abs(a-v))[0]

def load_ppms_file(filename):
    return np.transpose(np.array(np.genfromtxt (filename, delimiter=',',usecols=(3,8,10),skip_header=33,missing_values='',filling_values=1e-8)))

########### Network Analyzer Analysis Functions ####
def load_nwa_file(filename):
    """return three arrays: frequency magnitude and phase"""
    return np.transpose(np.loadtxt(filename,skiprows=3,delimiter=','))
    
def load_nwa_dir(datapath):
    fnames=glob.glob(os.path.join(datapath,"*.CSV"))
    fnames.sort()
    prefixes = [os.path.split(fname)[-1] for fname in fnames]
    data = [load_nwa_file(fname) for fname in fnames]
    return prefixes, data
    
############## Experiment helpers #####################

def next_file_index(datapath,prefix=''):
    """Searches directories for files of the form *_prefix* and returns next number
        in the series"""
        
    dirlist=glob.glob(os.path.join(datapath,'*_'+prefix+'*'))
    dirlist.sort()
    try:
        ii=int(os.path.split(dirlist[-1])[-1].split('_')[0])+1
    except:
        ii=0
    return ii

def date_tag(date_str=None):
    if date_str is None:
        lt=time.localtime()
        return "%02d%02d%02d" % (lt.tm_year % 100,lt.tm_mon,lt.tm_mday)
    else:
        lt=dateutil.parser.parse(date_str)
        return "%02d%02d%02d" % (lt.year % 100,lt.month,lt.day)
    
def next_path_index(expt_path,prefix=''):
    dirlist=glob.glob(os.path.join(expt_path,'*'+prefix+'*[0-9][0-9][0-9]'))
    if dirlist == []:
        return 0
    dirlist.sort()
    return int(os.path.split(dirlist[-1])[-1].split('_')[-1])+1
    
def get_next_filename(datapath,prefix,suffix=''):
    ii=next_file_index(datapath,prefix)
    return "%03d_" % (ii) + prefix +suffix

def make_datapath(expt_path,prefix,date_str=None):
    tag=date_tag(date_str)
    ii=next_path_index(expt_path,prefix)
    datapath=os.path.join(expt_path,"%s_%s_%03d" % (tag,prefix,ii) )
    os.mkdir(datapath)
    return datapath
    
def current_datapath(expt_path,prefix,date_str=None):
    tag=date_tag(date_str)
    ii=next_path_index(expt_path,prefix) - 1
    return os.path.join(expt_path,"%s_%s_%03d\\" % (tag,prefix,ii) )
    