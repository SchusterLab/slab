# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 14:04:03 2011

@author: Dave
"""

import numpy as np
#from guiqwt.pyplot import *
import glob
import os.path
from scipy.signal import decimate    
from numpy import linspace,argmin,argmax, argsort, genfromtxt,loadtxt,array,transpose,pi,cos,sin,arctan2,convolve,correlate,sum,sqrt,ones,zeros,arange
import dateutil
import time
#import inspect
from datamanagement import get_script
###################################################

#### General

def dBm_to_W(dBm):
    """convert dBm to Watts"""
    return 10**(dBm/10.)

dBmtoW=dBm_to_W

######################## File handling

def save_script(expt_path,prefix):
    """This function saves the file of the current running script to a file in the standard file naming convention:
        expt_path\date_prefix_number"""
    tag=date_tag(None)
    ii=next_path_index(expt_path,prefix)
    fname=os.path.join(expt_path,"%s_%s_%03d.py" % (tag,prefix,ii) )    
    fw=open(fname,'w')
    fw.write(get_script())
    fw.close()
    return fname
def find_closest_index(a,v):
    return argsort(abs(a-v))[0]

def load_ppms_file(filename):
    return transpose(array(genfromtxt (filename, delimiter=',',usecols=(3,8,10),skip_header=33,missing_values='',filling_values=1e-8)))

########### Network Analyzer Analysis Functions ####
def load_nwa_file(filename):
    """return three arrays: frequency magnitude and phase"""
    return transpose(loadtxt(filename,skiprows=3,delimiter=','))
    
def load_nwa_dir(datapath):
    fnames=glob.glob(os.path.join(datapath,"*.CSV"))
    fnames.sort()
    prefixes = [os.path.split(fname)[-1] for fname in fnames]
    data = [load_nwa_file(fname) for fname in fnames]
    return prefixes, data
    
def load_nwa_list(datapath, header_list):
    fnames=glob.glob(os.path.join(datapath,"*.CSV"))
    prefixes=[];fnames2=[];
    fnames.sort()
    for fname in fnames:
        prefix=os.path.split(fname)[-1]
        number=prefix.split('_')[0]
        if int(number) in header_list: 
            prefixes.append(prefix)
            fnames2.append(fname)
#            print 'filename', fname
#    print 'filenames 2',fnames2
    data = [load_nwa_file(fname) for fname in fnames2]
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
    return "%04d_" % (ii) + prefix +suffix

def make_datapath(expt_path,prefix,date_str=None):
    """Automatically makes a new folder in the experiment folder with new index"""
    tag=date_tag(date_str)
    ii=next_path_index(expt_path,prefix)
    datapath=os.path.join(expt_path,"%s_%s_%03d" % (tag,prefix,ii) )
    os.mkdir(datapath)
    if datapath[-1]!='\\': datapath+="\\"
    return datapath
    
def current_datapath(expt_path,prefix,date_str=None):
    tag=date_tag(date_str)
    ii=next_path_index(expt_path,prefix) - 1
    return os.path.join(expt_path,"%s_%s_%03d\\" % (tag,prefix,ii) )
    
def tic():
    global last_tic
    last_tic=time.time()
    
def toc(log=False):
    global last_tic
    t=time.time()
    if log: print "Tic-Toc: %.0f ms" % ((t-last_tic)*1000.)
    return t-last_tic
    
def digital_homodyne(time_pts,ch1_pts,ch2_pts=None,IFfreq=1,dfactor=1.,AmpPhase=False):
    '''digital_homodyne computes I/Q or Amp/Phase as a function of time of a particular frequency component.
       @param time_pts: time of each sample
       @param ch1_pts: Scope channel 1 points 
       @param ch2_pts: Scope channel 2 points (optional)
       @param IFfreq: Frequency to extract info for
       @param dfactor: decimation factor number of periods to include per point
       @param AmpPhase: returns I/Q if False (default) or Amp/Phase if True
       '''    

    df=int(dfactor/(time_pts[1]-time_pts[0])/IFfreq)
    cospts=(2./sqrt(df))*cos(2.*pi*IFfreq*time_pts)
    sinpts=(2./sqrt(df))*sin(2.*pi*IFfreq*time_pts)
    in2=ones(df,dtype=float)/sqrt(df)

    I1pts=decimate(convolve(ch1_pts*cospts,in2,mode='valid'),df,n=2)
    Q1pts=decimate(convolve(ch1_pts*sinpts,in2,mode='valid'),df,n=2)
    dtpts=arange(len(I1pts))*((time_pts[1]-time_pts[0])*df)+time_pts[0]
    
    if ch2_pts is not None:
        I2pts=decimate(convolve(ch2_pts*cospts,in2,mode='valid'),df,n=2)
        Q2pts=decimate(convolve(ch2_pts*sinpts,in2,mode='valid'),df,n=2)
        
    if AmpPhase:
        amp1pts=sqrt(I1pts**2+Q1pts**2)
        phi1pts=arctan2(I1pts,Q1pts)*180./pi
        if ch2_pts is not None:
            amp2pts=sqrt(I2pts**2+Q2pts**2)
            phi2pts=arctan2(I2pts,Q2pts)*180./pi
            return dtpts,amp1pts,phi1pts,amp2pts,phi2pts
        else: return dtpts,amp1pts,phi1pts
    else:
        if ch2_pts is not None:
            return dtpts,I1pts,Q1pts,I2pts,Q2pts            
        else:
            return dtpts,I1pts,Q1pts
 
def heterodyne(time_pts,ch1_pts,ch2_pts=None,IFfreq=1,AmpPhase=True):
    '''digital_homodyne computes I/Q or Amp/Phase as a function of time of a particular frequency component.
       @param time_pts: time of each sample
       @param ch1_pts: Scope channel 1 points 
       @param ch2_pts: Scope channel 2 points (optional)
       @param IFfreq: Frequency to extract info for
       @param dfactor: decimation factor number of periods to include per point
       @param AmpPhase: returns I/Q if False (default) or Amp/Phase if True
       '''
    if ch2_pts is None:
        ch2_pts = zeros(len(time_pts))
        
    cospts=cos(2.*pi*IFfreq*time_pts)
    sinpts=sin(2.*pi*IFfreq*time_pts)
    A1=2.*sum(cospts*ch1_pts)/len(time_pts)
    B1=2.*sum(sinpts*ch1_pts)/len(time_pts)
    A2=2.*sum(cospts*ch2_pts)/len(time_pts)
    B2=2.*sum(sinpts*ch2_pts)/len(time_pts)
    if AmpPhase:
        return sqrt(A1**2+B1**2), arctan2(A1,B1)*180./pi,sqrt(A2**2+B2**2), arctan2(A2,B2)*180./pi
    else:
        return A1,B1,A2,B2
