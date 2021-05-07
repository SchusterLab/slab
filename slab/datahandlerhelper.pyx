# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:18:24 2018

@author: slab
"""

cdef extern from "keysightSD1c.lib":
    short* SD_AIN_DAQbufferGet(int moduleID, int nDAQ, int &readPointsOut,
                               int &errorOut)
    int SD_AIN_DAQbufferPoolConfig(int moduleID, int nDAQ, short* dataBuffer,
                                   int nPoints, int timeOut,
                                   callbackEventPtr callbackFunction,
                                   void *callbackUserObj)