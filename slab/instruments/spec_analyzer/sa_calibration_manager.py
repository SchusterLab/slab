# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 12:10:40 2011

@author: Dai
"""

class SACalibrationManager:
    #calibration data is a 2D array with cell in [frequency, sa ouput, rf_power]
    #cali_data = None
    #calibrated frequencies
    cali_freq = []
    #calibrated [output, power] lists in a dictionary indexed by the frequency
    cali_pwr = {}
    #lowest calibrated frequency
    lower_freq_bound = 0
    #highest calibrated frequency
    upper_freq_bound = 0
    
    def __init__(self, cali_data = None):
        self.load_calibration_profile(cali_data)
    
    def load_calibration_profile(self, cali_data):
        if cali_data != None:
            #self.cali_data = np.array(cali_data)
            for freq, op, pwr in cali_data:
                if not freq in self.cali_pwr.keys():
                    self.cali_pwr[freq] = []
                else:
                    self.cali_pwr[freq].append([op, pwr])
                    
            for k in self.cali_pwr:
                self.cali_pwr[k].sort()                
            self.cali_freq = self.cali_pwr.keys()
            self.cali_freq.sort()
            self.lower_freq_bound = self.cali_freq[0]
            self.upper_freq_bound = self.cali_freq[len(self.cali_freq)-1]
            
    
    def isCalibrated(self):
        return not self.cali_data == None
    
    def interpolate(self, freq, op, ll, lh, hl, hh):
        """interpolate based on neareast four data points in 
        [freq, op, pwr] format to return a RF power reading"""
        
        #first linearly interpolate along output
        if lh[1] - ll[1] == 0:
            lpwr = ll[2]
        else:
            lpwr = (lh[2] * (op - ll[1]) + ll[2] * (lh[1] - op)) / (lh[1] - ll[1])

        if ll[0] == hl[0]:
            #the lower and higher frequencies are the same
            return lpwr
        
        if hh[1] - hl[1] == 0:
            hpwr = hl[2]
        else:
            hpwr = (hh[2] * (op - hl[1]) + hl[2] * (hh[1] - op)) / (hh[1] - hl[1])
        
        #then linearly interpolate along frequency
        return (lpwr * (hl[0] - freq) + hpwr * (freq - ll[0])) / (hl[0] - ll[0])
        
    def get_rf_power(self, frequency, sa_output):
        #rejecting uncalibrated frequency
        #first test frequency range
        if frequency < self.lower_freq_bound or frequency >self.upper_freq_bound:
            raise FrequencyOutOfRangeError(self.lower_freq_bound, self.upper_freq_bound)
        
        lf = 0
        hf = 0
        for f in self.cali_freq:
            if f <= frequency and f > lf:
                lf = f
            if f >= frequency:
                hf = f
                break

        #check whether output is out of bound
        if sa_output < self.cali_pwr[lf][0][0] or sa_output > self.cali_pwr[lf][len(self.cali_pwr[lf])-1][0]:
            raise OutputOutOfRangeError(lf, self.cali_pwr[lf][0][0], 
                                   self.cali_pwr[lf][len(self.cali_pwr[lf])-1][0])        
        
        ll = [lf, 0, 0]
        lh = [lf, 0, 0]
        for op, pwr in self.cali_pwr[lf]:
            if op <= sa_output and op > ll[1]:
                ll[1] = op
                ll[2] = pwr
            if op >= sa_output:
                lh[1] = op
                lh[2] = pwr
                break
        
        if lf == hf:
            return self.interpolate(frequency, sa_output, ll, lh, ll, lh)
        
        #check whether output is out of bound
        if sa_output < self.cali_pwr[hf][0][0] or sa_output > self.cali_pwr[hf][len(self.cali_pwr[hf])-1][0]:
            raise OutputOutOfRangeError(hf, self.cali_pwr[hf][0][0], 
                                   self.cali_pwr[hf][len(self.cali_pwr[hf])-1][0])
                        
        hl = [hf, 0, 0]
        hh = [hf, 0, 0]
        for op, pwr in self.cali_pwr[hf]:
            if op <= sa_output and op > hl[1]:
                hl[1] = op
                hl[2] = pwr
            if op >= sa_output:
                hh[1] = op
                hh[2] = pwr
                break
        
        return self.interpolate(frequency, sa_output, ll, lh, hl, hh)

class OutOfRangeError(Exception):
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __str__(self):
        return 'lower bound: '+str(self.lower_bound)+', upper bound: '+str(self.upper_bound)

class FrequencyOutOfRangeError(OutOfRangeError):
    def __init__(self, lower_bound, upper_bound):
        OutOfRangeError.__init__(self, lower_bound, upper_bound)
        
class OutputOutOfRangeError(OutOfRangeError):
    def __init__(self, frequency, lower_bound, upper_bound):
        self.frequency = frequency
        OutOfRangeError.__init__(self, lower_bound, upper_bound)

    def __str__(self):
        return 'at frequency '+str(self.frequency)+' whose ' + OutOfRangeError.__str__(self)

if __name__ == '__main__':
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    
    sam = SACalibrationManager(pickle.load(open('10dBm_cali.data')))
    
    from instruments import E8257D
    from spectrum_analyzer import *
        
    rf = E8257D(address='rfgen2.circuitqed.com')
    lo = E8257D(address='rfgen1.circuitqed.com')
    sa = SpectrumAnalyzer(protocol='serial', port=2, query_sleep=0.2)

    lo.set_power(sa.lo_power)
    lo.set_output()
    rf.set_output()
    
    le = []
    for f0 in range(304, 777, 10):
        f = f0*1e7
        rf.set_frequency(f)
        lo.set_frequency(f+sa.lo_offset)
        for pwr in range(-50, 0, 10):
            rf.set_power(pwr)
            time.sleep(0.2)
            op = sa.get_avg_power()
            try:
                ppwr = sam.get_rf_power(f, op)
                err = abs(ppwr-pwr)
                print 'RF frequency: '+str(f)+', SA output: '+str(op)            
                print 'Measured: '+str(pwr)+', Predicted: '+str(ppwr)
                print 'Difference: '+str(err)
                print
                if (err > 0.5):
                    le.append([f, op, pwr, ppwr, err])
            except OutOfRangeError as e:
                print 'frequency: '+str(f)+', sa output: '+str(op)+', rf power: '+str(pwr)
                print e
            
    le = np.array(le)
    le = le.transpose()
    plt.scatter(le[0], le[1])
    plt.show()