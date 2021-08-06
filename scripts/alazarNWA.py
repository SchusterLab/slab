# -*- coding: utf-8 -*-
"""
Created on Mon Feb 06 22:13:59 2012

@author: Phil
"""

from slab import *
from slab.instruments import InstrumentManager
from slab.plotting import FigureClient
from slab.script import ScriptPlotWin
from numpy import *
#from guiqwt.pyplot import *
import pickle
from slab.instruments import Alazar, AlazarConfig
#from scipy.signal import convolve,fftconvolve
from numpy import correlate,convolve
from scipy.signal import decimate    
import cProfile    
import pstats
    
def main():
    expt_path="S:\\_Data\\120425 - 50nm Nb 20um resonator with crystal on top of first 2\\spin echo\\"
    config="instruments.cfg"
    #datapath='S:\\_Data\\'
    prefix="test_dynamic"
    #datapath=make_datapath(expt_path,prefix)
    sweep_pts=1601
    ifbw=1e3
    
    Freqs=linspace(5.79125e9-2e6,5.79125e9+2e6,100)

    
    im=InstrumentManager(expt_path+config)    
    RF1=im['RF1']
    #RF2=im['RF2']
    RF2=im['LB1']
    #na=im['NWA']
    
    RF1.set_output(True)
    RF2.set_output(True)
    RF1.set_mod(True)
    RF2.set_mod(False)
    RF1.set_power(12)
    RF2.set_power(0)
    

    print("Configure NA")
#    na.set_default_state()
#    na.set_power(-20)
#    na.set_ifbw(ifbw)
#    na.set_span(0.)
#    na.set_sweep_points(1)

    IFfreq=1e6
    #na.set_center_frequency(2.5703e9)
    #RF2.set_frequency(2.5703e9+IFfreq)

    config={'clock_edge': 'rising', 'trigger_delay': 0, 'ch1_filter': False, 
            'ch1_enabled': True, 'samplesPerRecord': 50048, 'bufferCount': 1, 
            'trigger_edge1': 'rising', 'trigger_edge2': 'rising', 'ch2_range': 4, 
            'clock_source': 'internal', 'trigger_level2': 1.0, 'trigger_level1': 1.0,
            'ch2_coupling': 'DC', 'trigger_coupling': 'DC', 'ch2_filter': False, 
            'trigger_operation': 'or', 'ch1_coupling': 'AC', 'trigger_source2': 'disabled', 
            'trigger_source1': 'external', 'recordsPerBuffer': 1, 'sample_rate': 1000000, 
            'timeout': 5000, 'ch1_range': 4, 'ch2_enabled': False, 'recordsPerAcquisition': 1}
            
    print("Configuring card")
    scope_settings= AlazarConfig(config)
    
    card=Alazar(scope_settings)
    card.configure(scope_settings)

    print("go")
    print("Taking %d data points." % len(Freqs))
    print("|"+("  "*(len(Freqs)/10))+" |")
    print("|", end=' ')
    #figure(1)
    tpts,ch1_pts,ch2_pts=card.acquire_avg_data()    
    #fig1=FigureClient(xlabel='Time',ylabel='Amplitude',title='Scope')
    #fig2=FigureClient(xlabel='Time',ylabel='Amplitude',title='S21')
    #fig3=FigureClient(xlabel='Time',ylabel='Amplitude',title='S21')
    win = ScriptPlotWin(grid_x=2)
    scope_plot = win.add_linePlot(title="Scope")
    S21_plot_1 = win.add_linePlot(title="S21 1")
    S21_plot_2 = win.add_linePlot(title="S21 2")
    win.go()
    Amps=[]
    freqs=[]
    for ind,ff in enumerate(Freqs):
        if mod(ind,len(Freqs)/10.) ==0: print("-", end=' ')
        RF1.set_frequency(ff)
        RF2.set_frequency(ff+IFfreq)
#        na.set_center_frequency(ff)
        #print "freq=%f" % ff
        #time.sleep(1.15)
        tpts,ch1_pts,ch2_pts=card.acquire_avg_data()    
        #fig1.update_plot((tpts,ch1_pts))
        scope_plot.send((tpts, ch1_pts))
        dtpts,amp1pts,phi1pts,amp2pts,phi2pts=digital_homodyne(tpts,ch1_pts,ch2_pts,IFfreq,AmpPhase=True)
        #print "A1: %f, Phi1: %f, A2: %f, Phi2: %f" % card.heterodyne(tpts,ch1_pts,ch2_pts,IFfreq)      
        #A1= heterodyne(tpts,ch1_pts,ch2_pts,IFfreq)[0]
        freqs.append(ff/1e9)
        Amps.append(mean(amp1pts))
        #fig2.update_plot((dtpts,amp1pts))
        #fig3.update_plot((array(freqs),array(Amps)))
        S21_plot_1.send((dtpts, amp1pts))
        S21_plot_2.send((array(freqs),array(Amps)))
        
    print("|")
#    figure(2)
#    imshow(array(Amps))
#    show()

if __name__=="__main__":
#    cProfile.run("main()",'stats')
#    p=pstats.Stats('stats')
#    p.sort_stats('cumulative').print_stats(20)
    main()