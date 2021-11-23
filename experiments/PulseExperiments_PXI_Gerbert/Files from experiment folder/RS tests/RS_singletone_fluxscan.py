# -*- coding: utf-8 -*-
"""
Created on Sun Aug 04 2015

@author: Nate E
"""

from slab import *
from slab.datamanagement import SlabFile
from numpy import *
import os
import datetime
import os.path



from slab.instruments.PNAX import N5242A
# from slab.instruments.AD5780 import AD5780

from slab.instruments import InstrumentManager
from slab.instruments import Alazar, AlazarConfig
from slab.instruments import InstrumentManager

from slab.instruments.nwa import E5071
#from liveplot import LivePlotClient
#from slab.instruments import N5242A
from slab.dsfit import fithanger_new_withQc
from matplotlib.pyplot import *
from slab.instruments import *
DCBOX = 1
YOKO = 0
isRS = 1
if DCBOX == 1:

# dcflux2 = im['dacbox']
# dcflux2.initialize()

#from slab.instruments.AD5780DAC import AD5780
#dac = AD5780(name='dacbox',address = 'COM3',enabled = True,timeout=1)
    dac = AD5780_serial()
    time.sleep(2)
    dac.init()
    time.sleep(2)
    dac.init()
    time.sleep(2)

# print(dac.get_id())
#print dac.ramp(1, 2.70, 0.02)
#dcflux2 = im['YOKO4']

im = InstrumentManager()
if YOKO == 1:
    dcflux2 = im['YOKO4']
    print(dcflux2.get_id())
    dcflux2.set_mode('current')
    dcflux2.set_range(0.001)
    dcflux2.set_output(True)

im = InstrumentManager()
rs = im['RS']
print (rs.get_id())
# drive = im['RF3']
print ('Deviced Connected')
expt_path = os.getcwd() + '\data'
path = "C:\\210412 - PHMIV3_56 - BF4 cooldown 2"
expt_path = path + '\data'

def flux_map_onetone(qubitInd, fluxInd):

    # ###d
    isTakeData = 1

    #map resonator index to qubit indicies
    Res_index = [0, 2, 4, 6, 7, 5, 3, 1][qubitInd] # new mapping helps with maximizing res. spatial symmetry

    prefix = "flux_map-automate_"+ str(datetime.date.today()) + "_q" + str(qubitInd) + "_f" + str(fluxInd)

    numpts = [21]*8 #nb of points

    # Convert to volts assuming 50 ohms impedance

    #flux matrix measurement
    diag_offset = [0,0.6,1.4,-1.6,1.6,-0.8,-1.2,-2.040] # flux matrix offset
    diag_offset = [0.0]*8
    diag_range = [2.000]*8
    #diag_range = [0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.040]
    offdiag_range = [0.120,0.120,0.120,0.120,0.120,0.120,0.120,0.120]
    offdiag_range = [0.000]*8

    # test frequency position of qubits using inversion matrix
    # diag_offset = [ 0.06611462, -0.1974917 ,  0.4723145 , -1.29620915,  0.88503951, -0.84919053,  0.95141723,  0.09092804]
    # diag_range = [0.002]*8
    # offdiag_range = [0.002]*8


    # flux bias box rampspeed
    #ie how fast move in voltage, these two integer values in C code control how fast write to register
    # parallelramp() with rampspeed = 1, step = 1 ramps at 5mV/sec , roughly 1% flux quanta / second
    rampspeed = 1
    step = 1

    all_f_vary_pts_T = []

    for ii in range(8):
        if ii == fluxInd:
            if ii == qubitInd:
                frange = diag_range[ii]
                all_f_vary_pts_T.append(linspace(-frange + diag_offset[ii], frange + diag_offset[ii], numpts[Res_index]))
            else:
                frange = offdiag_range[ii]
                all_f_vary_pts_T.append(linspace(-frange + 0, frange + 0, numpts[Res_index]))
        else:
            if ii == qubitInd:
                all_f_vary_pts_T.append(linspace(diag_offset[ii], diag_offset[ii], numpts[Res_index]))
            else:
                all_f_vary_pts_T.append(linspace(0, 0, numpts[Res_index]))
                #all_f_vary_pts_T.append(linspace(diag_offset[ii], diag_offset[ii], numpts[Res_index]))

    all_f_vary_pts = array(all_f_vary_pts_T).transpose()



    #read_power = [-65,-65,-65,-65,-65,-65,-65,-65] #45
    read_power = [-63]*8
    read_freq_center = [6.175e9, 6.285e9, 6.4e9, 6.505e9, 6.615e9, 6.725e9, 6.835e9, 6.96e9]
    bw=50e6
    read_freq_start = []
    read_freq_stop = []
    for ii,elem in enumerate(read_freq_center):
        read_freq_start.append(elem - bw/2)
        read_freq_stop.append(elem + bw / 2)


    drive_power = [0,0,0,0,0,0,0,-15]
    drive_power = [-5,-5,-5,-5,-5,-5,-5,-20]

    drive_freq_start = [4.0e9,4.0e9,4.0e9,4.0e9,4.0e9,4.0e9,4.0e9,4.0e9]
    drive_freq_stop = [6.05e9,6.05e9,6.05e9,6.05e9,6.05e9,6.05e9,6.05e9,6.05e9]

    # flux crosstalk matrix drive frequencies
    # drive_freq_start = [4.80e9,4.7e9,4.7e9,4.87e9,4.7e9,4.9e9,5.1e9,4.8e9]
    # drive_freq_stop = [5.40e9,5.3e9,5.3e9,5.47e9,5.3e9,5.65e9,5.7e9,5.2e9]

    # single tone
    sweep_pts = 4001
    ifbw = [1000]*8

    avgs = 5# for the four res: 2,5

    # two tone
    sweep_pts2 = 8001
    avgs2 = 5 #10
    ifbw2 = [500]*8
    delay = 0

    if isRS:
        print ("Configuring the RhodeSchwarz")
        #
        # rs.set_output(state = False)
        # rs.query('*RST')
        # rs.set_power('-80',channel = 1)
        # rs.set_output(state =  True)


        rs.set_timeout(10E3)
        #rs.clear_traces()
        rs.setup_measurement("S21")

        dummy_freq = 5.0e9
        dummy_power = -50

    if isTakeData:

        vary_param = "flux"
        print ("Swept Parameter: %s" % (vary_param))
        fname = get_next_filename(expt_path, prefix, suffix='.h5')
        print (fname)
        fname = os.path.join(expt_path, fname)

        with SlabFile(fname) as f:
            print ("ISD:LJFJF")
            print (rs.get_settings())
            f.save_settings(rs.get_settings())

        for ii,pt_target in enumerate(all_f_vary_pts):

            print ('flux target =', pt_target)
            pt = (pt_target)

            print("Driving DCBOX at %s"%(pt_target) + " V")
            dac.parallelramp(pt_target,stepsize = step,steptime = rampspeed)
            time.sleep(2)

            print( "Set up for a Single Tone Sweep")
            #rs.configure(start=None, stop=None, center=None, span=None,power=None, ifbw=None, sweep_points=None, averages=None)
            rs.set_power(power=read_power[Res_index], channel=1, port=1, state=1)
            #rs.set_format('MLOG')
            rs.set_center_frequency(read_freq_center[Res_index])
            rs.set_span(bw)
            rs.set_ifbw(ifbw[Res_index])
            rs.set_sweep_points(sweep_pts)
            rs.setup_take(averages_state=True)
            rs.set_averages_and_group_count(avgs, True)
            time.sleep(22)
            data = rs.read_data()

            fpoints = data[0]
            mags = data[1]
            phases = data[2]
            print ("finished downloading")
            #time.sleep(na.get_query_sleep())

            #choose to use magnitude or phase information from singletone spectroscopy
            phasedetect = 0

            with SlabFile(fname) as f:
                for ii in range(8):
                    f.append_pt((vary_param + str(ii) + '_pts'), pt_target[ii])
                    f.append_pt((vary_param + str(ii) + '_actual_pts'), pt[ii])
                f.append_line('fpts-r', fpoints)
                f.append_line('mags-r', mags)
                f.append_line('phases-r', phases)
                # f.append_line('read_freq',read_freqs)
                f.append_pt('read_power', read_power[Res_index])
                f.append_pt('probe_power', drive_power[Res_index])


        print (fname)
######


for ii in [0,1,2,3,4,5,6,7]:
    for jj in [0,1,2,3,4,5,6,7]:
       if (ii == jj):
            flux_map_onetone(qubitInd=ii, fluxInd=jj)

print ("Returning DCBOX to 0.0V")
dac.parallelramp([0,0,0,0,0,0,0,0],stepsize = 2, steptime = 1)
time.sleep(1)