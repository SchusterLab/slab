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
DCBOX = 0
YOKO = 1
if DCBOX == 1:
    dac = AD5780_serial()
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

isPNAX = 1  # has to be PNAX
if isPNAX:
    nwa = im['PNAX']

print (nwa.get_id())
# drive = im['RF3']
print ('Deviced Connected')
expt_path = os.getcwd() + '\data'
path = "C:\\210801 - PHMIV3_56 - BF4 cooldown 4"
expt_path = path + '\data'

def flux_map_onetone(qubitInd, fluxInd):

    # ###d
    isTakeData = 1

    #map resonator index to qubit indicies
    Res_index = [0, 2, 4, 6, 7, 5, 3, 1][qubitInd] # new mapping helps with maximizing res. spatial symmetry

    prefix = "flux_map-automate_"+ str(datetime.date.today()) + "_q" + str(qubitInd) + "_f" + str(fluxInd)

    numpts = [24]*8 #nb of points

    # Convert to volts assuming 50 ohms impedance

    #flux matrix measurement
    diag_offset = [0,0,0,0,0,0,0,0]
    diag_range = [40]*8
    offdiag_range = [0.005]*8
    #

    # Solenoid tuning speed - measure this w/ 50 ohm terminator first!

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
                # all_f_vary_pts_T.append(linspace(diag_offset[ii], diag_offset[ii], numpts[Res_index]))

    all_f_vary_pts = array(all_f_vary_pts_T).transpose()

    read_power = [-25]*8
    read_freq_center = [6.180e9, 6.290e9, 6.405e9, 6.510e9, 6.610e9, 6.735e9, 6.835e9,  6.955e9]
    bw=40e6
    read_freq_start = []
    read_freq_stop = []
    for ii,elem in enumerate(read_freq_center):
        read_freq_start.append(elem - bw/2)
        read_freq_stop.append(elem + bw / 2)

    drive_power = [-10,-10,-10,-10,-10,-10,-10,-20]

    drive_freq_start = [4.0e9]*8
    drive_freq_stop = [5.9e9]*8

    # single tone
    sweep_pts = 4001
    ifbw = [500]*8

    avgs = 2# for the four res: 2,5

    # two tone
    sweep_pts2 = 4001
    avgs2 = 2 #10
    ifbw2 = [500]*8
    delay = 0

    if isPNAX:
        print ("Configuring the PNAX")

        # turning off the pulses
        nwa.write("SENS:PULS0 0")
        nwa.write("SENS:PULS1 0")
        nwa.write("SENS:PULS2 0")
        nwa.write("SENS:PULS3 0")
        nwa.write("SENS:PULS4 0")
        # turning on the inverting
        nwa.write("SENS:PULS1:INV 1")
        nwa.write("SENS:PULS2:INV 1")
        nwa.write("SENS:PULS3:INV 1")
        nwa.write("SENS:PULS4:INV 1")

        nwa.set_timeout(10E3)
        nwa.clear_traces()
        nwa.setup_measurement("S21")

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
            print (nwa.get_settings())
            f.save_settings(nwa.get_settings())

        for ii,pt_target in enumerate(all_f_vary_pts):

            print ('flux target =', pt_target)
            pt = (pt_target)

            if YOKO == 1:
                print( "Driving YOKO at (%.3f) uA" % (pt[fluxInd]))
                yokorampuA = pt[fluxInd]
                dcflux2.ramp_current( yokorampuA * 10 ** -6, 1500*10**-9) # ramp current in mA


            # print("Driving DCBOX at %s"%(pt_target) + " V")
            # dac.parallelramp(pt_target,stepsize = step,steptime = rampspeed)
            # for jj,elem in enumerate(pt_target):
            #     dac.ramp3(jj+1,elem+0.0001,step=step,speed=rampspeed)
            #     time.sleep(5)
                #WHICH index do we use for yoko here, should be 3?
                #yokorampnA = int ((3.0157) * (pt[fluxInd] * 10**3))
                #dcfluxoffset.set_current(yokorampnA)
                #dcfluxoffset.ramp_current( yokorampnA* 10 ** -9, 0.005) # ramp current in mA
                #print("Ramp YOKO to yokorampnA* 10 ** -6")
            time.sleep(2)

            print( "Set up for a Single Tone Sweep")
            nwa.set_ifbw(ifbw[Res_index])
            nwa.set_sweep_points(sweep_pts)
            nwa.setup_take(averages_state=True)
            nwa.set_averages_and_group_count(avgs, True)
            nwa.setup_two_tone_measurement(read_frequency=dummy_freq, read_power=read_power[Res_index],
                                           probe_start=read_freq_start[Res_index],
                                           probe_stop=read_freq_stop[Res_index], probe_power=dummy_power,
                                           two_tone=0)

            data = nwa.take_one_in_mag_phase()

            fpoints = data[0]
            mags = data[1]
            phases = data[2]
            print ("finished downloading")
            #time.sleep(na.get_query_sleep())

            #choose to use magnitude or phase information from singletone spectroscopy
            phasedetect = 0



            if phasedetect == 1:

                mr,br = np.polyfit(fpoints[50:100],phases[50:100],1)
                cleanphasedata = ((phases - fpoints*mr+br)*(2*np.pi/360))%(2*np.pi)
                cleanphasedata = np.unwrap(cleanphasedata,np.pi-np.pi*.0001)

                def smooth(y, box_pts):
                    box = np.ones(box_pts) / box_pts
                    y_smooth = np.convolve(y, box, mode='same')
                    return y_smooth
                cleanphasedata = smooth(cleanphasedata,20)
                freq_peak = fpoints[np.argmin(np.diff(cleanphasedata[10:-10]))]

                # if ii == 0:
                #     figure(figsize=(16, 16))
                #     subplot(111, title='cleanphase', xlabel='Frequency (GHz)', ylabel='phase')
                #     plt.scatter(fpoints,cleanphasedata)
                #     plt.axvline(freq_peak)
                #     plt.show()


                    #subplot(212, title='cleanphase', xlabel='Frequency (GHz)', ylabel='dphase')
                    #plt.scatter(fpoints, np.diff(cleanphasedata))

                print(freq_peak)
            else:
                freq_readout = fpoints[argmin(mags)]
                # freq_min = fpoints[argmax(mags)]
                # freq_readout = (freq_max + freq_min)/2
                #freq_readout = (6.978e9)
                # if ii==0:
                #     figure(figsize=(15,15))
                #     subplot(111,title='magpts',xlabel = 'Freq (GHz)',ylabel = 'mags')
                #     plt.scatter(fpoints,mags)
                #     plt.axvline(freq_readout)
                #     plt.show()
                # print(freq_readout)

            print ("Set up for Two Tone Measurement, ReadFreq =", freq_readout)
            nwa.set_ifbw(ifbw2[Res_index])
            nwa.set_sweep_points(sweep_pts2)
            nwa.setup_take(averages_state=True)
            nwa.set_averages_and_group_count(avgs2, True)
            nwa.setup_two_tone_measurement(read_frequency=freq_readout,
                                           read_power=read_power[qubitInd],
                                           probe_start=drive_freq_start[qubitInd],
                                           probe_stop=drive_freq_stop[qubitInd],
                                           probe_power=drive_power[qubitInd], two_tone=1)

            data = nwa.take_one_in_mag_phase()

            fpoints2 = data[0]
            mags2 = data[1]
            phases2 = data[2]
            print ("finished downloading")
            # time.sleep(na.get_query_sleep())

            #freq_peak_q = fpoints2[argmax(mags2)]

            #print "QubitFreq =", freq_peak_q

            with SlabFile(fname) as f:
                for ii in range(8):
                    f.append_pt((vary_param + str(ii) + '_pts'), pt_target[ii])
                    f.append_pt((vary_param + str(ii) + '_actual_pts'), pt[ii])
                f.append_line('fpts-r', fpoints)
                f.append_line('mags-r', mags)
                f.append_line('phases-r', phases)
                f.append_line('fpts-q', fpoints2)
                f.append_line('mags-q', mags2)
                f.append_line('phases-q', phases2)
                # f.append_line('read_freq',read_freqs)
                f.append_pt('read_power', read_power[Res_index])
                f.append_pt('probe_power', drive_power[Res_index])
                f.append_pt('read_freq', freq_readout)

        # for jj in range(len(diag_offset)):
        #     srs.ramp_volt( 0.0, sweeprate=0.15, channel=(jj + 1) )
        # time.sleep(0.2)

        print (fname)
######


# for ii in [0,1,2,3,4,5,6,7]:
#     flux_map_onetone(qubitInd=ii, fluxInd=ii)

print ("Returning DCBOX to vec V")
dcflux2.ramp_current(-37.6 * 10 ** -6, 1500*10**-9)
#dcflux2.ramp_current(0, 1500*10**-9)