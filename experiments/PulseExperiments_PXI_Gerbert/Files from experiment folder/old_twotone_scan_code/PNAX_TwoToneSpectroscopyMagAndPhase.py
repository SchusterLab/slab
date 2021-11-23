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
im = InstrumentManager()

# dac = AD5780(address='192.168.14.158')
# print(dac.get_id())
#print dac.ramp(1, 2.70, 0.02)

dcflux2 = im['YOKO4']
print(dcflux2.get_id())
dcflux2.set_mode('current')
dcflux2.set_range(0.10)
dcflux2.set_output(True)
#
# dcfluxoffset = im['YOKO4']
# dcfluxoffset.set_mode('current')
# dcfluxoffset.set_range(0.10)
# dcfluxoffset.set_output(True)
# dcfluxoffset.ramp_current(0*10**-3, 0.005)

isPNAX = 1  # has to be PNAX
if isPNAX:
    nwa = im['PNAX2']

print (nwa.get_id())
# drive = im['RF3']
print ('Deviced Connected')
expt_path = os.getcwd() + '\data'

def flux_map_onetone(qubitInd, fluxInd):

    # ###
    isTakeData = 1

    #map resonator index to qubit indicies
    Res_index = [0,4,1,5,2,6,3,7][qubitInd] # previous mapping that maximizes freq separation
    Res_index = [0, 2, 4, 6, 7, 5, 3, 1][qubitInd] # new mapping helps with maximizing res. spatial symmetry


    prefix = "flux_map-automate_"+ str(datetime.date.today()) + "_q" + str(qubitInd) + "_f" + str(fluxInd)

    numpts = [11]*8
#    numpts = [1,1,1,1]

    # in volts for SRS -20 to 20 range

    # value for each individual flux
    diag_offset =  [-1.5e-3]*8
    diag_range = [0.5e-3]*8

    #diag_range = [1e-3] * 8
    #diag_range = [0,0,0,0]

    rampspeed = 0.005
    offdiag_range = [1e-3]*8

    #offdiag_range = [0,0,0,0]

    # finally fixed 02-20-17
    all_f_vary_pts_T = []

    for ii in range(8):
        if ii == fluxInd:
            if ii == qubitInd:
                frange = diag_range[ii]
                all_f_vary_pts_T.append(linspace(-frange + diag_offset[ii], frange + diag_offset[ii], numpts[Res_index]))
            else:
                frange = offdiag_range[ii]
                #all_f_vary_pts_T.append(linspace(-frange + 0, frange + 0, numpts[Res_index]))
                all_f_vary_pts_T.append(linspace(-frange + diag_offset[ii], frange + diag_offset[ii], numpts[Res_index]))
        else:
            all_f_vary_pts_T.append(linspace(0, 0, numpts[Res_index]))
            # all_f_vary_pts_T.append(linspace(diag_offset[ii], diag_offset[ii], numpts[Res_index]))
            # if ii == qubitInd:
            #     all_f_vary_pts_T.append(linspace(diag_offset[ii], diag_offset[ii], numpts[Res_index]))
            # else:
            #     all_f_vary_pts_T.append(linspace(0, 0, numpts[Res_index]))

    all_f_vary_pts = array(all_f_vary_pts_T).transpose()


    read_power = [ -57.0]*8 #45


    read_freq_center = [6.007e9,6.127e9,6.2395e9,6.313e9,6.464e9,6.5675e9,6.68e9,6.7875e9]
    bw=10e6
    read_freq_start = []
    read_freq_stop = []
    for ii,elem in enumerate(read_freq_center):
        read_freq_start.append(elem - bw/2)
        read_freq_stop.append(elem + bw / 2)

    drive_power = [-20.0]*8 #30
    drive_freq_start = [3.25e9]*8
    drive_freq_stop = [4.75e9]*8


    # single tone
    sweep_pts = 1001
    ifbw = [200]*8

    avgs = 2# for the four res: 2,5

    # two tone
    sweep_pts2 = 4001 #8001
    ifbw2 = [100]*8

    avgs2 = 10 #10

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

            print( "Driving YOKO at (%.3f) mA" % (pt[fluxInd]*1000))
            dcflux2.ramp_current(pt[fluxInd], rampspeed)
            time.sleep(0.2)

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
                cleanphasedata = smooth(cleanphasedata,10)


                if ii == 0:
                    figure(figsize=(16, 16))
                    subplot(111, title='cleanphase', xlabel='Frequency (GHz)', ylabel='phase')
                    plt.scatter(fpoints,cleanphasedata)
                    plt.show()
                    #subplot(212, title='cleanphase', xlabel='Frequency (GHz)', ylabel='dphase')
                    #plt.scatter(fpoints, np.diff(cleanphasedata))
                freq_peak = fpoints[np.argmax(np.abs(np.diff(cleanphasedata[10:-10])))]
                print(freq_peak)
            else:
                ## TODO: Detecting min of readout hybridized state - unclear if this is right.  Can also use Max between res.
                freq_peak = fpoints[argmin(mags)]

                ## now try double resonator peak - readout on max peak height between resonators
                freq_peak = fpoints[argmax(mags)]
                if ii==0:
                    figure(figsize=(15,15))
                    subplot(111,title='magpts',xlabel = 'Freq (GHz)',ylabel = 'mags')
                    plt.scatter(fpoints,mags)
                    plt.show()
                print(freq_peak)

            print ("Set up for Two Tone Measurement, ReadFreq =", freq_peak)
            nwa.set_ifbw(ifbw2[Res_index])
            nwa.set_sweep_points(sweep_pts2)
            nwa.setup_take(averages_state=True)
            nwa.set_averages_and_group_count(avgs2, True)
            nwa.setup_two_tone_measurement(read_frequency=freq_peak,
                                           read_power=read_power[Res_index],
                                           probe_start=drive_freq_start[Res_index],
                                           probe_stop=drive_freq_stop[Res_index],
                                           probe_power=drive_power[Res_index], two_tone=1)

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
                f.append_pt('read_freq', freq_peak)

        # for jj in range(len(diag_offset)):
        #     srs.ramp_volt( 0.0, sweeprate=0.15, channel=(jj + 1) )
        # time.sleep(0.2)

        print (fname)

    print ("Returning YOKO to 0.0 mA")
    dcflux2.ramp_current(0.0, rampspeed)
    time.sleep(0.2)
    print("Returning DC coil to 0.0 mA")
    # dcfluxoffset.ramp_current(0.0 * 10 ** -3, 0.005)
    time.sleep(0.2)

######

# flux_map_onetone(qubitInd=0, fluxInd=6)
# flux_map_onetone(qubitInd=2, fluxInd=6)
# flux_map_onetone(qubitInd=5, fluxInd=6)
#flux_map_onetone(qubitInd=2, fluxInd=2)
for ii in [6]:
    flux_map_onetone(qubitInd=ii, fluxInd=ii)