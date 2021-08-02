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
if DCBOX == 1:

    # dcflux2 = im['dacbox']
    # dcflux2.initialize()

    #from slab.instruments.AD5780DAC import AD5780
    #dac = AD5780(name='dacbox',address = 'COM3',enabled = True,timeout=1)
    from DACInterface import AD5780_serial
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



# dcfluxoffset = im['YOKO4']
Q3flx = im['YOKO4']
Q3flx.set_mode('current')
Q3flx.set_range(0.0010)
Q3flx.set_output(True)


isPNAX = 1  # has to be PNAX
if isPNAX:
    nwa = im['PNAX2']

print (nwa.get_id())
# drive = im['RF3']
print ('Deviced Connected')
expt_path = os.getcwd() + '\data'

def flux_map_onetone(qubitInd):

    # ###d
    isTakeData = 1

    #map resonator index to qubit indicies
    Res_index = [0, 2, 4, 6, 7, 5, 3, 1][qubitInd] # new mapping helps with maximizing res. spatial symmetry

    prefix = "FluxMatrixTest_"+ str(datetime.date.today()) + "_q" + str(qubitInd)

    numpts = [3]*8 #nb of points

    # Convert to volts assuming 50 ohms impedance


    flux_array = [[0,0,0,0.0*10**-6,-0.27269654, -0.13036466,  0.00827724, -0.32537421]]
    #

    ## Interpolation between 5 flux vector points for independent Q4,5,6,7 control to tune on resonance
    # mvvec1 = [-0.39403048, -0.03488094, -0.05773415, -0.03942887]
    # mvvec2 = [-0.36166365, -0.21313627, -0.03027432, -0.00089514]
    # mvvec3 = [-0.37773313, -0.2300758,   0.06888742, -0.00925846]
    # mvvec4 = [-0.33084986, -0.17877626 , 0.13848821, -0.37491046]
    # mvvec0 = [0,0,0,0]
    # mvvec0 = np.concatenate(([0, 0, 0, 0], mvvec0))
    # mvvec1 = np.concatenate(([0, 0, 0, 0], mvvec1))
    # mvvec2 = np.concatenate(([0, 0, 0, 0], mvvec2))
    # mvvec3 = np.concatenate(([0, 0, 0, 0], mvvec3))
    # mvvec4 = np.concatenate(([0, 0, 0, 0], mvvec4))
    # mvveclist = [mvvec0, mvvec1, mvvec2, mvvec3, mvvec4]
    # from scipy.interpolate import interp1d
    # fluxvecinterpolator = interp1d(np.arange(len(mvveclist)), mvveclist, axis=0)
    # numpts = 11
    # flux_array = np.zeros((numpts * (len(mvveclist) - 1), len(mvvec0)))
    # for ii in range(len(mvveclist) - 1):
    #     sampler = np.linspace(ii, ii + 1, numpts)
    #     for jj, elem in enumerate(sampler):
    #         flux_array[jj + len(sampler) * (ii), :] = fluxvecinterpolator(elem)



    # Sampling npts between two flux vectors
    # mvvec1 = [-0.28528172, -0.12564481, -0.00233889, -0.32078319]
    # mvvec2 = [-0.25143269, -0.12264837,  0.00262074, -0.31739607]
    # mvvec1 = np.concatenate(([0, 0, 0, 0], mvvec1))
    # mvvec2 = np.concatenate(([0, 0, 0, 0], mvvec2))
    # mvveclist = [mvvec1, mvvec2]
    # from scipy.interpolate import interp1d
    # fluxvecinterpolator = interp1d(np.arange(len(mvveclist)), mvveclist, axis=0)
    # numpts = 11
    # flux_array = np.zeros((numpts * (len(mvveclist) - 1), len(mvvec1)))
    # for ii in range(len(mvveclist) - 1):
    #     sampler = np.linspace(ii, ii + 1, numpts)
    #     for jj, elem in enumerate(sampler):
    #         flux_array[jj + len(sampler) * (ii), :] = fluxvecinterpolator(elem)



    rampspeed = 0.005


    # flux bias box rampspeed
    #ie how fast move in voltage, these two integer values in C code control how fast write to register
    rampspeed = 2
    step = 2

    read_power = [-75,-75,-75,-75,-75,-75,-75,-75] #45
    read_power = [-70]*8
    read_freq_center = [6.19663e9, 6.3275e9, 6.4235e9, 6.545e9, 6.65254e9, 6.7725e9, 6.8710e9, 6.9925e9]
    bw=40e6
    read_freq_start = []
    read_freq_stop = []
    for ii,elem in enumerate(read_freq_center):
        read_freq_start.append(elem - bw/2)
        read_freq_stop.append(elem + bw / 2)

    drive_power = [-30,-30,-30,-30,-30,-25,-25,-40]
    drive_freq_start = [4.88e9]*8
    drive_freq_stop = [4.96e9]*8


    # single tone
    sweep_pts = 1001
    ifbw = [100]*8

    avgs = 1# for the four res: 2,5

    # two tone
    sweep_pts2 = 1001
    avgs2 = 2 #10
    ifbw2 = [50]*8
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

        for ii,pt_target in enumerate(flux_array):

            print ('flux target =', pt_target)
            pt = (pt_target)

            # if YOKO == 1:
            #     print( "Driving YOKO at (%.3f) mA" % (pt[fluxInd]*1000))
            #     dcflux2.ramp_current(pt[fluxInd], rampspeed)


            print("Driving DCBOX at %s"%(pt_target) + " V")
            for jj,elem in enumerate(pt_target):
                dac.ramp3(jj+1,elem+0.0001,step=step,speed=rampspeed)
                if jj == 3:
                    print("Driving Solneoid at %s" % (elem))
                    Q3flx.ramp_current(elem,sweeprate = 500*10**-9,)
                time.sleep(1)
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

    print ("Returning DCBOX to 0.0V")
    for ii in range(8):
        dac.ramp3(ii+1,0.001,step=step,speed=rampspeed)
        time.sleep(0.5)
    # print("Return YOKO to 0.0 mA")
    # Q3flx.ramp_current(0.00000, sweeprate=500 * 10 ** -9, )
    time.sleep(2)
    dac.init()
    time.sleep(2)
    time.sleep(0.2)
    #dcfluxoffset.ramp_current(0.0 * 10 ** -3, 0.005)
    time.sleep(0.2)

######


for ii in [4,5,6,7]:
    # for jj in [3,4,5,6,7]:
    flux_map_onetone(qubitInd=ii)