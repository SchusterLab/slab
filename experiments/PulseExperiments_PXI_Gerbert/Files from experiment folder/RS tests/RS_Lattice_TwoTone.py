# -*- coding: utf-8 -*-
"""
Created on Sun Aug 04 2015

@author: Nate E
"""

from slab import *
from slab.instruments import *
from slab.datamanagement import SlabFile
from matplotlib.pyplot import *
from slab.instruments import *
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


dac = AD5780_serial()
time.sleep(1)

from importlib import reload
import RhodeSchwarz
reload(RhodeSchwarz)
from RhodeSchwarz import RhodeSchwarz
from RhodeSchwarz import ZVB8
try:
    rs = ZVB8(address='192.168.14.231', reset = True)
    rs.get_id()
except:
    print("Rohde-Schwarz Not Activated!")
    raise

# drive = im['RF3']
print ('Deviced Connected')
expt_path = os.getcwd() + '\data'
path = "C:\\210801 - PHMIV3_56 - BF4 cooldown 4"
expt_path = path + '\data'


def flux_map_onetone(qubitInd, fluxInd):

    # ###d

    #map resonator index to qubit indicies
    Res_index = [0, 2, 4, 6, 7, 5, 3, 1][qubitInd] # new mapping helps with maximizing res. spatial symmetry

    prefix = "flux_map-automate_"+ str(datetime.date.today()) + "_q" + str(qubitInd) + "_f" + str(fluxInd)

    numpts = [16]*8 #nb of points

    #flux matrix measurement
    diag_offset = [0,0,0,0,0,0,0,0] # flux matrix offset
    # diag_offset = [0, 0.75, 1.200, 1.650, 1.395, -0.75, 1.125, 1.833]  # flux matrix offset
    diag_range = [2.000]*8
    # diag_range = [0.050,0.050,0.050,0.050,0.050,0.050,0.050,0.040]
    offdiag_range = [0.120,0.120,0.120,0.120,0.120,0.120,0.120,0.120]


    # flux bias box rampspeed
    #ie how fast move in voltage, these two integer values in C code control how fast write to register
    # parallelramp() with rampspeed = 1, step = 1 ramps at 2.5mV/sec , roughly 0.5% flux quanta / second
    rampspeed = 1
    step = 5

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

    read_power = [-25]*8

    ## Cooldown 4 freq
    read_freq_center = [6.180e9, 6.290e9, 6.405e9, 6.510e9, 6.610e9, 6.735e9, 6.835e9, 6.955e9]
    bw=40e6
    read_freq_start = []
    read_freq_stop = []

    for ii,elem in enumerate(read_freq_center):
        read_freq_start.append(elem - bw/2)
        read_freq_stop.append(elem + bw / 2)


    drive_power = [-10,-10,-10,-10,-10,-10,-10,-25]

    drive_freq_start = [4.0e9,4.0e9,4.0e9,4.0e9,4.0e9,4.0e9,4.0e9,4.75e9]
    drive_freq_stop = [6.05e9,6.05e9,6.05e9,6.05e9,6.05e9,6.05e9,6.05e9,5.25e9]

    # single tone
    sweep_pts = 1001
    ifbw = [1000]*8

    avgtime = 6

    # two tone
    sweep_pts2 = 1001
    avgtime2 = 310
    ifbw2 = [500]*8
    delay = 0

    time.sleep(0.3)
    rs.reset()
    time.sleep(0.3)
    trans_settings = rs.trans_default_settings()
    time.sleep(0.3)
    trans_settings['start_freq'] = read_freq_center[Res_index] - bw / 2
    trans_settings['stop_freq'] = read_freq_center[Res_index] + bw / 2
    trans_settings['avg_time'] = avgtime
    trans_settings['ifBW'] = ifbw[Res_index]
    trans_settings['RFpower'] = read_power[Res_index]
    trans_settings['freq_points'] = sweep_pts
    print(trans_settings)

    vary_param = "flux"
    print ("Swept Parameter: %s" % (vary_param))
    fname = get_next_filename(expt_path, prefix, suffix='.h5')
    print (fname)
    fname = os.path.join(expt_path, fname)

    with SlabFile(fname) as f:
        print ("ISD:LJFJF")
        f.save_settings(trans_settings)


    for ii,pt_target in enumerate(all_f_vary_pts):

        print ('flux target =', pt_target)
        pt = (pt_target)

        print("Driving DCBOX at %s"%(pt_target) + " V")
        dac.parallelramp(pt_target,stepsize = step,steptime = rampspeed)

        time.sleep(2)

        print( "Set up for a Single Tone Sweep")

        time.sleep(0.2)
        rs.reset()
        time.sleep(0.3)
        data = rs.trans_meas(trans_settings)
        # data = nwa.take_one_in_mag_phase()

        mags = np.array([float(i) for i in (data['mag'].split(','))])
        phases = np.array([float(i) for i in (data['phase'].split(','))])
        fpoints = np.array([float(i) for i in (((data['xaxis']).tolist()).split(','))]) / 1e9
        # fpoints = data[0]
        # mags = data[1]
        # phases = data[2]

        print ("finished downloading")

        freq_readout = fpoints[argmin(mags)]

        print ("Set up for Two Tone Measurement, ReadFreq =", freq_readout)

        time.sleep(0.3)
        rs.reset()
        time.sleep(0.3)
        spec_settings = rs.spec_default_settings()
        spec_settings['CAVport'] = 1
        spec_settings['RFport'] = 3
        spec_settings['Mport'] = 2

        spec_settings['start_freq'] = drive_freq_start[qubitInd]
        spec_settings['stop_freq'] = drive_freq_stop[qubitInd]
        spec_settings['RFpower'] = drive_power[qubitInd]
        spec_settings['CAVfreq'] = freq_readout*1e9
        spec_settings['CAVpower'] = read_power[Res_index]
        spec_settings['avg_time'] = avgtime2
        spec_settings['ifBW'] = ifbw2[qubitInd]
        spec_settings['freq_points'] = sweep_pts2
        spec_settings['measurement'] = 'S21'
        #     print(spec_settings)

        # nwa.set_ifbw(ifbw2[Res_index])
        # nwa.set_sweep_points(sweep_pts2)
        # nwa.setup_take(averages_state=True)
        # nwa.set_averages_and_group_count(avgs2, True)
        # nwa.setup_two_tone_measurement(read_frequency=freq_readout,
        #                                read_power=read_power[qubitInd],
        #                                probe_start=drive_freq_start[qubitInd],
        #                                probe_stop=drive_freq_stop[qubitInd],
        #                                probe_power=drive_power[qubitInd], two_tone=1)

        data = rs.spec_meas(spec_settings)
        # data = nwa.take_one_in_mag_phase()

        mags2 = np.array([float(i) for i in (data['mag'].split(','))])
        phases2 = np.array([float(i) for i in (data['phase'].split(','))])
        fpoints2 = np.array([float(i) for i in (((data['xaxis']).tolist()).split(','))]) / 1e9

        # fpoints2 = data[0]
        # mags2 = data[1]
        # phases2 = data[2]
        print ("finished downloading")

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

    print (fname)

for ii in [7]:
    for jj in [ii]:
        flux_map_onetone(qubitInd=ii, fluxInd=jj)

print ("Returning DCBOX to 0V")
rampspeed = 1
step = 8
dac.parallelramp([0,0,0,0,0,0,0,0],stepsize = step,steptime = rampspeed)
time.sleep(1)