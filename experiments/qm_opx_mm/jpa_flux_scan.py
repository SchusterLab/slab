# -*- coding: utf-8 -*-
"""
Created on Sun Aug 04 2015

@author: Nelson
"""

from slab import *
from slab.datamanagement import SlabFile
from slab.instruments import InstrumentManager
# from slab.instruments import Instrument
# from slab.instruments import voltsource, E5071
from slab.instruments import *
from numpy import *

# from liveplot import LivePlotClient

im=InstrumentManager()
#lp = LivePlotClient()

# vds = im['YOKO4']
# nwa = E5071(address="192.168.14.236")
nwa = im['NWA']
vds = im['YOKO1']

# nwa = E5071_USB('USB0::0x0957::0x0D09::MY46110978::0::INSTR')

# vds = voltsource.YokogawaGS200(address="192.168.14.149")
# sc1 = SignalCore.SignalCore(name="SignalCore",address="10001E48")
# vds = im['YOKO2']
# nwa = im['NWA']
#drive = im['RF3']
import os
path = os.getcwd()
expt_path = os.path.join(path, "data/")
# set_electrical_delay(self, electrical_delay, channel=1)

def nwa_flux_power_scan():
    prefix="nwa_flux_power_scan"
    #print "Saved script as: %s" % save_script(expt_path,prefix)
    fname=get_next_filename(expt_path,prefix,suffix='.h5')
    print(fname)
    fname=os.path.join(expt_path,fname)

    ppts=linspace(-15,-35,11)
    fluxpts=linspace(-0.5,0.5,101)
    nwa.set_remote_state()

    #lp.clear()
    for power in ppts:

        nwa.configure(center=6.126e9,span=300.0e6,power=power,ifbw=1000,sweep_pts=1001,avgs=1,remote=True)
        nwa.set_electrical_delay(87.4e-9)
        nwa.set_phase_offset(0)
        nwa.set_measure(mode='S21', channel=1)
        mags = []
        phases = []
        #lp.clear()

        for flux in fluxpts:
            dcflux.ramp_volt(flux)
            data=nwa.take_one()

            # lp.plot_xy('Current Trace',data[0],data[1])
            #lp.append_z('mags',data[1],start_step=((fluxpts[0],fluxpts[1]-fluxpts[0]),(data[0][0],data[0][1]-data[0][0])))
            #lp.append_z('phases',data[2],start_step=((fluxpts[0],fluxpts[1]-fluxpts[0]),(data[0][0],data[0][1]-data[0][0])))

            with SlabFile(fname) as f:
                f.append_pt('power',power)
                f.append_line('fpts',data[0])
                f.append_line('fluxpts',fluxpts)
                f.append_line('mags',data[1])
                f.append_line('phases',data[2])

def nwa_freq_power_scan():
    prefix="nwa_freq_power_scan"
    #print "Saved script as: %s" % save_script(expt_path,prefix)
    fname=get_next_filename(expt_path,prefix,suffix='.h5')
    print(fname)
    fname=os.path.join(expt_path,fname)

    fpts=linspace(5.05e9,5.09e9,161)
    ppts=linspace(-40,-50,51)
    nwa.set_remote_state()
    nwa.configure(center=5079400000,span=0,power=-60.0,ifbw=500,sweep_pts=2,avgs=1,remote=True)


   # lp.clear()
    for power in ppts:
        im.RF1.set_power(power)
        mags = []
        phases = []

        for freq in fpts:
            im.RF1.set_frequency(freq)

            data=nwa.take_one()

            # lp.plot_xy('Current Trace',data[0],data[1])
            # lp.append_z('mags',data[1],start_step=((fpts[0],fpts[1]-fpts[0]),(data[0][0],data[0][1]-data[0][0])))
            # lp.append_z('phases',data[2],start_step=((fpts[0],fpts[1]-fpts[0]),(data[0][0],data[0][1]-data[0][0])))

            mags.append(data[1][0])
            phases.append(data[2][0])

        #lp.append_z('mags',mags,start_step=((ppts[0],ppts[1]-ppts[0]),(fpts[0],fpts[1]-fpts[0])))
        #lp.append_z('phases',phases,start_step=((ppts[0],ppts[1]-ppts[0]),(fpts[0],fpts[1]-fpts[0])))

        with SlabFile(fname) as f:
            f.append_pt('power',power)
            f.append_line('fpts',fpts)
            f.append_line('mags',mags)
            f.append_line('phases',phases)

def nwa_probe_flux_scan():

    #initial NWA configuration
    nwa.set_remote_state()
    nwa.configure(center=5.24e9,span=0.0e6,power=-30.0,ifbw=50,sweep_pts=2,avgs=1,remote=True)

    #parameter sweep arrays
    flux_pts = linspace(3.75,-4.15,401)

    read_pts = load(r'S:\_Data\151229 - Al Tunable Coupler\data\read_frequencies.npy')

    probe_span = 20e6

    if len(read_pts)!=len(flux_pts):
        raise NameError("Length of read frequency array does not match length of flux array")

    prefix = "nwa_probe_flux_scan"
    fname=get_next_filename(expt_path,prefix,suffix='.h5')
    print(fname)
    fname=os.path.join(expt_path,fname)

    with SlabFile(fname) as f:
        f.save_settings(nwa.get_settings())

    dcflux.set_output(True)
    # im.RF2
    # im.RF2.set_output(True)
    # im.RF2.set_power(-45.0)
    drive.set_output(True)
    drive.set_power(-20.0)

   # lp.clear()
    for flux, read in zip(flux_pts,read_pts):

        #dcflux.set_volt(flux)
        dcflux.ramp_volt(flux, sweeprate=0.5)
        nwa.set_center_frequency(read)

        probe_guess = 3.983e9
        probe_pts = linspace(probe_guess-probe_span/2.0,probe_guess+probe_span/2.0,101)

        print('Flux: %s V' %(dcflux.get_volt()))

        flux_line = []
        probe_freq_line = []
        read_freq_line = []
        mag_line =[]
        phase_line = []

        for freq in probe_pts:

            #im.RF2.set_frequency(freq)
            drive.set_frequency(freq)
            data=nwa.take_one()
            flux_line.append(flux)
            probe_freq_line.append(freq)
            read_freq_line.append(data[0][0])
            mag_line.append(data[1][0])
            phase_line.append(data[2][0])

        with SlabFile(fname) as f:
            f.append_line('flux',flux_line)
            f.append_line('probe_freq',probe_freq_line)
            f.append_line('read_freq',read_freq_line)
            f.append_line('mags',mag_line)
            f.append_line('phases',phase_line)

    #im.RF2.set_output(False)
    drive.set_output(False)
    nwa.configure(span=100.0e6,ifbw=1000,sweep_pts=1601,avgs=1,remote=False)
    print(fname)

def nwa_flux_scan_driving_qubit():

    #initial NWA configuration
    nwa.set_remote_state()
    nwa.configure(center=6.44e9,span=60.0e6,power=-40.0,ifbw=200,sweep_pts=1000,avgs=1,remote=True)

    #parameter sweep arrays
    flux_pts = linspace(4.85,-2.33,301)

    drive_pts = load(r'S:\_Data\151121 - Yao Lumped Tunable Coupler with more filters\data\drive_frequencies2.npy')

    if len(drive_pts)!=len(flux_pts):
        raise NameError("Length of drive frequency array does not match length of flux array")

    prefix = "nwa_flux_scan_driving_qubit"
    fname=get_next_filename(expt_path,prefix,suffix='.h5')
    print(fname)
    fname=os.path.join(expt_path,fname)

    with SlabFile(fname) as f:
        f.save_settings(nwa.get_settings())

    dcflux.set_output(True)
    im.RF1
    im.RF1.set_output(True)
    im.RF1.set_power(-45.0)
    # drive.set_output(True)
    # drive.set_power(-25.0)

   # lp.clear()
    for flux, drive in zip(flux_pts,drive_pts):

        #dcflux.set_volt(flux)
        dcflux.ramp_volt(flux, sweeprate=0.5)

        print('Flux: %s V' %(dcflux.get_volt()))
        im.RF1.set_frequency(drive)
        data=nwa.take_one()

        with SlabFile(fname) as f:
            f.append_pt(('flux_pts'),flux)
            f.append_line('fpts',data[0])
            f.append_line('mags',data[1])
            f.append_line('phases',data[2])

    im.RF1.set_output(False)
    #drive.set_output(False)
    print(fname)

def nwa_general_scan():
    #set your swept parameter here
    vary_param = "flux"
    # vary_param = "read_power"
    # dc_current = 0.00025
    print("Swept Parameter: %s" %(vary_param))

    #initial NWA configuration
    freq_start = 3e9
    freq_stop = 8.5e9
    power = -50
    ifbw = 1000
    avgs = 4
    sweep_pts = 551
    elec_delay = 76e-9
    phase_offset = 160
    
    print( "Configuring the NWA")
    nwa.set_remote_state()
    # nwa.configure(power=power, ifbw=ifbw, sweep_points=sweep_pts, averages=avgs, remote=True)
    nwa.configure(start=freq_start,stop=freq_stop,power=power,ifbw=ifbw,sweep_points=sweep_pts,averages=avgs,remote=True)
    nwa.set_average_state(True)
    nwa.set_electrical_delay(elec_delay)
    nwa.set_phase_offset(phase_offset)
    nwa.set_measure(mode='S21', channel=1)
    # nwa.set_format(trace_format='PHASE')
    # nwa.set_format(trace_format='MLOG')
    #parameter sweep arrays
    if vary_param == "probe_freq":
        vary_pts = linspace(3.5e9,4.2e9,701)
        # im.RF1
        # im.RF1.set_output(True)
        # im.RF1.set_power(-5.0)
        drive.set_output(True)
        drive.set_power(-35.0)
    elif vary_param == "probe_power":
        vary_pts = linspace(-60.0,0.0,11)
        #im.RF1.set_output(True)
        drive.set_output(True)
    elif vary_param == "flux":
        vary_pts = linspace(-10e-3, 10e0-3, 101)
        # vary_pts = vary_pts[::-1]
    elif vary_param == "read_power":
        vary_pts = linspace(10.0,-50.0,61)
        vds.ramp_current(dc_current)
    else:
        raise NameError("Invalid vary parameter")

    prefix = "nwa_"+vary_param+"_scan"
    fname=get_next_filename(expt_path,prefix,suffix='.h5')
    print(fname)
    fname=os.path.join(expt_path,fname)

    # print(nwa.get_id())

    #lp.clear()
    with SlabFile(fname, 'a') as f:
        print(nwa.get_settings())
        f.save_settings(nwa.get_settings())

    if vary_param == "flux":
        # vds.set_current(0)
        # vds.set_output(.2)
        vds.set_output(True)
    else:
        pass

    for var in vary_pts:

        print(var)

        if vary_param == "probe_freq":
            #im.RF1.set_frequency(var)
            drive.set_frequency(var)
        elif vary_param == "probe_power":
            #im.RF1.set_power(var)
            drive.set_power(var)
        elif vary_param == "flux":
            #dcflux.set_volt(var)
            vds.ramp_current(var)
        elif vary_param == "read_power":
            nwa.set_power(var)
        else:
            raise NameError("Invalid vary parameter")

        data=nwa.take_one()
        #data=nwa.take_one_averaged_trace()

        with SlabFile(fname, 'a') as f:
            f.append_pt((vary_param+'_pts'),var)
            f.append_pt(('read_power'),power)
            f.append_pt(('flux'), dc_current)
            f.append_pt(('ifbw'),ifbw)
            f.append_line('fpts',data[0])
            f.append_line('mags',data[1])
            f.append_line('phases',data[2])
    # drive.set_output(False)
    print(fname)

def nwa_freq_scan():
    prefix="nwa_freq_scan"
    #print "Saved script as: %s" % save_script(expt_path,prefix)
    fname=get_next_filename(expt_path,prefix,suffix='.h5')
    print(fname)
    fname=os.path.join(expt_path,fname)

    fpts=linspace(6850000000,7050000000,5)
    fpts = linspace(6.85e9, 7.05e9, 5)
    # nwa.set_remote_state()
    # nwa.configure(center=7e9,span=0,power=-60.0,ifbw=500,sweep_pts=2,avgs=1,remote=True)


   # lp.clear
    sc1.set_power(-5.5)

    for freq in fpts:
        sc1.set_frequency(freq)
        mags = []
        phases = []

        data=nwa.take_one()

        # lp.plot_xy('Current Trace',data[0],data[1])
        # lp.append_z('mags',data[1],start_step=((fpts[0],fpts[1]-fpts[0]),(data[0][0],data[0][1]-data[0][0])))
        # lp.append_z('phases',data[2],start_step=((fpts[0],fpts[1]-fpts[0]),(data[0][0],data[0][1]-data[0][0])))

        mags.append(data[1][0])
        phases.append(data[2][0])

    #lp.append_z('mags',mags,start_step=((ppts[0],ppts[1]-ppts[0]),(fpts[0],fpts[1]-fpts[0])))
    #lp.append_z('phases',phases,start_step=((ppts[0],ppts[1]-ppts[0]),(fpts[0],fpts[1]-fpts[0])))

    with SlabFile(fname, 'a') as f:
        f.append_pt('power',power)
        f.append_line('fpts',fpts)
        f.append_line('mags',mags)
        f.append_line('phases',phases)

nwa_general_scan()
# nwa_freq_scan()
# vds.set_current(0)
# sc1.cslose_device()
#nwa_freq_power_scan()
#nwa_flux_power_scan()
#nwa_probe_flux_scan()
#nwa_flux_scan_driving_qubit()