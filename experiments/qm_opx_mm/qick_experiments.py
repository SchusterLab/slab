"""
Created Dec 2021

@author: Ankur Agrawal, Riju Banerjee and David Schuster, Schuster Lab
"""

"""Collating all the experiments classes in a single file to be called later. 
We will pass the class object in the actual run file and save the data there."""


from qick import *
from qick.helpers import gauss
from tqdm import tqdm_notebook as tqdm
import os

"""Time of flight experiment to calibrate the delay in the RF lines; helps to trigger the ADC when the signal actually arrives"""
class TimeofFlight(AveragerProgram):
    def initialize(self):
        cfg=self.cfg
        cfg["adc_lengths"]=[cfg["readout_length"]]*2     #add length of adc acquisition to config
        cfg["adc_freqs"]=[adcfreq(cfg["frequency"])]*2   #add frequency of adc ddc to config

        self.add_pulse(ch=cfg["res_ch"], name="measure", style="const", length=cfg["pulse_length"])  #add a constant pulse to the pulse library

        self.freq=freq2reg(adcfreq(cfg["frequency"]))  # convert frequency to dac frequency (ensuring it is an available adc frequency)
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        cfg=self.cfg
        self.trigger_adc(adc1=1, adc2=1,adc_trig_offset=cfg["adc_trig_offset"])  # trigger the adc acquisition
        self.pulse(ch=cfg["res_ch"], name="measure", freq=self.freq, phase=0, gain=cfg["pulse_gain"],  length=cfg['pulse_length'], t= 0, play=True)
        self.sync_all(us2cycles(self.cfg["relax_delay"]))  # sync all channels

"""Resonator spectroscopy, the IF frequency is varied in a python 'for' loop"""
class ResonatorSpectroscopy(AveragerProgram):
    def initialize(self):
        cfg=self.cfg
        self.cfg["adc_lengths"]=[self.cfg["readout_length"]]*2          #add length of adc acquisition to config
        self.cfg["adc_freqs"]=[adcfreq(self.cfg["frequency"])]*2   #add frequency of adc ddc to config

        self.add_pulse(ch=self.cfg["res_ch"], name="measure", style="const", length=self.cfg["readout_length"])  #add a constant pulse to the pulse library of res_ch
        freq=freq2reg(adcfreq(cfg["frequency"]))  # convert frequency to dac frequency (ensuring it is an available adc frequency)

        self.pulse(ch=cfg["res_ch"], name="measure", freq=freq, phase=0, gain=cfg["res_gain"],  length=self.cfg['readout_length'], t= 0, play=False) # pre-configure readout pulse
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.trigger_adc(adc1=1, adc2=0,adc_trig_offset=self.cfg["adc_trig_offset"])  # trigger the adc acquisition
        self.pulse(ch=self.cfg["res_ch"], play=True) # play readout pulse
        self.sync_all(us2cycles(self.cfg["relax_delay"]))  # sync all channels

"""Resonator spectroscopy without a pi pulse and with a pi pulse"""
class geReadoutChi(AveragerProgram):
    def initialize(self):
        cfg=self.cfg
        self.cfg["adc_lengths"]=[self.cfg["readout_length"]]*2          #add length of adc acquisition to config
        self.cfg["adc_freqs"]=[adcfreq(self.cfg["frequency"])]*2   #add frequency of adc ddc to config

        self.add_pulse(ch=self.cfg["res_ch"], name="measure", style="const", length=self.cfg["readout_length"])  #add a constant pulse to the pulse library of res_ch
        freq=freq2reg(adcfreq(cfg["frequency"]))  # convert frequency to dac frequency (ensuring it is an available adc frequency)

        self.pulse(ch=cfg["res_ch"], name="measure", freq=freq, phase=0, gain=cfg["res_gain"],  length=self.cfg['readout_length'], t= 0, play=False) # pre-configure readout pulse
        self.synci(200)  # give processor some time to configure pulses

    def body(self):
        self.trigger_adc(adc1=1, adc2=0,adc_trig_offset=self.cfg["adc_trig_offset"])  # trigger the adc acquisition
        self.pulse(ch=self.cfg["res_ch"], play=True) # play readout pulse
        self.sync_all(us2cycles(self.cfg["relax_delay"]))  # sync all channels

# These program uses the RAveragerProgram class, which allows you to sweep a parameter directly on the processor
# rather than in python as in the above example
# Because the whole sweep is done on the processor there is less downtime (especially for fast experiments)

"""Qubit ge spectroscopy, varying the IF frequency in the program itself"""
class geSpectroscopy(RAveragerProgram):
    def initialize(self):
        cfg=self.cfg

        self.q_rp=self.ch_page(self.cfg["qubit_ch"])     # get register page for qubit_ch
        self.r_freq=self.sreg(cfg["qubit_ch"], "freq")   # get frequency register for qubit_ch

        f_res=freq2reg(adcfreq(cfg["f_res"]))            # conver f_res to dac register value

        self.cfg["adc_lengths"]=[self.cfg["readout_length"]]*2   #copy over adc acquisition parameters
        self.cfg["adc_freqs"]=[adcfreq(cfg["f_res"])]*2

        # add qubit and readout pulses to respective channels
        self.add_pulse(ch=self.cfg["qubit_ch"], name="qubit",style="const", length=self.cfg["probe_length"])
        self.add_pulse(ch=self.cfg["res_ch"], name="measure",style="const", length=self.cfg["readout_length"])
        self.f_start =freq2reg(cfg["start"])  # get start/step frequencies
        self.f_step =freq2reg(cfg["step"])

        # pre-initialize pulses
        self.pulse(ch=cfg["qubit_ch"], name="qubit", phase=0, freq=self.f_start, gain=cfg["qubit_gain"], play=False)
        self.pulse(ch=cfg["res_ch"], name="measure", freq=f_res, phase=cfg['res_phase'], gain=cfg["res_gain"], play=False)

        self.sync_all(us2cycles(1))

    def body(self):
        self.pulse(ch=self.cfg["qubit_ch"], play=True, phase=deg2reg(0))  #play probe pulse
        self.sync_all(us2cycles(0.05)) # align channels and wait 50ns
        self.trigger_adc(adc1=1, adc2=0, adc_trig_offset=self.cfg["adc_trig_offset"])  #trigger measurement
        self.pulse(ch=self.cfg["res_ch"], play=True) # play measurement pulse
        self.sync_all(us2cycles(self.cfg["relax_delay"]))  # wait for qubit to relax

    def update(self):
        self.mathi(self.q_rp, self.r_freq, self.r_freq, '+', self.f_step) # update frequency list index


"""Probe the ef by pi pulsing the qubit from g->e first"""
class efSpectroscopy(RAveragerProgram):
    def initialize(self):
        cfg=self.cfg

        self.q_rp=self.ch_page(self.cfg["qubit_ch"])     # get register page for qubit_ch
        self.r_freq=self.sreg(cfg["qubit_ch"], "freq")   # get special frequency register for qubit_ch
        self.r_freq2=4   # get frequency register for qubit_ch

        f_res=freq2reg(adcfreq(cfg["f_res"]))            # conver f_res to dac register value

        self.cfg["adc_lengths"]=[self.cfg["readout_length"]]*2   #copy over adc acquisition parameters
        self.cfg["adc_freqs"]=[adcfreq(cfg["f_res"])]*2

        # add qubit and readout pulses to respective channels
        self.add_pulse(ch=self.cfg["qubit_ch"], name="qubit",style="const", length=self.cfg["probe_length"])
        self.add_pulse(ch=self.cfg["res_ch"], name="measure",style="const", length=self.cfg["readout_length"])
        self.add_pulse(ch=self.cfg["qubit_ch"], name="pi_qubit",style="arb", idata=gauss(
            mu=cfg["sigma"]*16*4/2,si=cfg["sigma"]*16,length=4*cfg["sigma"]*16,maxv=2**15-1))
        self.f_start =freq2reg(cfg["start"])  # get start/step frequencies
        self.f_step =freq2reg(cfg["step"])

        self.safe_regwi(self.q_rp, self.r_freq2, self.f_start) #set start frequency to r_freq2

        # pre-initialize pulses
        self.pulse(ch=cfg["qubit_ch"], name="pi_qubit", phase=0, freq=freq2reg(cfg["f_ge"]), gain=cfg["pi_gain"], play=False)
        self.pulse(ch=cfg["qubit_ch"], name="qubit", phase=0, freq=self.f_start, gain=cfg["qubit_gain"], play=False)
        self.pulse(ch=cfg["res_ch"], name="measure", freq=f_res, phase=cfg['res_phase'], gain=cfg["res_gain"], play=False)

        self.sync_all(us2cycles(1))

    def body(self):
        #note that I added the freq=freq2reg(cfg["f_ge"]) so that it will play at the ge frequency
        self.pulse(ch=self.cfg["qubit_ch"], name="pi_qubit", gain=self.cfg["pi_gain"], phase=deg2reg(0), freq=freq2reg(self.cfg["f_ge"]), play=True)  # put qubit from g to e
        #self.sync_all()  you don't need to sync because they are on the same channel
        self.mathi(self.q_rp, self.r_freq, self.r_freq2,"+",0) #copy the probe frequency into the r_freq special register
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit", gain=self.cfg["qubit_gain"], phase=deg2reg(0), play=True)  # play probe pulse
        self.sync_all(us2cycles(0.05)) # align channels and wait 50ns
        self.trigger_adc(adc1=1, adc2=0, adc_trig_offset=self.cfg["adc_trig_offset"])  #trigger measurement
        self.pulse(ch=self.cfg["res_ch"], play=True) # play measurement pulse
        self.sync_all(us2cycles(self.cfg["relax_delay"]))  # wait for qubit to relax

    def update(self):
        self.mathi(self.q_rp, self.r_freq2, self.r_freq2, '+', self.f_step) # update frequency list index

"""Choose a sigma for a Gaussian pulse (length = 4*sigma) and then vary the amplitude to calibrate the pi amplitude"""
class PowerRabi(RAveragerProgram):

    def initialize(self):
        cfg=self.cfg

        self.q_rp=self.ch_page(self.cfg["qubit_ch"])     # get register page for qubit_ch
        self.r_gain=self.sreg(cfg["qubit_ch"], "gain")   # get gain register for qubit_ch

        f_res=freq2reg(adcfreq(cfg["f_res"]))            # conver f_res to dac register value
        f_ge=freq2reg(cfg["f_ge"])

        self.cfg["adc_lengths"]=[self.cfg["readout_length"]]*2   #copy over adc acquisition parameters
        self.cfg["adc_freqs"]=[adcfreq(cfg["f_res"])]*2

        # add qubit and readout pulses to respective channels
        self.add_pulse(ch=self.cfg["qubit_ch"], name="qubit", style="arb",
                       idata=gauss(mu=cfg["sigma"]*16*4/2,si=cfg["sigma"]*16,length=4*cfg["sigma"]*16,maxv=2**15-1),
                       qdata=0*gauss(mu=cfg["sigma"]*16*4/2,si=cfg["sigma"]*16,length=4*cfg["sigma"]*16,maxv=2**15-1))
        self.add_pulse(ch=self.cfg["res_ch"], name="measure", style="const", length=self.cfg["readout_length"])

        # pre-initialize pulses
        self.pulse(ch=cfg["qubit_ch"], name="qubit", phase=0, freq=f_ge, gain=cfg["start"], play=False)
        self.pulse(ch=cfg["res_ch"], name="measure", freq=f_res, phase=cfg["res_phase"], gain=cfg["res_gain"], play=False)

        self.sync_all(us2cycles(1))

    def body(self):
        self.pulse(ch=self.cfg["qubit_ch"], play=True, phase=deg2reg(0))  #play probe pulse
        self.sync_all(us2cycles(0.05)) # align channels and wait 50ns
        self.trigger_adc(adc1=1, adc2=0, adc_trig_offset=self.cfg["adc_trig_offset"])  #trigger measurement
        self.pulse(ch=self.cfg["res_ch"], name="measure", play=True) # play measurement pulse
        self.sync_all(us2cycles(self.cfg["relax_delay"]))  # wait for qubit to relax

    def update(self):
        self.mathi(self.q_rp, self.r_gain, self.r_gain, '+', self.cfg["step"]) # update gain of the Gaussian pi pulse

"""Pi pulse the qubit and vary the wait time before measurement to get T1"""
class geT1(RAveragerProgram):
    def __init__(self,cfg):
        AveragerProgram.__init__(self,cfg)

    def initialize(self):
        cfg=self.cfg

        self.q_rp=self.ch_page(self.cfg["qubit_ch"])     # get register page for qubit_ch
        self.r_wait = 3
        self.regwi(self.q_rp, self.r_wait, cfg["start"])

        self.cfg["adc_lengths"]=[self.cfg["readout_length"]]*2   #copy over adc acquisition parameters
        self.cfg["adc_freqs"]=[adcfreq(cfg["f_res"])]*2

        # add qubit and readout pulses to respective channels
        self.add_pulse(ch=self.cfg["qubit_ch"], name="qubit",style="arb",
                       idata=gauss(mu=cfg["sigma"]*16*4/2,si=cfg["sigma"]*16,length=4*cfg["sigma"]*16,maxv=2**15-1))
        self.add_pulse(ch=self.cfg["res_ch"], name="measure",style="const", length=self.cfg["readout_length"])

        # pre-initialize pulses
        self.pulse(ch=cfg["qubit_ch"], name="qubit", phase=0, freq=freq2reg(cfg["f_ge"]), gain=cfg["pi_gain"], play=False)
        self.pulse(ch=cfg["res_ch"], name="measure", freq=freq2reg(adcfreq(cfg["f_res"])), phase=cfg['res_phase'], gain=cfg["res_gain"], play=False)
        self.sync_all(us2cycles(5))

    def body(self):

        self.pulse(ch=self.cfg["qubit_ch"], play=True, phase=deg2reg(0))  #play probe pulse
        self.sync_all()
        self.sync(self.q_rp,self.r_wait)
        self.trigger_adc(adc1=1, adc2=0, adc_trig_offset=self.cfg["adc_trig_offset"])  #trigger measurement
        self.pulse(ch=self.cfg["res_ch"], name="measure", play=True) # play measurement pulse

        self.sync_all(us2cycles(self.cfg["relax_delay"]))  # wait for qubit to relax

    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, '+', us2cycles(self.cfg["step"])) # update frequency list index

"""Ramsey experiment to get the dephasing and qubit frequency accurately.
We advance the phase of the second pi/2 pulse to mimick frequency detuning. 
This method is better than actually detuning the drive frequency as the pi calibration is off in the latter."""
class gePhaseRamsey(RAveragerProgram):
    def __init__(self,cfg):
        AveragerProgram.__init__(self,cfg)

    def initialize(self):
        cfg=self.cfg

        self.q_rp=self.ch_page(self.cfg["qubit_ch"])     # get register page for qubit_ch
        self.r_wait = 3
        self.r_phase2 = 4
        self.r_phase=self.sreg(cfg["qubit_ch"], "phase")
        self.regwi(self.q_rp, self.r_wait, cfg["start"])
        self.regwi(self.q_rp, self.r_phase2, 0)

        f_res=freq2reg(adcfreq(cfg["f_res"]))            # conver f_res to dac register value
        f_ge=freq2reg(cfg["f_ge"])

        self.cfg["adc_lengths"]=[self.cfg["readout_length"]]*2   #copy over adc acquisition parameters
        self.cfg["adc_freqs"]=[adcfreq(cfg["f_res"])]*2

        # add qubit and readout pulses to respective channels
        self.add_pulse(ch=self.cfg["qubit_ch"], name="qubit", style="arb", idata=gauss(mu=cfg["sigma"]*16*4/2,si=cfg["sigma"]*16, length=4*cfg["sigma"]*16, maxv=2**15-1))
        self.add_pulse(ch=self.cfg["res_ch"], name="measure", style="const", length=self.cfg["readout_length"])

        # pre-initialize pulses
        self.pulse(ch=cfg["qubit_ch"], name="qubit", phase=0, freq=f_ge, gain=cfg["pi2_gain"], play=False)
        self.pulse(ch=cfg["res_ch"], name="measure", freq=f_res, phase=cfg['res_phase'], gain=cfg["res_gain"], play=False)

        self.sync_all(us2cycles(0.2))

    def body(self):

        self.pulse(ch=self.cfg["qubit_ch"], phase=deg2reg(0), play=True)  #play probe pulse
        self.mathi(self.q_rp, self.r_phase, self.r_phase2,"+",0)
        self.sync_all()
        self.sync(self.q_rp,self.r_wait)
        self.pulse(ch=self.cfg["qubit_ch"], play=True)  #play second pi/2 pulse with the updated phase
        self.sync_all(us2cycles(0.05))
        self.trigger_adc(adc1=1, adc2=1, adc_trig_offset=self.cfg["adc_trig_offset"])  #trigger measurement
        self.pulse(ch=self.cfg["res_ch"], name="measure", play=True) # play measurement pulse
        self.sync_all(us2cycles(self.cfg["relax_delay"]))  # wait for qubit to relax

    def update(self):
        self.mathi(self.q_rp, self.r_wait, self.r_wait, '+', self.cfg["step"]) # update the time between two π/2 pulses
        self.mathi(self.q_rp, self.r_phase2, self.r_phase2, '+', self.cfg["phase_step"]) # advance the phase of the LO for the second π/2 pulse