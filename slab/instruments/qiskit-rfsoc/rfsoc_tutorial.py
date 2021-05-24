"""
rfsoc_tutorial.py
"""

import sys
import time

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from slab import generate_file_path

sys.path.append("/home/xilinx") # for pynq
sys.path.append("/home/xilinx/repos/qsystem0/pynq")
from qsystem0 import *
from qsystem0_asm2 import *
from averager_program import AveragerProgram

# constants
DPI = 300

class ResonatorSpectroscopyProgram(AveragerProgram):
    def __init__(self,cfg):
        AveragerProgram.__init__(self,cfg)
        
        
    def initialize(self):
        cfg=self.cfg
   
        self.r_freq=self.sreg(cfg["res_ch"],"freq")
        self.r_adc_freq=self.sreg(cfg["res_ch"],"adc_freq")
        
        self.f_start=self.freq2reg(cfg["start"])
        self.f_step=self.freq2reg(cfg["step"])

        self.measure(ch=cfg["res_ch"],  freq=self.f_start,
                     gain=cfg["res_gain"], phase=cfg["res_phase"],
                     length=self.cfg['readout_length'], play=False)
    
    def get_expt_pts(self):
        return self.reg2freq(self.f_start)+np.arange(self.cfg['expts'])*self.reg2freq(self.f_step)
    
    def initialize_round(self):
        self.regwi(self.ch_page(self.cfg["res_ch"]),self.r_freq, self.f_start)
        self.regwi(self.ch_page(self.cfg["res_ch"]),self.r_adc_freq, 2*self.f_start)
    
    def body(self):
        self.measure(ch=self.cfg["res_ch"],  length=self.cfg['readout_length'], play=True)
        self.sync_all()
        self.delay(self.us2cycles(self.cfg["relax_delay"]))
         
    def update(self):
        self.mathi(self.ch_page(self.cfg["res_ch"]), self.r_freq, self.r_freq, '+', self.f_step)
        self.mathi(self.ch_page(self.cfg["res_ch"]), self.r_adc_freq, self.r_adc_freq,
                   '+', 2*self.f_step)
        self.seti(5, self.ch_page(self.cfg["res_ch"]), self.r_adc_freq, 0)
        self.synci(10)
#ENDCLASS

def rs():
    soc = PfbSoc("/home/xilinx/repos/qsystem0/pynq/qsystem_0.bit")
    
    config = {
        "res_ch":3,
        "start":90, "step":.1, "expts":200, "reps": 1000,"rounds":1,
        "readout_length":2000, "res_gain":4000,"res_phase":0,
        "relax_delay":10
    }

    rspec = ResonatorSpectroscopyProgram(cfg=config)
    x_pts, avgi, avgq, avgamp = rspec.acquire(soc)

    plt.subplot(111, xlabel='DDS Frequency (MHz)', ylabel='Amplitude',
                title='Resonator Spectroscopy')
    #plot(avg_di, label='I')
    #plot(avg_dq, label='Q')
    plt.plot(x_pts, avgamp, label='A')
    fmax = x_pts[np.argmax(avgamp)]
    #plot(amps, label='A')
    plt.axvline(fmax)
    plt.legend()
    plt.tight_layout()

    # kernel_size = 1000
    # kernel = np.ones(kernel_size) / kernel_size
    # data_convolved_i = np.convolve(rspec.di_buf, kernel, mode='same')
    # data_convolved_q = np.convolve(rspec.dq_buf, kernel, mode='same')
    # plt.plot(sqrt(data_convolved_i**2+data_convolved_q**2))

    save_file_path = generate_file_path(".", "res_spec", "png")
    plt.savefig(save_file_path, dpi=DPI)
    print("plotted to {}".format(save_file_path))
    return None
#ENDDEF

class QubitSpectroscopyProgram(AveragerProgram):
    def __init__(self,cfg):
        AveragerProgram.__init__(self,cfg)
        
        
    def initialize(self):
        cfg=self.cfg

        self.r_freq=self.sreg(cfg["qubit_ch"], "freq")
        self.add_pulse(cfg["qubit_ch"],"Rabi",cfg['pulses']['Rabi'])
    
        f_res=self.freq2reg(cfg["f_res"])
        self.f_start =self.freq2reg(cfg["start"])
        self.f_step =self.freq2reg(cfg["step"])

        self.set_wave(ch=cfg["qubit_ch"], pulse="Rabi", phase=0,
                      freq=self.f_start, gain=cfg["qubit_gain"], play=False)
        self.measure(ch=cfg["res_ch"],  freq=f_res, gain=cfg["res_gain"],
                     phase=cfg["res_phase"], length=self.cfg['readout_length'], play=False)
        
        self.synci(1000)
        
    def initialize_round(self):
        self.regwi(self.ch_page(self.cfg["qubit_ch"]),self.r_freq, self.f_start)
    
    def body(self):
        self.set_wave(ch=self.cfg["qubit_ch"], play=True) 
        self.align((3,4))
        self.measure(ch=self.cfg["res_ch"],  length=self.cfg['readout_length'], play=True)
        self.sync_all()
        self.delay(self.us2cycles(self.cfg["relax_delay"]))

    
    def update(self):
        self.mathi(self.ch_page(self.cfg["qubit_ch"]), self.r_freq, self.r_freq, '+', self.f_step)
#ENDCLASS

def qubit_spec():
    pulse_length=256

    config={"pulses": {"Rabi":gauss(mu=pulse_length*16/2,si=pulse_length*16/7,length=pulse_length*16,maxv=32000)},
            "amp_ch":1,"storage_ch":2,"res_ch":3,"qubit_ch":4,
            "f_res":100.25,
            "start":80, "step":0.2, "expts":200, "reps": 2000, "rounds":2,
            "readout_length":1000, "res_gain":8000, "qubit_gain":2000, "res_phase":0,
            "relax_delay":200
           }

    qspec=QubitSpectroscopyProgram(cfg=config)
    x_pts, avgi, avgq, avgamp = qspec.acquire(soc)

    subplot(111, xlabel='DDS Frequency (MHz)', ylabel='Amplitude', title='Qubit Spectroscopy')
    #plot(avg_di, label='I')
    #plot(avg_dq, label='Q')
    plot(x_pts,avgamp, label='A')
    #plot(amps, label='A')
    #axvline(fmin)
    legend()
    tight_layout()
    fmin=x_pts[argmin(avgamp)]
    axvline(fmin)
    print(fmin)

    plot (x_pts,avgi)
    plot (x_pts,avgq)

    kernel_size=1000
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved_i = np.convolve(qspec.di_buf, kernel, mode='same')
    data_convolved_q = np.convolve(qspec.dq_buf, kernel, mode='same')
    plot(sqrt(data_convolved_i**2+data_convolved_q**2))
#ENDDEF

class AmplitudeRabiProgram(AveragerProgram):
    def __init__(self,cfg):
        AveragerProgram.__init__(self,cfg)
        
        
    def initialize(self):
        cfg=self.cfg

        self.r_gain=self.sreg(cfg["qubit_ch"], "gain")
        self.add_pulse(cfg["qubit_ch"],"Rabi",cfg['pulses']['Rabi'])
    
        f_res=self.freq2reg(cfg["f_res"])
        f_qubit= self.freq2reg(cfg["f_ge"])

        self.set_wave(ch=cfg["qubit_ch"], pulse="Rabi", phase=0,
                      freq=f_qubit, gain=cfg["start"], play=False)
        self.measure(ch=cfg["res_ch"],  freq=f_res, gain=cfg["res_gain"],
                     phase=cfg["res_phase"], length=self.cfg['readout_length'], play=False)
        
        self.synci(1000)
        
    def initialize_round(self):
        self.regwi(self.ch_page(self.cfg["qubit_ch"]),self.r_gain, self.cfg["start"])
    
    def body(self):
        self.set_wave(ch=self.cfg["qubit_ch"], play=True) 
        self.align((3,4))
        #self.sync_all()
        self.measure(ch=self.cfg["res_ch"],  length=self.cfg['readout_length'], play=True)
        self.sync_all()
        self.delay(self.us2cycles(self.cfg["relax_delay"]))

    
    def update(self):
        self.mathi(self.ch_page(self.cfg["qubit_ch"]), self.r_gain, self.r_gain,
                   '+', self.cfg["step"])
#ENDCLASS

def rabi():
    soc = PfbSoc("/home/xilinx/repos/qsystem0/pynq/qsystem_0.bit")
    
    pulse_length=256

    config = {
        "fs_dac":384*16,"fs_adc":384*8,
        "pulses": {
            "Rabi": gauss(mu=pulse_length*16/2, si=pulse_length*16/7,
                          length=pulse_length*16, maxv=32000)
        },
        "amp_ch":1,"storage_ch":2,"res_ch":3,"qubit_ch":4,
        "f_ge":100, "f_res":99.65625,
        "start":0, "step":100,
        "expts": 200,
        "reps": 2000,
        "rounds":1,
        "readout_length":1000, "res_gain":8000,"res_phase":0,
        "relax_delay":500
    }

    rabi = AmplitudeRabiProgram(cfg=config)
    x_pts, avgi, avgq, avgamp = rabi.acquire(soc)
    plt.subplot(111, xlabel='Gain', ylabel='Amplitude', title='Amplitude Rabi')
    # plt.plot(avg_di, label='I')
    # plt.plot(avg_dq, label='Q')
    plt.plot(x_pts,avgamp, label='A')
    #plot(amps, label='A')
    #axvline(fmin)
    plt.legend()
    plt.tight_layout()
    #print(fmin)
    plt.axvline(4950)
    plt.axvline(2745)
    save_file_path = generate_file_path(".", "rabi", "png")
    plt.savefig(save_file_path, dpi=DPI)
    print("saved plot to {}".format(save_file_path))
#ENDDEF

class RamseyProgram(AveragerProgram):
    def __init__(self,cfg):
        AveragerProgram.__init__(self,cfg)
        
    def get_expt_pts(self):
        return (self.cycles2us(self.us2cycles(self.cfg["start"]))
                + arange(self.cfg['expts'])*self.cycles2us(self.us2cycles(self.cfg["step"])))
        
    def initialize(self):
        cfg=self.cfg

        self.r_dd = 1
        self.r_num = 2
        self.r_phase2 = 3
        self.r_phase= self.sreg(cfg["qubit_ch"], "phase")
        
        self.add_pulse(cfg["qubit_ch"],"Rabi",cfg['pulses']['Rabi'])
    
        f_res=self.freq2reg( cfg["f_res"])
        f_qubit=self.freq2reg( cfg["f_ge"])

        self.set_wave(ch=cfg["qubit_ch"], pulse="Rabi", phase =0, freq=f_qubit, gain=cfg["qubit_gain"], play=False)
        self.measure(ch=cfg["res_ch"],  freq=f_res, gain=cfg["res_gain"], phase=cfg["res_phase"], length=self.cfg['readout_length'], play=False)
        
        self.synci(1000)
        
    def initialize_round(self):
        self.regwi(self.ch_page(self.cfg["qubit_ch"]),self.r_num, 0)
        self.regwi(self.ch_page(self.cfg["qubit_ch"]),self.r_phase2, 0)
    
    def body(self):
        self.set_wave(ch=self.cfg["qubit_ch"], phase=0, play=True) 
        self.sync_all()
        
        ### Delay for variable length time 
        self.delay(self.us2cycles(self.cfg["start"]))
        self.mathi(self.ch_page(self.cfg["qubit_ch"]), self.r_dd, self.r_num, "+", 0)
        self.label("LOOP_D")
        self.delay(self.us2cycles(self.cfg["step"]))
        self.loopnz(self.ch_page(self.cfg["qubit_ch"]),self.r_dd, "LOOP_D")
        ###
        
        self.mathi(self.ch_page(self.cfg["qubit_ch"]), self.r_phase, self.r_phase2, "+", 0) #adjust phase
        self.set_wave(ch=self.cfg["qubit_ch"], play=True)       # 2nd pi/2 pulse
        self.sync_all()
        
        self.measure(ch=self.cfg["res_ch"],  length=self.cfg['readout_length'], play=True)
        self.sync_all()
        
        self.delay(self.us2cycles(self.cfg["relax_delay"]))

    
    def update(self):
        self.mathi(self.ch_page(self.cfg["qubit_ch"]), self.r_num, self.r_num, '+', 1)
        self.mathi(self.ch_page(self.cfg["qubit_ch"]), self.r_phase2, self.r_phase2, "+", self.cfg["phase_step"])
#ENDCLASS

def ramsey():
    pulse_length = 256

    config = {
        "pulses": {
            "Rabi": gauss(mu=pulse_length*16/2, si=pulse_length*16/7,
                          length=pulse_length*16,maxv=32000)
        },
        "amp_ch":1,"storage_ch":2,"res_ch":3,"qubit_ch":4,
        "f_ge":101, "f_res":99.65625,
        "start":0, "step":.5, "phase_step": 2**16//36, "expts":200, "reps": 2000, "rounds":1,
        "readout_length":1000, "res_gain":8000, "qubit_gain": 2745, "res_phase":0,
        "relax_delay":250
       }

    ramsey=RamseyProgram(cfg=config)
    x_pts, avgi, avgq, avgamp = ramsey.acquire(soc)
    subplot(111, xlabel='Delay ($\mu$s)', ylabel='Amplitude', title='Amplitude Rabi')
    #plot(avg_di, label='I')
    #plot(avg_dq, label='Q')
    plot(x_pts,avgamp, label='A')
    #plot(amps, label='A')
    #axvline(fmin)
    legend()
    tight_layout()
#ENDDEF

class T1Program(AveragerProgram):
    def __init__(self,cfg):
        AveragerProgram.__init__(self,cfg)
        
    def get_expt_pts(self):
        return self.cycles2us(self.us2cycles(self.cfg["start"]))+arange(self.cfg['expts'])*self.cycles2us(self.us2cycles(self.cfg["step"]))
        
    def initialize(self):
        cfg=self.cfg

        self.r_dd = 1
        self.r_num = 2
        
        self.add_pulse(cfg["qubit_ch"],"Rabi",cfg['pulses']['Rabi'])
    
        f_res=self.freq2reg( cfg["f_res"])
        f_qubit=self.freq2reg( cfg["f_ge"])

        self.set_wave(ch=cfg["qubit_ch"], pulse="Rabi", phase =0, freq=f_qubit, gain=cfg["qubit_gain"], play=False)
        self.measure(ch=cfg["res_ch"],  freq=f_res, gain=cfg["res_gain"], phase=cfg["res_phase"], length=self.cfg['readout_length'], play=False)
        
        self.synci(1000)
        
    def initialize_round(self):
        self.regwi(self.ch_page(self.cfg["qubit_ch"]),self.r_num, 0)
    
    def body(self):
        self.set_wave(ch=self.cfg["qubit_ch"], play=True) 
        self.sync_all()
        
        ### Delay for variable length time 
        self.delay(self.us2cycles(self.cfg["start"]))
        self.mathi(self.ch_page(self.cfg["qubit_ch"]), self.r_dd, self.r_num, "+", 0)
        self.label("LOOP_D")
        self.delay(self.us2cycles(self.cfg["step"]))
        self.loopnz(self.ch_page(self.cfg["qubit_ch"]),self.r_dd, "LOOP_D")
        ###
               
        self.measure(ch=self.cfg["res_ch"],  length=self.cfg['readout_length'], play=True)
        self.sync_all()
        
        self.delay(self.us2cycles(self.cfg["relax_delay"]))

    def update(self):
        self.mathi(self.ch_page(self.cfg["qubit_ch"]), self.r_num, self.r_num, '+', 1)
#ENDCLASS

def t1():
    config={"pulses": {"Rabi":gauss(mu=pulse_length*16/2,si=pulse_length*16/7,length=pulse_length*16,maxv=32000)},
        "amp_ch":1,"storage_ch":2,"res_ch":3,"qubit_ch":4,
        "f_ge":101, "f_res":99.65625,
        "start":0, "step":2, "expts":200, "reps": 2000, "rounds":1,
        "readout_length":1000, "res_gain":8000, "qubit_gain": 4950, "res_phase":0,
        "relax_delay":250
       }

    T1p=T1Program(cfg=config)
    x_pts, avgi, avgq, avgamp = T1p.acquire(soc)
    subplot(111, xlabel='Delay ($\mu$s)', ylabel='Amplitude', title='T1 Measurement')
    #plot(avg_di, label='I')
    #plot(avg_dq, label='Q')
    plot(x_pts,avgamp, label='A')
    #plot(amps, label='A')
    #axvline(fmin)
    legend()
    tight_layout()
#ENDDEF

class SingleShotProgram(AveragerProgram):
    def __init__(self,cfg):
        cfg["start"]=0
        cfg["step"]=cfg["qubit_gain"]
        cfg["expts"]=2
        cfg["reps"]=cfg["shots"]
        cfg["rounds"]=1
        AveragerProgram.__init__(self,cfg)     
        
    def initialize(self):
        cfg=self.cfg

        self.r_gain=self.sreg(cfg["qubit_ch"], "gain")
        self.add_pulse(cfg["qubit_ch"],"Rabi",cfg['pulses']['Rabi'])
    
        f_res=self.freq2reg(cfg["f_res"])
        f_amp=self.freq2reg(cfg["f_amp"])
        f_qubit= self.freq2reg(cfg["f_ge"])

        self.set_wave(ch=cfg["qubit_ch"], pulse="Rabi", phase =0, freq=f_qubit, gain=cfg["start"], play=False)
        self.set_wave(ch=cfg["amp_ch"], phase =cfg["amp_phase"]*2**16//360, freq=f_res, gain=cfg["amp_gain"], outsel=1, length=self.cfg['readout_length'], play=False)
        self.measure(ch=cfg["res_ch"],  freq=f_res, gain=cfg["res_gain"], phase=cfg["res_phase"], length=self.cfg['readout_length'], play=False)
        
        self.synci(1000)
        
    def initialize_round(self):
        self.regwi(self.ch_page(self.cfg["qubit_ch"]),self.r_gain, self.cfg["start"])
    
    def body(self):
        self.set_wave(ch=self.cfg["qubit_ch"], play=True) 
        self.align((1,3,4))
        self.set_wave(ch=self.cfg["amp_ch"], outsel=1, length=self.cfg['readout_length'], play=True) 
        self.measure(ch=self.cfg["res_ch"],  length=self.cfg['readout_length'], play=True)
        self.sync_all()
        self.delay(self.us2cycles(self.cfg["relax_delay"]))

    
    def update(self):
        self.mathi(self.ch_page(self.cfg["qubit_ch"]), self.r_gain, self.r_gain, '+', self.cfg["step"])
        
    def acquire(self,soc):
        super().acquire(soc)
        return self.make_histogram()
        
        
    def make_histogram(self):
        shots_i=self.di_buf.reshape((self.cfg["expts"],self.cfg["reps"]))/self.cfg['readout_length']
        shots_q=self.dq_buf.reshape((self.cfg["expts"],self.cfg["reps"]))/self.cfg['readout_length']
        return shots_i,shots_q
        
    def analyze(self,shots_i,shots_q):
        subplot(111,xlabel='I',ylabel='Q', title='Single Shot Histogram')
        plot(shots_i[0],shots_q[0],'.',label='g')
        plot(shots_i[1],shots_q[1],'.',label='e')
        legend()
        gca().set_aspect('equal', 'datalim')
#ENDCLASS

def single_shot():
    pulse_length=256

    config={
            "pulses": {"Rabi":gauss(mu=pulse_length*16/2,si=pulse_length*16/7,length=pulse_length*16,maxv=32000)},
            "amp_ch":1,"storage_ch":2,"res_ch":3,"qubit_ch":4,
            "f_ge":100, "f_res":99.65625,"f_amp":1089.65625, "qubit_gain": 4950, 
            "shots": 1000,
            "readout_length":2000, "res_gain":5000,"res_phase":0,
            "amp_gain": 0*4000, "amp_phase":0, 
            "relax_delay":500
           }

    ssp=SingleShotProgram(cfg=config)
    di,dq = ssp.acquire(soc)
    ssp.analyze(di,dq)

    pulse_length=256

    config={
            "pulses": {"Rabi":gauss(mu=pulse_length*16/2,si=pulse_length*16/7,length=pulse_length*16,maxv=32000)},
            "amp_ch":1,"storage_ch":2,"res_ch":3,"qubit_ch":4,
            "f_ge":100, "f_res":1099.65625, "qubit_gain": 4950, 
            "shots": 1000,
            "readout_length":3000, "res_gain":7500,"res_phase":0,
            "amp_gain": 0, "amp_phase":0, 
            "relax_delay":500
           }

    ssp=SingleShotProgram(cfg=config)
    di,dq = ssp.acquire(soc)
    ssp.analyze(di,dq)

    ssp.analyze(di,dq)

    ig, qg = 55,60

    axvline(ig)
    axhline(qg)

    ie,qe =15,50
    axvline(ie)
    axhline(qe)

    si,sq=(ig+ie)/2, (qg+qe)/2
    ige,qge = (ig-ie)/2, (qg-qe)/2

    g_data=np.dot(transpose((di[0],dq[0])),(ige,qge))
    e_data=np.dot(transpose((di[1],dq[1])),(ige,qge))

    g_histo=histogram(g_data, bins=50, range=(np.min(g_data),np.max(g_data)))
    e_histo=histogram(e_data, bins=50, range=(np.min(e_data),np.max(e_data)))
    g_sum=np.cumsum(g_histo[0])
    g_sum=g_sum/np.max(g_sum)
    e_sum=np.cumsum(e_histo[0])
    e_sum=e_sum/np.max(e_sum)

    subplot(211)
    plot(g_histo[1][:-1],g_histo[0])
    plot(e_histo[1][:-1],e_histo[0])
    subplot(212)
    plot(g_histo[1][:-1],g_sum)
    plot(e_histo[1][:-1],e_sum)
    #plot(histogram(e_data, bins=50, range=(np.min(g_data),np.max(e_data)))[0])
    ylim()
#ENDDEF

def main():
    res_spec()
#ENDDEF

if __name__ == "__main__":
    main()
#ENDIF
