try:
    from .sequencer_pxi import Sequencer
    from .pulse_classes import Gauss, Idle, Ones, Square, DRAG, ARB_freq_a,Square_two_tone, linear_ramp, adb_ramp, linear_ramp_rev
    from .pulse_classes import * #exp_ramp,multiexponential_ramp,reversability_ramp, FreqSquare, WURST20_I, WURST20_Q
except:
    from sequencer import Sequencer
    from pulse_classes import Gauss, Idle, Ones, Square, DRAG, ARB_freq_a
# from qutip_experiment import run_qutip_experiment

import numpy as np
import visdom
import os
import pickle

import copy
import time

class PulseSequences:
    # channels and awgs

    def set_parameters(self, quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg):
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg
        self.lattice_cfg = lattice_cfg

        self.pulse_info = self.lattice_cfg['pulse_info']
        self.pulse_info_ancilla = self.lattice_cfg['edge']

        self.channels = hardware_cfg['channels']

        self.channels_awg = hardware_cfg['channels_awg'] #which awg each channel correspons to

        self.awg_info = hardware_cfg['awg_info']

        self.channels_delay = hardware_cfg['channels_delay_array_roll_ns']
        
        gauss_z = np.linspace(-2,2,20)
        gauss_envelop = np.exp(-gauss_z**2)

        gauss_z = np.linspace(-2,2,20)
        gauss_envelop = np.exp(-gauss_z**2)


    def  __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg, plot_visdom=True):
        self.set_parameters(quantum_device_cfg, experiment_cfg, hardware_cfg, lattice_cfg)
        self.plot_visdom = plot_visdom


    def gen_q(self,sequencer, qb =0,len = 10,amp = 1,add_freq = 0, iq_overwrite = None, Q_phase_overwrite = None, phase = 0,pulse_type = 'square'):
        setup = self.lattice_cfg["qubit"]["setup"][qb]
        if iq_overwrite!=None:
            iq_freq = iq_overwrite
        else:
            iq_freq = self.pulse_info[setup]['iq_freq'][qb]

        if Q_phase_overwrite!=None:
            Q_phase = Q_phase_overwrite
        else:
            Q_phase = self.pulse_info[setup]['Q_phase'][qb]
        
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s_I' % setup, Square(max_amp=amp, flat_len=len,
                                    ramp_sigma_len=0.001, cutoff_sigma=2, freq=iq_freq+add_freq,
                                    phase=phase))
            sequencer.append('charge%s_Q' % setup,
                         Square(max_amp=amp, flat_len= len,
                                ramp_sigma_len=0.001, cutoff_sigma=2, freq=iq_freq+add_freq,
                                phase=phase + Q_phase))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s_I' % setup, Gauss(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2,freq=iq_freq+add_freq,phase=phase))
            sequencer.append('charge%s_Q' % setup, Gauss(max_amp=amp, sigma_len=len,
                                                             cutoff_sigma=2,freq=iq_freq+add_freq,phase=phase+Q_phase))

        elif pulse_type.lower() == 'flatfreq':
            sequencer.append('charge%s_I' % setup, FreqSquare(max_amp=amp,
                                                              flat_freq_Ghz=self.pulse_info['flat_len_Ghz'][qb],
                                                              pulse_len=self.pulse_info['flat_freq_pulse_len'][qb],
                                                              conv_gauss_sig_Ghz=self.pulse_info['conv_gauss_sig_Ghz'][qb],
                                                              freq=iq_freq+add_freq,
                                                              phase=phase))

            sequencer.append('charge%s_Q' % setup, FreqSquare(max_amp=amp,
                                                              flat_freq_Ghz= self.pulse_info['flat_len_Ghz'][qb],
                                                              pulse_len=self.pulse_info['flat_freq_pulse_len'][qb],
                                                              conv_gauss_sig_Ghz=self.pulse_info['conv_gauss_sig_Ghz'][qb],
                                                              freq=iq_freq+add_freq,
                                                              phase=phase+Q_phase))
        elif pulse_type.lower() == 'multiflatfreq':
            sequencer.append('charge%s_I' % setup, FreqSquare(max_amp=amp,
                                                              flat_freq_Ghz=self.pulse_info['flat_len_Ghz'][qb],
                                                              pulse_len=self.pulse_info['flat_freq_pulse_len'][qb],
                                                              conv_gauss_sig_Ghz=self.pulse_info['conv_gauss_sig_Ghz'][qb],
                                                              freq=iq_freq + add_freq,
                                                              phase=phase))
            sequencer.append('charge%s_I' % setup, FreqSquare(max_amp=amp,
                                                              flat_freq_Ghz=self.pulse_info['flat_len_Ghz'][qb],
                                                              pulse_len=self.pulse_info['flat_freq_pulse_len'][qb],
                                                              conv_gauss_sig_Ghz=self.pulse_info['conv_gauss_sig_Ghz'][
                                                                  qb],
                                                              freq=iq_freq + add_freq,
                                                              phase=phase))


            sequencer.append('charge%s_Q' % setup, FreqSquare(max_amp=amp,
                                                              flat_freq_Ghz=self.pulse_info['flat_len_Ghz'][qb],
                                                              pulse_len=self.pulse_info['flat_freq_pulse_len'][qb],
                                                              conv_gauss_sig_Ghz=self.pulse_info['conv_gauss_sig_Ghz'][qb],
                                                              freq=iq_freq + add_freq,
                                                              phase=phase + Q_phase))
            sequencer.append('charge%s_Q' % setup, FreqSquare(max_amp=amp,
                                                              flat_freq_Ghz=self.pulse_info['flat_len_Ghz'][qb],
                                                              pulse_len=self.pulse_info['flat_freq_pulse_len'][qb],
                                                              conv_gauss_sig_Ghz=self.pulse_info['conv_gauss_sig_Ghz'][
                                                                  qb],
                                                              freq=iq_freq + add_freq,
                                                              phase=phase + Q_phase))
        elif pulse_type.lower() == 'wurst':
            sequencer.append('charge%s_I' % setup,
                             WURST20_I(max_amp=self.pulse_info['wurst_amp'][qb],
                                       bandwidth=self.pulse_info['wurst_bw'][qb],
                                       t_length=self.pulse_info['wurst_pulselen'][qb],
                                       freq=iq_freq + add_freq,
                                       phase=phase))
            sequencer.append('charge%s_Q' % setup,
                             WURST20_Q(max_amp=self.pulse_info['wurst_amp'][qb],
                                       bandwidth=self.pulse_info['wurst_bw'][qb],
                                       t_length=self.pulse_info['wurst_pulselen'][qb],
                                       freq=iq_freq + add_freq,
                                       phase=phase,
                                       phase_t0= Q_phase))
        elif pulse_type.lower() == 'bir4_degraaf':
            sequencer.append('charge%s_I' % setup,
                             BIR_4(max_amp=self.pulse_info['BIR4_amp'][qb],
                                       bandwidth=self.pulse_info['BIR4_bw'][qb],
                                       t_length=self.pulse_info['BIR4_pulse_len'][qb],
                                       theta = self.pulse_info['BIR4_theta'][qb],
                                       freq=iq_freq + add_freq,
                                       phase=phase))
            sequencer.append('charge%s_Q' % setup,
                             BIR_4(max_amp=self.pulse_info['BIR4_amp'][qb],
                                       bandwidth=self.pulse_info['BIR4_bw'][qb],
                                       t_length=self.pulse_info['BIR4_pulse_len'][qb],
                                       theta=self.pulse_info['BIR4_theta'][qb],
                                       freq=iq_freq + add_freq,
                                       phase=phase,
                                       phase_t0= Q_phase))
        elif pulse_type.lower() == 'bir4_hyp':
            sequencer.append('charge%s_I' % setup,
                             BIR_4_hyp(max_amp=self.pulse_info['BIR4_hyp_amp'][qb],
                                       bandwidth=self.pulse_info['BIR4_hyp_bw'][qb],
                                       t_length=self.pulse_info['BIR4_hyp_pulse_len'][qb],
                                       theta = self.pulse_info['BIR4_hyp_theta'][qb],
                                       kappa=self.pulse_info['BIR4_hyp_kappa'][qb],
                                       beta=self.pulse_info['BIR4_hyp_beta'][qb],
                                       freq=iq_freq + add_freq,
                                       phase=phase))
            sequencer.append('charge%s_Q' % setup,
                             BIR_4_hyp(max_amp=self.pulse_info['BIR4_hyp_amp'][qb],
                                       bandwidth=self.pulse_info['BIR4_hyp_bw'][qb],
                                       t_length=self.pulse_info['BIR4_hyp_pulse_len'][qb],
                                       theta=self.pulse_info['BIR4_hyp_theta'][qb],
                                       kappa=self.pulse_info['BIR4_hyp_kappa'][qb],
                                       beta=self.pulse_info['BIR4_hyp_beta'][qb],
                                       freq=iq_freq + add_freq,
                                       phase=phase,
                                       phase_t0= Q_phase))

        elif pulse_type.lower() == 'knill':
            phase = np.pi / 6
            if self.pulse_info['knill_pulse_type'][qb] == "square":
                sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb], phase=phase))
                sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb],
                                                              phase=phase + self.pulse_info[setup]['Q_phase'][qb]))
            elif self.pulse_info['knill_pulse_type'][qb] == "gauss":
                sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase))
                sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                 qb]))
            phase = 0
            if self.pulse_info['knill_pulse_type'][qb] == "square":
                sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb],
                                                              phase=phase))
                sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb],
                                                              phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                  qb]))
            elif self.pulse_info['knill_pulse_type'][qb] == "gauss":
                sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase))
                sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                 qb]))
            phase = np.pi / 2
            if self.pulse_info['knill_pulse_type'][qb] == "square":
                sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb],
                                                              phase=phase))
                sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb],
                                                              phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                  qb]))
            elif self.pulse_info['knill_pulse_type'][qb] == "gauss":
                sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase))
                sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                 qb]))
            phase = 0
            if self.pulse_info['knill_pulse_type'][qb] == "square":
                sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb],
                                                              phase=phase))
                sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb],
                                                              phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                  qb]))
            elif self.pulse_info['knill_pulse_type'][qb] == "gauss":
                sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase))
                sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                 qb]))
            phase = np.pi / 6
            if self.pulse_info['knill_pulse_type'][qb] == "square":
                sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb],
                                                              phase=phase))
                sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb],
                                                              phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                  qb]))
            elif self.pulse_info['knill_pulse_type'][qb] == "gauss":
                sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase))
                sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                 qb]))




    # def double_gen_q(self,sequencer,qb = 0,len1=10,len2 = 10,amp1 = 1,amp2 = 1,add_freq1 = 0,add_freq2 = 0,iq_overwrite1 = None,Q_phase_overwrite1 = None,iq_overwrite2 = None,Q_phase_overwrite2 = None,phase1 = 0,phase2 = 0,pulse_type1 = 'square',pulse_type2 = 'square'):
    #     setup = self.lattice_cfg["qubit"]["setup"][qb]
    #     if iq_overwrite1 != None:
    #         iq_freq1 = iq_overwrite1
    #     else:
    #         iq_freq1 = self.pulse_info[setup]['iq_freq'][qb]
    #     if iq_overwrite2 != None:
    #         iq_freq2 = iq_overwrite2
    #     else:
    #         iq_freq2 = self.pulse_info[setup]['iq_freq'][qb]
    #
    #     if Q_phase_overwrite1 != None:
    #         Q_phase1 = Q_phase_overwrite1
    #     else:
    #         Q_phase1 = self.pulse_info[setup]['Q_phase'][qb]
    #     if Q_phase_overwrite2 != None:
    #         Q_phase2 = Q_phase_overwrite2
    #     else:
    #         Q_phase2 = self.pulse_info[setup]['Q_phase'][qb]
    #
    #     if pulse_type.lower() == 'square':
    #         sequencer.append('charge%s_I' % setup, Square(max_amp=amp, flat_len=len,
    #                                                       ramp_sigma_len=0.001, cutoff_sigma=2, freq=iq_freq + add_freq,
    #                                                       phase=phase))
    #         sequencer.append('charge%s_Q' % setup,
    #                          Square(max_amp=amp, flat_len=len,
    #                                 ramp_sigma_len=0.001, cutoff_sigma=2, freq=iq_freq + add_freq,
    #                                 phase=phase + Q_phase))
    #
    #     elif pulse_type.lower() == 'gauss':
    #         sequencer.append('charge%s_I' % setup, Gauss(max_amp=amp, sigma_len=len,
    #                                                      cutoff_sigma=2, freq=iq_freq + add_freq, phase=phase))
    #         sequencer.append('charge%s_Q' % setup, Gauss(max_amp=amp, sigma_len=len,
    #                                                      cutoff_sigma=2, freq=iq_freq + add_freq,
    #                                                      phase=phase + Q_phase))


    def pi_q(self,sequencer,qb = 0,phase = 0,pulse_type = 'square'):
        setup = self.lattice_cfg["qubit"]["setup"][qb]
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb], flat_len=self.pulse_info[setup]['pi_len'][qb],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=self.pulse_info[setup]['iq_freq'][qb],phase=phase))
            sequencer.append('charge%s_Q' % setup,Square(max_amp=self.pulse_info[setup]['pi_amp'][qb], flat_len= self.pulse_info[setup]['pi_len'][qb],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=self.pulse_info[setup]['iq_freq'][qb],phase=phase+self.pulse_info[setup]['Q_phase'][qb]))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb], sigma_len=self.pulse_info[setup]['pi_len'][qb],cutoff_sigma=2,
                            freq=self.pulse_info[setup]['iq_freq'][qb],phase=phase))
            sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb], sigma_len=self.pulse_info[setup]['pi_len'][qb],cutoff_sigma=2,
                            freq=self.pulse_info[setup]['iq_freq'][qb],phase=phase+self.pulse_info[setup]['Q_phase'][qb]))
        elif pulse_type.lower() == 'wurst':
            sequencer.append('charge%s_I' % setup,
                             WURST20_I(max_amp=self.pulse_info['wurst_amp'][qb],
                                       bandwidth=self.pulse_info['wurst_bw'][qb],
                                       t_length=self.pulse_info['wurst_pulselen'][qb],
                                       freq=self.pulse_info[setup]['iq_freq'][qb],
                                       phase=phase))
            sequencer.append('charge%s_Q' % setup,
                             WURST20_Q(max_amp=self.pulse_info['wurst_amp'][qb],
                                       bandwidth=self.pulse_info['wurst_bw'][qb],
                                       t_length=self.pulse_info['wurst_pulselen'][qb],
                                       freq=self.pulse_info[setup]['iq_freq'][qb],
                                       phase=phase,
                                       phase_t0= self.pulse_info[setup]['Q_phase'][qb]))
        elif pulse_type.lower() == 'bir4_degraaf':
            sequencer.append('charge%s_I' % setup,
                             BIR_4(max_amp=self.pulse_info['BIR4_amp'][qb],
                                       bandwidth=self.pulse_info['BIR4_bw'][qb],
                                       t_length=self.pulse_info['BIR4_pulse_len'][qb],
                                       theta=self.pulse_info['BIR4_theta'][qb],
                                       freq=self.pulse_info[setup]['iq_freq'][qb],
                                       phase=phase))
            sequencer.append('charge%s_Q' % setup,
                             BIR_4(max_amp=self.pulse_info['BIR4_amp'][qb],
                                       bandwidth=self.pulse_info['BIR4_bw'][qb],
                                       t_length=self.pulse_info['BIR4_pulse_len'][qb],
                                       theta=self.pulse_info['BIR4_theta'][qb],
                                       freq=self.pulse_info[setup]['iq_freq'][qb],
                                       phase=phase,
                                       phase_t0= self.pulse_info[setup]['Q_phase'][qb]))
        elif pulse_type.lower() == 'bir4_hyp':
            sequencer.append('charge%s_I' % setup,
                             BIR_4_hyp(max_amp=self.pulse_info['BIR4_hyp_amp'][qb],
                                       bandwidth=self.pulse_info['BIR4_hyp_bw'][qb],
                                       t_length=self.pulse_info['BIR4_hyp_pulse_len'][qb],
                                       theta=self.pulse_info['BIR4_hyp_theta'][qb],
                                       kappa=self.pulse_info['BIR4_hyp_kappa'][qb],
                                       beta=self.pulse_info['BIR4_hyp_beta'][qb],
                                       freq=self.pulse_info[setup]['iq_freq'][qb],
                                       phase=phase))
            sequencer.append('charge%s_Q' % setup,
                             BIR_4_hyp(max_amp=self.pulse_info['BIR4_hyp_amp'][qb],
                                       bandwidth=self.pulse_info['BIR4_hyp_bw'][qb],
                                       t_length=self.pulse_info['BIR4_hyp_pulse_len'][qb],
                                       theta=self.pulse_info['BIR4_hyp_theta'][qb],
                                       kappa=self.pulse_info['BIR4_hyp_kappa'][qb],
                                       beta=self.pulse_info['BIR4_hyp_beta'][qb],
                                       freq=self.pulse_info[setup]['iq_freq'][qb],
                                       phase=phase,
                                       phase_t0= self.pulse_info[setup]['Q_phase'][qb]))
        elif pulse_tpye.lower() == 'knill':
            phase = np.pi / 6
            if self.expt_cfg['pulse_type'] == "square":
                sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb], phase=phase))
                sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb],
                                                              phase=phase + self.pulse_info[setup]['Q_phase'][qb]))
            elif self.expt_cfg['pulse_type'] == "gauss":
                sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase))
                sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                 qb]))
            phase = 0
            if self.expt_cfg['pulse_type'] == "square":
                sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb],
                                                              phase=phase))
                sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb],
                                                              phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                  qb]))
            elif self.expt_cfg['pulse_type'] == "gauss":
                sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase))
                sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                 qb]))
            phase = np.pi / 2
            if self.expt_cfg['pulse_type'] == "square":
                sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb],
                                                              phase=phase))
                sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb],
                                                              phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                  qb]))
            elif self.expt_cfg['pulse_type'] == "gauss":
                sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase))
                sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                 qb]))
            phase = 0
            if self.expt_cfg['pulse_type'] == "square":
                sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb],
                                                              phase=phase))
                sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb],
                                                              phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                  qb]))
            elif self.expt_cfg['pulse_type'] == "gauss":
                sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase))
                sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                 qb]))
            phase = np.pi / 6
            if self.expt_cfg['pulse_type'] == "square":
                sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb],
                                                              phase=phase))
                sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                              flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                              ramp_sigma_len=0.001, cutoff_sigma=2,
                                                              freq=self.pulse_info[setup]['iq_freq'][qb],
                                                              phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                  qb]))
            elif self.expt_cfg['pulse_type'] == "gauss":
                sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase))
                sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                             sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                             cutoff_sigma=2,
                                                             freq=self.pulse_info[setup]['iq_freq'][qb],
                                                             phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                 qb]))

    def half_pi_q(self,sequencer,qb = 0,phase = 0,pulse_type = 'square'):
        setup = self.lattice_cfg["qubit"]["setup"][qb]
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['half_pi_amp'][qb], flat_len=self.pulse_info[setup]['half_pi_len'][qb],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=self.pulse_info[setup]['iq_freq'][qb],phase=phase))
            sequencer.append('charge%s_Q' % setup,Square(max_amp=self.pulse_info[setup]['half_pi_amp'][qb], flat_len= self.pulse_info[setup]['half_pi_len'][qb],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=self.pulse_info[setup]['iq_freq'][qb],phase=phase+self.pulse_info[setup]['Q_phase'][qb]))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['half_pi_amp'][qb], sigma_len=self.pulse_info[setup]['half_pi_len'][qb],cutoff_sigma=2,
                            freq=self.pulse_info[setup]['iq_freq'][qb],phase=phase))
            sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['half_pi_amp'][qb], sigma_len=self.pulse_info[setup]['half_pi_len'][qb],cutoff_sigma=2,
                            freq=self.pulse_info[setup]['iq_freq'][qb],phase=phase+self.pulse_info[setup]['Q_phase'][qb]))

    def half_pi_q_ancilla(self,sequencer,qb = 0,phase = 0,pulse_type = 'square'):
        setup = self.lattice_cfg["qubit"]["setup"][qb]
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['half_pi_amp'][qb], flat_len=self.pulse_info[setup]['half_pi_len'][qb],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=self.pulse_info[setup]['iq_freq'][qb],phase=phase))
            sequencer.append('charge%s_Q' % setup,Square(max_amp=self.pulse_info[setup]['half_pi_amp'][qb], flat_len= self.pulse_info[setup]['half_pi_len'][qb],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=self.pulse_info[setup]['iq_freq'][qb],phase=phase+self.pulse_info[setup]['Q_phase'][qb]))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info_ancilla['half_pi_amp'][qb], sigma_len=self.pulse_info_ancilla['half_pi_len'][qb],cutoff_sigma=2,
                            freq=self.pulse_info_ancilla['iq_freq'][qb],phase=phase))
            sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info_ancilla['half_pi_amp'][qb], sigma_len=self.pulse_info_ancilla['half_pi_len'][qb],cutoff_sigma=2,
                            freq=self.pulse_info_ancilla['iq_freq'][qb],phase=phase+self.pulse_info[setup]['Q_phase'][qb]))

    def pi_q_ancilla(self,sequencer,qb = 0,phase = 0,pulse_type = 'gauss'):
        setup = self.lattice_cfg["qubit"]["setup"][qb]
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['half_pi_amp'][qb], flat_len=self.pulse_info[setup]['half_pi_len'][qb],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=self.pulse_info[setup]['iq_freq'][qb],phase=phase))
            sequencer.append('charge%s_Q' % setup,Square(max_amp=self.pulse_info[setup]['half_pi_amp'][qb], flat_len= self.pulse_info[setup]['half_pi_len'][qb],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=self.pulse_info[setup]['iq_freq'][qb],phase=phase+self.pulse_info[setup]['Q_phase'][qb]))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info_ancilla['pi_amp'][qb], sigma_len=self.pulse_info_ancilla['pi_len'][qb],cutoff_sigma=2,
                            freq=self.pulse_info_ancilla['iq_freq'][qb],phase=phase))
            sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info_ancilla['pi_amp'][qb], sigma_len=self.pulse_info_ancilla['pi_len'][qb],cutoff_sigma=2,
                            freq=self.pulse_info_ancilla['iq_freq'][qb],phase=phase+self.pulse_info[setup]['Q_phase'][qb]))

    def pi_q_ef(self,sequencer,qb = 0,phase = 0,pulse_type = 'square'):
        setup = self.lattice_cfg["qubit"]["setup"][qb]
        freq = self.pulse_info[setup]['iq_freq'][qb] + self.lattice_cfg['qubit']['anharmonicity'][qb]
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_ef_amp'][qb], flat_len=self.pulse_info[setup]['pi_ef_len'][qb],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=freq,phase=phase))
            sequencer.append('charge%s_Q' % setup,Square(max_amp=self.pulse_info[setup]['pi_ef_amp'][qb], flat_len= self.pulse_info[setup]['pi_ef_len'][qb],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=freq,phase=phase+self.pulse_info[setup]['Q_phase'][qb]))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_ef_amp'][qb], sigma_len=self.pulse_info[setup]['pi_ef_len'][qb],cutoff_sigma=2,
                            freq=freq,phase=phase))
            sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_ef_amp'][qb], sigma_len=self.pulse_info[setup]['pi_ef_len'][qb],cutoff_sigma=2,
                            freq=freq,phase=phase+self.pulse_info[setup]['Q_phase'][qb]))

    def half_pi_q_ef(self,sequencer,qb = 0,phase = 0,pulse_type = 'square'):
        setup = self.lattice_cfg["qubit"]["setup"][qb]
        freq = self.pulse_info[setup]['iq_freq'][qb] + self.lattice_cfg['qubit']['anharmonicity'][qb]
        if pulse_type.lower() == 'square':
            sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['half_pi_ef_amp'][qb], flat_len=self.pulse_info[setup]['half_pi_ef_len'][qb],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=freq,phase=phase))
            sequencer.append('charge%s_Q' % setup,Square(max_amp=self.pulse_info[setup]['half_pi_ef_amp'][qb], flat_len= self.pulse_info[setup]['half_pi_ef_len'][qb],ramp_sigma_len=0.001, cutoff_sigma=2,
                            freq=freq,phase=phase+self.pulse_info[setup]['Q_phase'][qb]))
        elif pulse_type.lower() == 'gauss':
            sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['half_pi_ef_amp'][qb], sigma_len=self.pulse_info[setup]['half_pi_ef_len'][qb],cutoff_sigma=2,
                            freq=freq,phase=phase))
            sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['half_pi_ef_amp'][qb], sigma_len=self.pulse_info[setup]['half_pi_ef_len'][qb],cutoff_sigma=2,
                            freq=freq,phase=phase+self.pulse_info[setup]['Q_phase'][qb]))
    def idle_q(self,sequencer,qb = None,time=0):
        if qb==None:
            qb=self.on_qbs[0]
        setup = self.lattice_cfg["qubit"]["setup"][qb]
        sequencer.append('charge%s_I' % setup, Idle(time=time))
        sequencer.append('charge%s_Q' % setup, Idle(time=time))

    #def return_ff_pulse(self, qb, flux, ff_len, pulse_type,amp, freq=0, flip_amp=False, modulate_qb=[]):
    def return_ff_pulse(self, qb, flux, ff_len, pulse_type, freq=0, flip_amp=False, modulate_qb=[], mod_amp=0, mod_pulse=True):
        if flip_amp:
            flux = -flux

        if qb in modulate_qb and mod_pulse:
            if pulse_type == "lin_ramp_rev":
                pulse = linear_ramp_rev(max_amp = flux, flat_len=ff_len[qb], ramp1_len = self.lattice_cfg['ff_info']['ff_lin_ramp1'][qb],
                                    ramp2_len= self.lattice_cfg['ff_info']['ff_lin_ramp2'][qb], freq=freq, phase=0, mod_freq_times = self.expt_cfg["mod_freq"][qb])
            if pulse_type == "rexp":
                # mod_during_ramp should be a qubit specific list property
                if not isinstance(self.expt_cfg["mod_during_ramp"], list): # make it a list if its a number
                    modRamp = copy.deepcopy(self.expt_cfg["mod_during_ramp"])
                    self.expt_cfg["mod_during_ramp"] = [modRamp]*8
                if not "mod_during_ramp" in self.expt_cfg.keys():
                    self.expt_cfg["mod_during_ramp"]==[False]*8
                if self.expt_cfg["mod_during_ramp"][qb]==True:
                    pulse = reversability_ramp_modulate_all(max_amp=flux,
                                                        exp_ramp_len=self.lattice_cfg['ff_info']['ff_exp_ramp_len'][qb],
                                                        flat_len=ff_len[qb],
                                                        tau_ramp=self.lattice_cfg['ff_info']['ff_exp_ramp_tau'][qb],
                                                        ramp2_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][
                                                            qb],
                                                        cutoff_sigma=2, freq=freq, phase=0, modulate_freq=self.expt_cfg[
                            "mod_freq"][qb], modulate_amp=self.expt_cfg["mod_amp"][qb], modulate_phase=self.expt_cfg[
                            "mod_phase"][qb])
                else:
                    pulse = reversability_ramp_modulate(max_amp=flux,
                                                        exp_ramp_len=self.lattice_cfg['ff_info']['ff_exp_ramp_len'][qb],
                                                        flat_len=ff_len[qb],
                                                        tau_ramp=self.lattice_cfg['ff_info']['ff_exp_ramp_tau'][qb],
                                                        ramp2_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][
                                                            qb],
                                                        cutoff_sigma=2, freq=freq, phase=0, modulate_freq=self.expt_cfg[
                            "mod_freq"][qb], modulate_amp=self.expt_cfg["mod_amp"][qb], modulate_phase=self.expt_cfg[
                            "mod_phase"][qb])

            if pulse_type == "exp":
                # mod_during_ramp should be a qubit specific list property
                if not isinstance(self.expt_cfg["mod_during_ramp"], list):  # make it a list if its a number
                    modRamp = copy.deepcopy(self.expt_cfg["mod_during_ramp"])
                    self.expt_cfg["mod_during_ramp"] = [modRamp] * 8
                if not "mod_during_ramp" in self.expt_cfg.keys():
                    self.expt_cfg["mod_during_ramp"]==[False]*8
                if self.expt_cfg["mod_during_ramp"][qb] == True:
                    pulse = exp_ramp_modulate_all(max_amp=flux,
                                              exp_ramp_len=self.lattice_cfg['ff_info']['ff_exp_ramp_len'][qb],
                                              flat_len=ff_len[qb],
                                              tau_ramp=self.lattice_cfg['ff_info']['ff_exp_ramp_tau'][qb],
                                              ramp2_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][
                                                  qb],
                                              cutoff_sigma=2, freq=freq, phase=0, modulate_freq=self.expt_cfg[
                            "mod_freq"][qb], modulate_amp=self.expt_cfg["mod_amp"][qb], modulate_phase=self.expt_cfg[
                            "mod_phase"][qb])
                else:
                    pulse = exp_ramp_modulate(max_amp=flux,
                                              exp_ramp_len=self.lattice_cfg['ff_info']['ff_exp_ramp_len'][qb],
                                              flat_len=ff_len[qb],
                                              tau_ramp=self.lattice_cfg['ff_info']['ff_exp_ramp_tau'][qb],
                                              ramp2_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][
                                                  qb],
                                              cutoff_sigma=2, freq=freq, phase=0, modulate_freq=self.expt_cfg[
                            "mod_freq"][qb], modulate_amp=self.expt_cfg["mod_amp"][qb], modulate_phase=self.expt_cfg[
                            "mod_phase"][qb])

            if pulse_type == "square":
                if not "mod_amp_sq" in self.expt_cfg.keys():
                    self.expt_cfg["mod_amp_sq"]=0
                if isinstance(self.expt_cfg["mod_amp_sq"], list):
                    # special case for modulating during quench experiments - applies only if you have the parameter "mod_amp_sq"
                    pulse = Square_modulate(max_amp=flux, flat_len=ff_len[qb],
                                            ramp_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][qb],
                                            cutoff_sigma=2, freq=freq, phase=0, phase_t0=0,
                                            modulate_amp=self.expt_cfg["mod_amp_sq"][qb],
                                            modulate_freq=self.expt_cfg["mod_freq"][qb], modulate_phase=self.expt_cfg["mod_phase"][qb])
                else:
                    # the normal code for square pulse
                    pulse = Square_modulate(max_amp=flux, flat_len=ff_len[qb], ramp_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][qb],
                                            cutoff_sigma=2,freq=freq, phase=0, phase_t0 = 0,
                                            modulate_amp=self.expt_cfg["mod_amp"][qb],
                                            modulate_freq=self.expt_cfg["mod_freq"][qb], modulate_phase=self.expt_cfg["mod_phase"][qb])

            if pulse_type == "rexp_cut":
                pulse = reversability_ramp_cut(max_amp=flux,
                                           exp_ramp_len=self.lattice_cfg['ff_info']['ff_exp_ramp_len'][qb],
                                           flat_len=ff_len[qb],
                                           tau_ramp=self.lattice_cfg['ff_info']['ff_exp_ramp_tau'][qb],
                                           ramp_perc_cut=self.lattice_cfg['ff_info']['ff_adb_ramp_cut'][qb],
                                           cutoff_sigma=2, freq=freq, phase=0)



        else:
            if pulse_type == "square":
                pulse = Square(max_amp=flux, flat_len=ff_len[qb],
                               ramp_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][qb], cutoff_sigma=2,
                               freq=freq,
                               phase=0)

            if pulse_type == "linear":
                pulse = linear_ramp(max_amp=flux, flat_len=ff_len[qb],
                                    ramp1_len=self.lattice_cfg['ff_info']['ff_linear_ramp_len'],
                                    ramp2_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][qb],
                                    cutoff_sigma=2, freq=freq, phase=0)

            if pulse_type == "lin_ramp_rev":
                pulse = linear_ramp_rev(max_amp = flux, flat_len=ff_len[qb], ramp1_len = self.lattice_cfg['ff_info']['ff_lin_ramp1'][qb],
                                    ramp2_len= self.lattice_cfg['ff_info']['ff_lin_ramp2'][qb], freq=freq, phase=0)

            if pulse_type == "adb":
                pulse = adb_ramp(max_amp=flux, flat_len=ff_len[qb],
                                 adb_ramp1_sig=self.lattice_cfg['ff_info']['ff_adb_ramp_sig'][qb],
                                 ramp2_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][qb],
                                 cutoff_sigma=2, freq=freq, phase=0)

            if pulse_type == "exp":
                pulse = exp_ramp(max_amp=flux, exp_ramp_len=self.lattice_cfg['ff_info']['ff_exp_ramp_len'][qb],
                                 flat_len=ff_len[qb],
                                 tau_ramp=self.lattice_cfg['ff_info']['ff_exp_ramp_tau'][qb],
                                 ramp2_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][qb],
                                 cutoff_sigma=2, freq=freq, phase=0)

            if pulse_type == "exp_flip_tau":
                pulse = exp_ramp(max_amp=flux, exp_ramp_len=self.lattice_cfg['ff_info']['ff_exp_ramp_len'][qb],
                                 flat_len=ff_len[qb],
                                 tau_ramp=(-1)*self.lattice_cfg['ff_info']['ff_exp_ramp_tau'][qb],
                                 ramp2_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][qb],
                                 cutoff_sigma=2, freq=freq, phase=0)

            if pulse_type == "mexp":
                pulse = multiexponential_ramp(max_amp=flux,
                                              exp_ramp_len=self.lattice_cfg['ff_info']['ff_exp_ramp_len'][qb],
                                              flat_len=ff_len[qb],
                                              tau_ramp=self.lattice_cfg['ff_info']['ff_exp_ramp_tau'][qb],
                                              ramp2_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][qb],
                                              cutoff_sigma=2, freq=freq, phase=0, multiples=30)

            if pulse_type == "rexp":
                pulse = reversability_ramp(max_amp=flux,
                                           exp_ramp_len=self.lattice_cfg['ff_info']['ff_exp_ramp_len'][qb],
                                           flat_len=ff_len[qb],
                                           tau_ramp=self.lattice_cfg['ff_info']['ff_exp_ramp_tau'][qb],
                                           ramp2_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][qb],
                                           cutoff_sigma=2, freq=freq, phase=0)
            if pulse_type == "rexp_cut":
                pulse = reversability_ramp_cut(max_amp=flux,
                                           exp_ramp_len=self.lattice_cfg['ff_info']['ff_exp_ramp_len'][qb],
                                           flat_len=ff_len[qb],
                                           tau_ramp=self.lattice_cfg['ff_info']['ff_exp_ramp_tau'][qb],
                                           ramp_perc_cut=self.lattice_cfg['ff_info']['ff_adb_ramp_cut'][qb],
                                           cutoff_sigma=2, freq=freq, phase=0)
            if pulse_type == "exp_and_modulate":
                pulse = exp_ramp_and_modulate(max_amp=flux, exp_ramp_len=self.lattice_cfg['ff_info']['ff_exp_modulate_ramp_len'][qb],
                                 flat_len=ff_len[qb],
                                 tau_ramp=self.lattice_cfg['ff_info']['ff_exp_modulate_ramp_tau'][qb],
                                 ramp2_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][qb],
                                 cutoff_sigma=2, amp = mod_amp,freq=freq, phase=0)

            if pulse_type == "square_and_modulate":
                pulse = Square_and_modulate(max_amp=flux, flat_len=ff_len[qb],
                               ramp_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][qb], cutoff_sigma=2,
                               amp = mod_amp,
                               freq= freq,
                               phase=0)


        return pulse

    def ff_pulse(self, sequencer, ff_len, pulse_type, flux_vec, freq=0, flip_amp=False, modulate_qb=[]):
        area_vec = []
        for qb, flux in enumerate(flux_vec):
            pulse = self.return_ff_pulse(qb, flux, ff_len, pulse_type, freq, flip_amp, modulate_qb)
            sequencer.append('ff_Q%s' % qb, pulse)
            area_vec.append(pulse.get_area())
        return np.asarray(area_vec)

    def ff_pulse_vec(self, sequencer, ff_len, pulse_type_vec, flux_vec, freq=0, flip_amp=False, modulate_qb=[]):
        area_vec = []
        for qb, flux in enumerate(flux_vec):
            pulse_type = pulse_type_vec[qb]
            pulse = self.return_ff_pulse(qb, flux, ff_len, pulse_type, freq, flip_amp, modulate_qb)
            sequencer.append('ff_Q%s' % qb, pulse)
            area_vec.append(pulse.get_area())
        return np.asarray(area_vec)

    #def ff_pulse_combined(self, sequencer, ff_len_list, pulse_type_list, flux_vec_list, delay_list,amp_list = [], freq_list = [], flip_amp=False, modulate_qb=[]):
    def ff_pulse_combined(self, sequencer, ff_len_list, pulse_type_list, flux_vec_list, delay_list,
                              freq_list=[0]*7, flip_amp=False, modulate_qb=[], mod_amp_list = [0]*7, mod_pulse_list=True):
        """

        :param sequencer:
        :param ff_len_list: list of ff lens for each pulse, ie if we had three qubits 2pulses: [ [50, 50, 50], [100, 100, 100]]
        :param pulse_type_list: list of pulse types for each pulse, ie if we had 2 pulses: ["gauss", "square"]
        :param flux_vec_list: list of ff vecs for each pulse, ie if 3 qbs 2 pulses [[0.1, 0.2, 0.15], [0.2, 0.4, 0.25]]
        :param delay_list: delay between pulse and t0, ie for two pulses, [0, 500] -> first pulse no delay, second pulse 500ns
        :param freq:
        :param flip_amp:
        :param modulate_qb:
        :return:
        """
        # add a bunch of pulses together on top of each other for some delay
        area_vec = []
        if not isinstance(mod_pulse_list,list): # make it a list if its a number
            mod_pulse_list =[mod_pulse_list]*len(pulse_type_list)# says if a given pulse should be modulated or not

        for qb in range(len(flux_vec_list[0])): #just get nb of qbs/channels
            pulse_list = []
            for ii in range(len(flux_vec_list)): #nb of different pulses we will be doing
                pulse = self.return_ff_pulse(qb, flux_vec_list[ii][qb], ff_len_list[ii], pulse_type_list[ii], freq_list[ii], flip_amp, modulate_qb, mod_amp_list[ii], mod_pulse_list[ii])
                # pulse = self.return_ff_pulse(qb, flux_vec_list[ii][qb], ff_len_list[ii], pulse_type_list[ii], freq_list[ii], flip_amp, modulate_qb, mod_amp_list)
                #pulse = self.return_ff_pulse(qb, flux_vec_list[ii][qb], ff_len_list[ii], pulse_type_list[ii],amp_list[ii][qb], freq_list[ii][qb], flip_amp, modulate_qb)

                pulse_list.append(pulse)
            sequencer.append_composite('ff_Q%s' % qb, pulse_list, delay_list)

    def ff_comp(self, sequencer, area_vec):
        for qb in range(len(self.expt_cfg['ff_vec'])):
            ff_len = area_vec[qb]/self.lattice_cfg['ff_info']['comp_pulse_amp'][qb]
            sequencer.append('ff_Q%s' % qb,
                             Square(max_amp=self.lattice_cfg['ff_info']['comp_pulse_amp'][qb], flat_len=ff_len,
                                    ramp_sigma_len=self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][qb], cutoff_sigma=2, freq=0,
                                    phase=0))

    def ff_square_and_comp(self, sequencer, ff_len, flux_vec):
        area_vec = self.ff_pulse(sequencer, ff_len, pulse_type="square", flux_vec = flux_vec, flip_amp=False)

        # COMPENSATION PULSE
        if self.lattice_cfg["ff_info"]["ff_comp_sym"]:
            self.ff_pulse(sequencer, ff_len, pulse_type="square", flux_vec = flux_vec, flip_amp=True)
        else:
            self.ff_comp(self, sequencer, area_vec)

    def pad_start_pxi(self,sequencer,on_qubits=None, time = 500):
        # Need 500 ns of padding for the sequences to work reliably. Not sure exactly why.
        for channel in self.channels:
            sequencer.append(channel,
                             Square(max_amp=0.0, flat_len= time, ramp_sigma_len=0.001, cutoff_sigma=2, freq=0.0,
                                    phase=0))

    def readout(self, sequencer, on_qubits=None, sideband = False):
        if on_qubits == None:
            on_qubits = self.on_rds

        sequencer.sync_channels_time(self.channels)

        readout_time = sequencer.get_time('digtzr_trig') # Earlies was alazar_trig

        readout_time_5ns_multiple = np.ceil(readout_time / 5) * 5

        sequencer.append_idle_to_time('digtzr_trig', readout_time_5ns_multiple)
        sequencer.sync_channels_time(self.channels)


        for rd in on_qubits:
            setup = self.lattice_cfg["qubit"]["setup"][rd]

            sequencer.append('readout%s' % setup,
                             Square(max_amp=self.lattice_cfg['readout']['amp'][rd],
                                    flat_len=self.lattice_cfg['readout']['length'][rd],
                                    ramp_sigma_len=0.001, cutoff_sigma=2, freq=self.lattice_cfg['readout'][setup]['freq'][rd],
                                    phase=0, phase_t0=readout_time_5ns_multiple))
        sequencer.append('digtzr_trig', Ones(time=self.hardware_cfg['trig_pulse_len']['default']))

        return readout_time

    def readout_pxi(self, sequencer, on_qubits=None, sideband = False, overlap = False, synch_channels=None):
        if on_qubits == None:
            on_qubits = self.on_rds

        if synch_channels==None:
            synch_channels = self.channels
        sequencer.sync_channels_time(synch_channels)

        for rd in on_qubits:
            setup = self.lattice_cfg["readout"]["setup"][rd]
            if "flip_eg" in self.expt_cfg.keys() and self.expt_cfg["flip_eg"]:
                self.pi_q(sequencer, qb=rd, phase=0, pulse_type=self.pulse_info["pulse_type"][rd])
                if synch_channels == None:
                    synch_channels = self.channels
                sequencer.sync_channels_time(synch_channels)

        readout_time = sequencer.get_time('digtzr_trig') # Earlies was alazar_tri
        readout_time_5ns_multiple = np.ceil(readout_time / 5) * 5
        sequencer.append_idle_to_time('digtzr_trig', readout_time_5ns_multiple)
        if overlap:
            pass
        else:
            sequencer.sync_channels_time(synch_channels)

        for rd in on_qubits:
            setup = self.lattice_cfg["readout"]["setup"][rd]
            sequencer.append('readout%s' % setup,
                             Square(max_amp=self.lattice_cfg['readout']['amp'][rd],
                                    flat_len=self.lattice_cfg['readout']['length'][rd],
                                    ramp_sigma_len=0.001, cutoff_sigma=2, freq=0,
                                    phase=0, phase_t0=readout_time_5ns_multiple))
        sequencer.append('digtzr_trig', Ones(time=self.hardware_cfg['trig_pulse_len']['default']))

        return readout_time

    def excited_readout_pxi(self, sequencer, on_qubits=None, sideband = False, overlap = False):
        if on_qubits == None:
            on_qubits = self.on_rds

        sequencer.sync_channels_time(self.channels)
        readout_time = sequencer.get_time('digtzr_trig') # Earlies was alazar_tri
        readout_time_5ns_multiple = np.ceil(readout_time / 5) * 5
        sequencer.append_idle_to_time('digtzr_trig', readout_time_5ns_multiple)
        if overlap:
            pass
        else:
            sequencer.sync_channels_time(self.channels)

        for rd in on_qubits:
            setup = self.lattice_cfg["qubit"]["setup"][rd]
            self.gen_q(sequencer, rd, len=2000, amp=1, phase=0, pulse_type=self.pulse_info['pulse_type'][rd])
            sequencer.append('readout%s' % setup,
                             Square(max_amp=self.lattice_cfg['readout']['amp'][rd],
                                    flat_len=self.lattice_cfg['readout']['length'][rd],
                                    ramp_sigma_len=20, cutoff_sigma=2, freq=0,
                                    phase=0, phase_t0=readout_time_5ns_multiple))

        sequencer.append('digtzr_trig', Ones(time=self.hardware_cfg['trig_pulse_len']['default']))

        return readout_time

    def resonator_spectroscopy(self, sequencer):

        sequencer.new_sequence(self)
        self.readout_pxi(sequencer, self.on_rds)
        sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def resonator_spectroscopy_pi(self, sequencer):
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
        for qb in self.on_qbs:
            pulse_info = self.lattice_cfg["pulse_info"]
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            self.gen_q(sequencer, qb=qb , len=pulse_info[setup]["pi_len"][qb],
                       amp=pulse_info[setup]["pi_amp"][qb], add_freq=0, phase=0,
                       pulse_type=pulse_info["pulse_type"][qb])
            sequencer.sync_channels_time(self.channels)
        self.idle_q(sequencer, time=self.expt_cfg["delay"])
        self.readout_pxi(sequencer, self.on_rds)
        sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def resonator_spectroscopy_ef_pi(self, sequencer):
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
        for qb in self.on_qbs:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            self.pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
            self.pi_q_ef(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
        self.readout_pxi(sequencer, self.on_rds)
        sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def melting_single_readout_full_ramp_2setups(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ##################################GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=100)
            ##############################GENERATE RAMP###########################################
            flux_vec = self.expt_cfg["ff_vec"]
            self.ff_pulse(sequencer, ff_len = [evolution_t]*8, pulse_type = self.expt_cfg["ramp_type"], flux_vec= flux_vec, flip_amp=False)
            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################
            flux_vec = self.expt_cfg["ff_vec"]
            self.ff_pulse(sequencer, [evolution_t]*8, pulse_type=self.expt_cfg["ramp_type"], flux_vec=flux_vec, flip_amp=True)
            sequencer.end_sequence()

        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)
        return sequencer.complete(self, plot=True)


    def melting_doublons_single_readout(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI eg and ef PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                # ============== pi eg
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                # ============== pi ef
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_ef_len"][qb],
                           amp=self.pulse_info[setup]["pi_ef_amp"][qb],
                           pulse_type=self.pulse_info["ef_pulse_type"][qb],
                           add_freq = self.lattice_cfg['qubit']['anharmonicity'][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=100)
            ##############################GENERATE RAMP###########################################
            flux_vec = self.expt_cfg["ff_vec"]
            self.ff_pulse(sequencer, ff_len = [evolution_t]*8, pulse_type = self.expt_cfg["ramp_type"], flux_vec= flux_vec, flip_amp=False)
            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################
            flux_vec = self.expt_cfg["ff_vec"]
            self.ff_pulse(sequencer, [evolution_t]*8, pulse_type=self.expt_cfg["ramp_type"], flux_vec=flux_vec, flip_amp=True)
            sequencer.end_sequence()

        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)
        return sequencer.complete(self, plot=True)

    def melting_reversability_sweep_ramplen(self, sequencer):

        qb_list = self.expt_cfg["Mott_qbs"]
        idle_post_pulse = 100
        wait_post_flux = 500
        for expramplen in self.expt_cfg['ramplenlist']:
            self.lattice_cfg['ff_info']['ff_exp_ramp_len'] = list(np.ones(7) * expramplen)
            self.lattice_cfg['ff_info']['ff_exp_ramp_tau'] = list(np.ones(7) * expramplen * (self.expt_cfg["tau_factor"] / 5))

            for  evolution_t in range(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

                ##################################GENERATE PI PULSES ################################################
                #pi 1, 3, 5
                for i, qb in enumerate(qb_list):
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                               amp=self.pulse_info[setup]["pi_amp"][qb],
                               pulse_type=self.pulse_info["pulse_type"][qb])
                    self.idle_q(sequencer, time=20)
                    sequencer.sync_channels_time(self.channels)

                self.idle_q(sequencer, time=idle_post_pulse)
                sequencer.sync_channels_time(self.channels)

                ##############################GENERATE RAMP - snap qbs to lattice, then adb ramp in others ###########################################
                ff_vec_list = self.expt_cfg["ff_vec_list"]
                delay_list = self.expt_cfg["delay_list"]
                ff_len_list = self.expt_cfg["ff_len_list"]
                ff_len_list[-1] =  [evolution_t] * 7  # ONE HARDCODED THING
                self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
                                       flux_vec_list = ff_vec_list, delay_list = delay_list, freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"])

                sequencer.sync_channels_time(self.channels)
                ############################## readout ###########################################
                sequencer.sync_channels_time(self.channels)
                self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
                sequencer.sync_channels_time(self.channels)
                self.readout_pxi(sequencer, self.on_rds, overlap=False)
                sequencer.sync_channels_time(self.channels)

                ############################## generate compensation ###########################################

                self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                       flux_vec_list=ff_vec_list, delay_list=delay_list, freq_list=[0]*7, flip_amp=True, modulate_qb=self.expt_cfg["modulate_qb"])
                sequencer.end_sequence()

        ############################## pi cal  ###########################################
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs=[rd_qb], pi_rds=self.on_rds)

        return sequencer.complete(self, plot=True)

    # def melting_reversability_shake_odds_sweep_ramplen(self, sequencer):
    #
    #     qb_list = self.expt_cfg["Mott_qbs"]
    #     idle_post_pulse = 100
    #     wait_post_flux = 500
    #     for expramplen in self.expt_cfg['ramplenlist']:
    #         self.lattice_cfg['ff_info']['ff_exp_ramp_len'] = list(np.ones(7) * expramplen)
    #         self.lattice_cfg['ff_info']['ff_exp_ramp_tau'] = list(np.ones(7) * (1) * expramplen * (self.expt_cfg["tau_factor"] / 5))
    #
    #         for  evolution_t in range(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
    #             sequencer.new_sequence(self)
    #             self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
    #
    #             ##################################GENERATE PI PULSES ################################################
    #             #pi 1, 3, 5
    #             for i, qb in enumerate(qb_list):
    #                 setup = self.lattice_cfg["qubit"]["setup"][qb]
    #                 self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
    #                            amp=self.pulse_info[setup]["pi_amp"][qb],
    #                            pulse_type=self.pulse_info["pulse_type"][qb])
    #                 self.idle_q(sequencer, time=20)
    #                 sequencer.sync_channels_time(self.channels)
    #
    #             self.idle_q(sequencer, time=idle_post_pulse)
    #             sequencer.sync_channels_time(self.channels)
    #
    #             ##############################GENERATE RAMP - snap qbs to lattice, then adb ramp in others ###########################################
    #             ff_vec_list = self.expt_cfg["ff_vec_list"]
    #             delay_list = self.expt_cfg["delay_list"]
    #             ff_len_list = self.expt_cfg["ff_len_list"]
    #             ff_ac_amp_list = self.expt_cfg['ff_ac_amp_list']
    #             ff_freq_list = self.expt_cfg['ff_freq_list']
    #             ff_len_list[2] =  [evolution_t] * 7  # ONE HARDCODED THING
    #             # delay_list[3] += evolution_t
    #             # delay_list[4] += evolution_t + expramplen
    #             self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
    #                                    flux_vec_list = ff_vec_list, delay_list = delay_list,mod_amp_list = ff_ac_amp_list, freq_list=ff_freq_list, flip_amp=False)
    #
    #             sequencer.sync_channels_time(self.channels)
    #             ############################## readout ###########################################
    #             sequencer.sync_channels_time(self.channels)
    #             self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
    #             sequencer.sync_channels_time(self.channels)
    #             self.readout_pxi(sequencer, self.on_rds, overlap=False)
    #             sequencer.sync_channels_time(self.channels)
    #
    #             ############################## generate compensation ###########################################
    #             self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
    #                                    flux_vec_list = ff_vec_list, delay_list = delay_list,mod_amp_list = ff_ac_amp_list, freq_list=ff_freq_list, flip_amp=True)
    #
    #             # self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
    #             #                        flux_vec_list=ff_vec_list, delay_list=delay_list, freq=0, flip_amp=True)
    #             sequencer.end_sequence()
    #
    #     ############################## pi cal  ###########################################
    #     if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
    #         for rd_qb in self.on_rds:
    #             self.pi_cal_sequence(sequencer, pi_qbs=[rd_qb], pi_rds=self.on_rds)
    #
    #     return sequencer.complete(self, plot=True)


    ## Take each FF vec in series, no more adding
    def melting_reversability_shake_odds_sweep_ramplen(self, sequencer):

        qb_list = self.expt_cfg["Mott_qbs"]
        idle_post_pulse = 100
        wait_post_flux = 500
        for expramplen in self.expt_cfg['ramplenlist']:
            self.lattice_cfg['ff_info']['ff_exp_ramp_len'] = list(np.ones(7) * expramplen)
            self.lattice_cfg['ff_info']['ff_exp_ramp_tau'] = list(np.ones(7) * (1) * expramplen * (self.expt_cfg["tau_factor"] / 5))

            for evolution_t in range(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"],
                                     self.expt_cfg["evolution_t_step"]):
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)

                ##################################GENERATE PI PULSES ################################################
                # pi 1, 3, 5
                for i, qb in enumerate(qb_list):
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                               amp=self.pulse_info[setup]["pi_amp"][qb],
                               pulse_type=self.pulse_info["pulse_type"][qb])
                    self.idle_q(sequencer, time=20)
                    sequencer.sync_channels_time(self.channels)

                self.idle_q(sequencer, time=idle_post_pulse)
                sequencer.sync_channels_time(self.channels)

                ##############################GENERATE RAMP - snap qbs to lattice, then adb ramp in others ###########################################
                ff_vec_list = copy.deepcopy(self.expt_cfg["ff_vec_list"])
                delay_list = copy.deepcopy(self.expt_cfg["delay_list"])
                ff_len_list = copy.deepcopy(self.expt_cfg["ff_len_list"])
                ff_len_list[2] = [evolution_t] * 7  # ONE HARDCODED THING

                delay_list[2] += expramplen
                delay_list[3] += expramplen + evolution_t #evolution_t
                delay_list[4] += 2*expramplen + evolution_t#2*expramplen #evolution_t + expramplen
                self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list,
                                       pulse_type_list=self.expt_cfg["pulse_type_list"],
                                       flux_vec_list=ff_vec_list, delay_list=delay_list, freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"])
                ## Square wave, exponential ramp, "square wave" constant resonant ramp delay for t_evol, then reverse the first two back out
                # self.return_ff_pulse()


                sequencer.sync_channels_time(self.channels)
                ############################## readout ###########################################
                sequencer.sync_channels_time(self.channels)
                self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
                sequencer.sync_channels_time(self.channels)
                self.readout_pxi(sequencer, self.on_rds, overlap=False)
                sequencer.sync_channels_time(self.channels)

                ############################## generate compensation ###########################################
                self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list,
                                       pulse_type_list=self.expt_cfg["pulse_type_list"],
                                       flux_vec_list=ff_vec_list, delay_list=delay_list, freq_list=[0]*7, flip_amp=True, modulate_qb=self.expt_cfg["modulate_qb"])

                # self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                #                        flux_vec_list=ff_vec_list, delay_list=delay_list, freq=0, flip_amp=True)
                sequencer.end_sequence()

        ############################## pi cal  ###########################################
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs=[rd_qb], pi_rds=self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_reversability_sweep_ff_vec_list(self, sequencer):

        qb_list = self.expt_cfg["Mott_qbs"]
        idle_post_pulse = 100
        wait_post_flux = 500
        # for expramplen in self.expt_cfg['ramplenlist']:
        expramplen = self.expt_cfg['ramplen']
        for ff_vec_list in self.expt_cfg['ff_vec_list_list']:
            self.lattice_cfg['ff_info']['ff_exp_ramp_len'] = list(np.ones(7) * expramplen)
            self.lattice_cfg['ff_info']['ff_exp_ramp_tau'] = list(np.ones(7) * expramplen * (self.expt_cfg["tau_factor"] / 5))

            for  evolution_t in range(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

                ##################################GENERATE PI PULSES ################################################
                #pi 1, 3, 5
                for i, qb in enumerate(qb_list):
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                               amp=self.pulse_info[setup]["pi_amp"][qb],
                               pulse_type=self.pulse_info["pulse_type"][qb])
                    self.idle_q(sequencer, time=20)
                    sequencer.sync_channels_time(self.channels)

                self.idle_q(sequencer, time=idle_post_pulse)
                sequencer.sync_channels_time(self.channels)

                ##############################GENERATE RAMP - snap qbs to lattice, then adb ramp in others ###########################################
                # ff_vec_list = self.expt_cfg["ff_vec_list"]
                delay_list = self.expt_cfg["delay_list"]
                ff_len_list = self.expt_cfg["ff_len_list"]
                ff_len_list[-1] =  [evolution_t] * 7  # ONE HARDCODED THING
                self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
                                       flux_vec_list = ff_vec_list, delay_list = delay_list, freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"])

                sequencer.sync_channels_time(self.channels)
                ############################## readout ###########################################
                sequencer.sync_channels_time(self.channels)
                self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
                sequencer.sync_channels_time(self.channels)
                self.readout_pxi(sequencer, self.on_rds, overlap=False)
                sequencer.sync_channels_time(self.channels)

                ############################## generate compensation ###########################################

                self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                       flux_vec_list=ff_vec_list, delay_list=delay_list, freq_list=[0]*7, flip_amp=True, modulate_qb=self.expt_cfg["modulate_qb"])
                sequencer.end_sequence()

        ############################## pi cal  ###########################################
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs=[rd_qb], pi_rds=self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_reversability_sweep_ramplen_fq5test(self, sequencer):

        # qb_list = self.expt_cfg["Mott_qbs"]
        idle_post_pulse = 100
        wait_post_flux = 500
        for expramplen in self.expt_cfg['ramplenlist']:
            self.lattice_cfg['ff_info']['ff_exp_ramp_len'] = list(np.ones(7) * expramplen)
            self.lattice_cfg['ff_info']['ff_exp_ramp_tau'] = list(np.ones(7) * expramplen * (self.expt_cfg["tau_factor"] / 5))

            for  evolution_t in range(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

                ##################################GENERATE PI PULSES ################################################
                #pi 1, 3, 5, ef(pi)
                for i, qb in enumerate([5]):
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                               amp=self.pulse_info[setup]["pi_amp"][qb],
                               pulse_type=self.pulse_info["pulse_type"][qb])
                    self.idle_q(sequencer, time=20)
                    sequencer.sync_channels_time(self.channels)

                for i, qb in enumerate([5]):
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.pi_q_ef(sequencer, qb=qb,pulse_type=self.pulse_info["pulse_type"][qb])
                    self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

                for i, qb in enumerate([1,3]):
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                               amp=self.pulse_info[setup]["pi_amp"][qb],
                               pulse_type=self.pulse_info["pulse_type"][qb])
                    self.idle_q(sequencer, time=20)
                    sequencer.sync_channels_time(self.channels)

                self.idle_q(sequencer, time=idle_post_pulse)
                sequencer.sync_channels_time(self.channels)

                ##############################GENERATE RAMP - snap qbs to lattice, then adb ramp in others ###########################################
                # ff_vec_list = self.expt_cfg["ff_vec_list"]
                # delay_list = self.expt_cfg["delay_list"]
                # ff_len_list = self.expt_cfg["ff_len_list"]
                # freq_list = self.expt_cfg['ff_freq_list']
                # amp_list = self.expt_cfg['ff_ac_amp_list']
                # ff_len_list[-1] =  [evolution_t] * 7  # ONE HARDCODED THING
                ff_vec_list = copy.deepcopy(self.expt_cfg["ff_vec_list"])
                delay_list = copy.deepcopy(self.expt_cfg["delay_list"])
                ff_len_list = copy.deepcopy(self.expt_cfg["ff_len_list"])
                ff_ac_amp_list = copy.deepcopy(self.expt_cfg['ff_ac_amp_list'])
                ff_freq_list = copy.deepcopy(self.expt_cfg['ff_freq_list'])
                ff_len_list[2] = [evolution_t] * 7  # ONE HARDCODED THING

                delay_list[2] += expramplen
                delay_list[3] += expramplen + evolution_t  # evolution_t
                delay_list[4] += 2 * expramplen + evolution_t  # 2*expramplen #evolution_t + expramplen
                self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
                                       flux_vec_list = ff_vec_list, delay_list = delay_list, freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"])

                sequencer.sync_channels_time(self.channels)
                ############################## readout ###########################################
                sequencer.sync_channels_time(self.channels)
                self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
                sequencer.sync_channels_time(self.channels)
                self.readout_pxi(sequencer, self.on_rds, overlap=False)
                sequencer.sync_channels_time(self.channels)

                ############################## generate compensation ###########################################
                self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list,
                                       pulse_type_list=self.expt_cfg["pulse_type_list"],
                                       flux_vec_list=ff_vec_list, delay_list=delay_list, freq_list=[0]*7, flip_amp=True, modulate_qb=self.expt_cfg["modulate_qb"])
                # self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                #                        flux_vec_list=ff_vec_list, delay_list=delay_list, freq=0, flip_amp=True)
                sequencer.end_sequence()

        ############################## pi cal  ###########################################
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs=[rd_qb], pi_rds=self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_wurst(self, sequencer):

        # qb_list = self.expt_cfg["Mott_qbs"]
        idle_post_pulse = 100
        wait_post_flux = 500
        for expramplen in self.expt_cfg['ramplenlist']:
            self.lattice_cfg['ff_info']['ff_exp_ramp_len'] = list(np.ones(7) * expramplen)
            self.lattice_cfg['ff_info']['ff_exp_ramp_tau'] = list(np.ones(7) * expramplen * (self.expt_cfg["tau_factor"] / 5))

            for  evolution_t in range(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

                ##################################GENERATE PI PULSES (will be outside flux)################################################
                sequencer.sync_channels_time(self.channels)
                for i, qb in enumerate(self.expt_cfg["Mott_qbs"]):
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                               amp=self.pulse_info[setup]["pi_amp"][qb],
                               pulse_type=self.pulse_info["pulse_type"][qb])
                    self.idle_q(sequencer, time=20)

                ##################################GENERATE WURST PULSES (will be inside flux)################################################
                sequencer.sync_channels_time(self.channels)
                self.idle_q(sequencer, time=self.expt_cfg["wurst_delay"])
                channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
                sequencer.sync_channels_time(channels_excluding_fluxch)

                qb = self.expt_cfg["wurst_qb"]
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])

                ##############################GENERATE RAMP - snap qbs to lattice, then adb ramp in others ###########################################
                ff_vec_list = self.expt_cfg["ff_vec_list"]
                delay_list = self.expt_cfg["delay_list"]
                ff_len_list = self.expt_cfg["ff_len_list"]
                ff_len_list[-1] =  [evolution_t] * 7  # ONE HARDCODED THING
                self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
                                       flux_vec_list = ff_vec_list, delay_list = delay_list, freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"])

                sequencer.sync_channels_time(self.channels)
                ############################## readout ###########################################
                sequencer.sync_channels_time(self.channels)
                self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
                sequencer.sync_channels_time(self.channels)
                self.readout_pxi(sequencer, self.on_rds, overlap=False)
                sequencer.sync_channels_time(self.channels)

                ############################## generate compensation ###########################################

                self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                       flux_vec_list=ff_vec_list, delay_list=delay_list, freq_list=[0]*7, flip_amp=True, modulate_qb=self.expt_cfg["modulate_qb"])
                sequencer.end_sequence()

        ############################## pi cal  ###########################################
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs=[rd_qb], pi_rds=self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_mb_ramsey(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        ram_qb = self.expt_cfg["ram_qb"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ##################################GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            self.half_pi_q(sequencer, ram_qb, pulse_type=self.pulse_info['pulse_type'][ram_qb])
            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)

            ##############################GENERATE RAMP###########################################
            flux_vec = self.expt_cfg["ff_vec"]
            self.ff_pulse(sequencer, ff_len = [evolution_t]*8, pulse_type = self.expt_cfg["ramp_type"], flux_vec= flux_vec, flip_amp=False, modulate_qb=self.expt_cfg["mod_qb"])

            ############################## ramsey_pulse ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            self.half_pi_q(sequencer, ram_qb, pulse_type=self.pulse_info['pulse_type'][ram_qb],
                           phase=2 * np.pi * evolution_t * self.expt_cfg['ramsey_freq'])

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################
            flux_vec = self.expt_cfg["ff_vec"]
            self.ff_pulse(sequencer, [evolution_t]*8, pulse_type=self.expt_cfg["ramp_type"], flux_vec=flux_vec, flip_amp=True)
            sequencer.end_sequence()

        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)
        return sequencer.complete(self, plot=True)

    def melting_mb_ramsey_superposition(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        sup_qbs = self.expt_cfg["sup_qbs"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ##################################GENERATE SUP ################################################
            self.pi_q(sequencer, sup_qbs[0], pulse_type=self.pulse_info['pulse_type'][sup_qbs[0]])
            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)
            flux_vec = self.expt_cfg["ff_vec_sup"]
            self.ff_pulse(sequencer, ff_len = [self.expt_cfg["sup_idle"]]*8, pulse_type = "square", flux_vec= flux_vec, flip_amp=False)
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)

            ##################################GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            ##############################GENERATE RAMP###########################################
            flux_vec = self.expt_cfg["ff_vec"]
            self.ff_pulse(sequencer, ff_len = [evolution_t]*8, pulse_type = self.expt_cfg["ramp_type"], flux_vec= flux_vec, flip_amp=False)

            ############################## ramsey_pulse ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)

            for ii, qb in enumerate(sup_qbs):
                self.half_pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb],phase=2 * np.pi * evolution_t * self.expt_cfg['ramsey_freq'][ii] )

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################
            flux_vec = self.expt_cfg["ff_vec"]
            self.ff_pulse(sequencer, [evolution_t]*8, pulse_type=self.expt_cfg["ramp_type"], flux_vec=flux_vec, flip_amp=True)
            flux_vec = self.expt_cfg["ff_vec_sup"]
            self.ff_pulse(sequencer, ff_len=[self.expt_cfg["sup_idle"]] * 8, pulse_type="square", flux_vec=flux_vec,
                          flip_amp=True)
            sequencer.end_sequence()

        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            self.twoqb_cal_sequence(sequencer, pi_qbs=self.on_rds, pi_rds=self.on_rds)
        return sequencer.complete(self, plot=True)

    def melting_mb_ramsey_superposition_combo_flux(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        sup_qbs = self.expt_cfg["sup_qbs"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ##################################GENERATE SUP ################################################
            self.pi_q(sequencer, sup_qbs[0], pulse_type=self.pulse_info['pulse_type'][sup_qbs[0]])
            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)
            flux_vec = self.expt_cfg["ff_vec_sup"]
            self.ff_pulse(sequencer, ff_len = [self.expt_cfg["sup_idle"]]*8, pulse_type = "square", flux_vec= flux_vec, flip_amp=False)
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)

            ##################################GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            ##############################GENERATE RAMP###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            ff_len_list = copy.deepcopy(self.expt_cfg["ff_len_list"])
            ff_len_list[-1] = [ff_len_list[-1][ii] + evolution_t for ii in range(len(ff_len_list[-1]))]
            self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
                                   flux_vec_list = ff_vec_list, delay_list = self.expt_cfg["delay_list"], freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["mod_qb"], mod_amp_list = self.expt_cfg["mod_amp"])

            ############################## ramsey_pulse ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)

            for ii, qb in enumerate(sup_qbs):
                self.half_pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb],phase=2 * np.pi * evolution_t * self.expt_cfg['ramsey_freq'][ii] )

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################
            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=self.expt_cfg["delay_list"], freq_list=[0] * 7,
                                   flip_amp=True, modulate_qb=self.expt_cfg["mod_qb"],
                                   mod_amp_list=self.expt_cfg["mod_amp"])
            flux_vec = self.expt_cfg["ff_vec_sup"]
            self.ff_pulse(sequencer, ff_len=[self.expt_cfg["sup_idle"]] * 8, pulse_type="square", flux_vec=flux_vec,
                          flip_amp=True)
            sequencer.end_sequence()

        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            self.twoqb_cal_sequence(sequencer, pi_qbs=self.on_rds, pi_rds=self.on_rds)
        return sequencer.complete(self, plot=True)


    def melting_cond_prep(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        iq_freq = self.expt_cfg["iq_freq"]
        qb_amp = self.expt_cfg["qb_amp"]
        qb_len = self.expt_cfg["qb_len"]
        cond_qbs = self.expt_cfg["cond_qbs"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
            channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]

            ##################################GENERATE COND ################################################
            sequencer.sync_channels_time(self.channels)
            flux_vec = self.expt_cfg["ff_vec_cond"]
            ff_len = sum(np.array(qb_len))*4+100
            self.ff_pulse(sequencer, ff_len = [ff_len]*8, pulse_type = "square", flux_vec= flux_vec, flip_amp=False)
            for qq, qb in enumerate(cond_qbs):
                self.gen_q(sequencer, qb=qb, len=qb_len[qq],
                           amp=qb_amp[qq],
                           pulse_type=self.pulse_info["pulse_type"][qb], iq_overwrite=iq_freq[qq])
                sequencer.sync_channels_time(self.channels)

            sequencer.sync_channels_time(channels_excluding_fluxch)

            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)

            ##################################GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            ##############################GENERATE RAMP###########################################
            flux_vec = self.expt_cfg["ff_vec"]
            self.ff_pulse(sequencer, ff_len = [evolution_t]*8, pulse_type = self.expt_cfg["ramp_type"], flux_vec= flux_vec, flip_amp=False)

            ############################## ramsey_pulse ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)

            # for ii, qb in enumerate(sup_qbs):
            #     self.half_pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb],phase=2 * np.pi * evolution_t * self.expt_cfg['ramsey_freq'][ii] )

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################
            flux_vec = self.expt_cfg["ff_vec"]
            self.ff_pulse(sequencer, [evolution_t]*8, pulse_type=self.expt_cfg["ramp_type"], flux_vec=flux_vec, flip_amp=True)
            flux_vec = self.expt_cfg["ff_vec_cond"]
            self.ff_pulse(sequencer, ff_len=[ff_len] * 8, pulse_type="square", flux_vec=flux_vec,
                          flip_amp=True)
            sequencer.end_sequence()

        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            self.twoqb_cal_sequence(sequencer, pi_qbs=self.on_rds, pi_rds=self.on_rds)
        return sequencer.complete(self, plot=True)

    def melting_small_lat(self, sequencer):
        lattice_og = copy.deepcopy(self.lattice_cfg)

        if self.expt_cfg["shift"]:
            self.lattice_cfg["pulse_info"]["A"]["pi_len"] = self.lattice_cfg["smol_lat"]["pi_len_shift"]
            self.lattice_cfg["pulse_info"]["B"]["pi_len"] = self.lattice_cfg["smol_lat"]["pi_len_shift"]
            self.lattice_cfg['pulse_info']["A"]['pi_amp'] = self.lattice_cfg["smol_lat"]["pi_amp_shift"]
            self.lattice_cfg['pulse_info']["B"]['pi_amp'] = self.lattice_cfg["smol_lat"]["pi_amp_shift"]
            self.lattice_cfg["qubit"]["freq"] = self.lattice_cfg["smol_lat"]["qb_freq_shift"]
            self.lattice_cfg["pulse_info"]['A']["iq_freq"] = self.lattice_cfg["smol_lat"]["iq_freq_shift"]
            self.lattice_cfg["pulse_info"]['B']["iq_freq"] = self.lattice_cfg["smol_lat"]["iq_freq_shift"]
            self.lattice_cfg["pulse_info"]['B']["Q_phase"] = self.lattice_cfg["smol_lat"]["Q_phase"]
            self.lattice_cfg["pulse_info"]['A']["Q_phase"] = self.lattice_cfg["smol_lat"]["Q_phase"]
        else:
            self.lattice_cfg["pulse_info"]["A"]["pi_len"] = self.lattice_cfg["smol_lat"]["pi_len"]
            self.lattice_cfg["pulse_info"]["B"]["pi_len"] = self.lattice_cfg["smol_lat"]["pi_len"]
            self.lattice_cfg['pulse_info']["A"]['pi_amp'] = self.lattice_cfg["smol_lat"]["pi_amp"]
            self.lattice_cfg['pulse_info']["B"]['pi_amp'] = self.lattice_cfg["smol_lat"]["pi_amp"]
            self.lattice_cfg["qubit"]["freq"] = self.lattice_cfg["smol_lat"]["qb_freq"]
            self.lattice_cfg["pulse_info"]['A']["iq_freq"] =  self.lattice_cfg["smol_lat"]["iq_freq"]
            self.lattice_cfg["pulse_info"]['B']["iq_freq"] =  self.lattice_cfg["smol_lat"]["iq_freq"]
            self.lattice_cfg["pulse_info"]['B']["Q_phase"] = self.lattice_cfg["smol_lat"]["Q_phase"]
            self.lattice_cfg["pulse_info"]['A']["Q_phase"] = self.lattice_cfg["smol_lat"]["Q_phase"]

        # qb_list = self.expt_cfg["Mott_qbs"]
        for expramplen in self.expt_cfg['ramplenlist']:
            self.lattice_cfg['ff_info']['ff_exp_ramp_len'] = list(np.ones(7) * expramplen)
            self.lattice_cfg['ff_info']['ff_exp_ramp_tau'] = list(np.ones(7) * expramplen * (self.expt_cfg["tau_factor"] / 5))

            for  evolution_t in range(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

                ##################################GENERATE pi PULSES (will be inside flux)################################################
                sequencer.sync_channels_time(self.channels)
                self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
                channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
                sequencer.sync_channels_time(channels_excluding_fluxch)

                for qb in self.expt_cfg["Mott_qbs"]:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                               amp=self.pulse_info[setup]["pi_amp"][qb],
                               pulse_type=self.pulse_info["pulse_type"][qb])
                    self.idle_q(sequencer, qb=qb, time=20)
                    sequencer.sync_channels_time(channels_excluding_fluxch)
                self.idle_q(sequencer, qb=qb, time=25)
                sequencer.sync_channels_time(channels_excluding_fluxch)

                ##############################GENERATE RAMP - snap qbs to lattice, then adb ramp in others ###########################################
                ff_vec_list = self.expt_cfg["ff_vec_list"]
                delay_list = self.expt_cfg["delay_list"]
                ff_len_list = copy.deepcopy(self.expt_cfg["ff_len_list"])
                ff_len_list[-1] = [ff_len_list[-1][ii] + evolution_t for ii in range(len(ff_len_list[-1]))]  # ONE HARDCODED THING
                self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
                                       flux_vec_list = ff_vec_list, delay_list = delay_list, freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"])

                sequencer.sync_channels_time(self.channels)
                ############################## readout ###########################################
                sequencer.sync_channels_time(self.channels)
                self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
                sequencer.sync_channels_time(self.channels)
                self.readout_pxi(sequencer, self.on_rds, overlap=False)
                sequencer.sync_channels_time(self.channels)

                ############################## generate compensation ###########################################

                self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                       flux_vec_list=ff_vec_list, delay_list=delay_list, freq_list=[0]*7, flip_amp=True, modulate_qb=self.expt_cfg["modulate_qb"])
                sequencer.end_sequence()

        ############################## pi cal  ###########################################
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            self.lattice_cfg = copy.deepcopy(lattice_og)
            self.pulse_info = copy.deepcopy(lattice_og['pulse_info'])

            if self.expt_cfg["twoqb_calibration"] == True:
                self.twoqb_cal_sequence(sequencer, pi_qbs=self.on_rds, pi_rds=self.on_rds)
            else:
                for rd_qb in self.on_rds:
                    self.pi_cal_sequence(sequencer, pi_qbs=[rd_qb], pi_rds=self.on_rds)




        return sequencer.complete(self, plot=True)

    def melting_small_lat_tomo(self, sequencer):
        lattice_og = copy.deepcopy(self.lattice_cfg)

        if self.expt_cfg["shift"]:
            self.lattice_cfg["pulse_info"]["A"]["pi_len"] = self.lattice_cfg["smol_lat"]["pi_len_shift"]
            self.lattice_cfg["pulse_info"]["B"]["pi_len"] = self.lattice_cfg["smol_lat"]["pi_len_shift"]
            self.lattice_cfg['pulse_info']["A"]['pi_amp'] = self.lattice_cfg["smol_lat"]["pi_amp_shift"]
            self.lattice_cfg['pulse_info']["B"]['pi_amp'] = self.lattice_cfg["smol_lat"]["pi_amp_shift"]
            self.lattice_cfg["qubit"]["freq"] = self.lattice_cfg["smol_lat"]["qb_freq_shift"]
            self.lattice_cfg["pulse_info"]['A']["iq_freq"] = self.lattice_cfg["smol_lat"]["iq_freq_shift"]
            self.lattice_cfg["pulse_info"]['B']["iq_freq"] = self.lattice_cfg["smol_lat"]["iq_freq_shift"]
            self.lattice_cfg["pulse_info"]['B']["Q_phase"] = [self.lattice_cfg["smol_lat"]["Q_phase_evens"]] * 7
        else:
            self.lattice_cfg["pulse_info"]["A"]["pi_len"] = self.lattice_cfg["smol_lat"]["pi_len"]
            self.lattice_cfg["pulse_info"]["B"]["pi_len"] = self.lattice_cfg["smol_lat"]["pi_len"]
            self.lattice_cfg['pulse_info']["A"]['pi_amp'] = self.lattice_cfg["smol_lat"]["pi_amp"]
            self.lattice_cfg['pulse_info']["B"]['pi_amp'] = self.lattice_cfg["smol_lat"]["pi_amp"]
            self.lattice_cfg["qubit"]["freq"] = self.lattice_cfg["smol_lat"]["qb_freq"]
            self.lattice_cfg["pulse_info"]['A']["iq_freq"] =  self.lattice_cfg["smol_lat"]["iq_freq"]
            self.lattice_cfg["pulse_info"]['B']["iq_freq"] =  self.lattice_cfg["smol_lat"]["iq_freq"]
            self.lattice_cfg["pulse_info"]['B']["Q_phase"] = [self.lattice_cfg["smol_lat"]["Q_phase_evens"]] * 7
        FF_vec_orig = copy.copy(self.expt_cfg["ff_vec_list"][-1])

        # qb_list = self.expt_cfg["Mott_qbs"]
        for expramplen in self.expt_cfg['ramplenlist']:
            self.lattice_cfg['ff_info']['ff_exp_ramp_len'] = list(np.ones(7) * expramplen)
            self.lattice_cfg['ff_info']['ff_exp_ramp_tau'] = list(np.ones(7) * expramplen * (self.expt_cfg["tau_factor"] / 5))

            for det in self.expt_cfg["detfactor_list"]:
                self.expt_cfg["ff_vec_list"][-1] = list(np.array(copy.deepcopy(FF_vec_orig)) * det)
                for gate in ["X", "Y", "Z"]:
                    sequencer.new_sequence(self)
                    self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

                    ##################################GENERATE pi PULSES (will be inside flux)################################################
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
                    channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
                    sequencer.sync_channels_time(channels_excluding_fluxch)

                    for qb in self.expt_cfg["Mott_qbs"]:
                        setup = self.lattice_cfg["qubit"]["setup"][qb]
                        self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                                   amp=self.pulse_info[setup]["pi_amp"][qb],
                                   pulse_type=self.pulse_info["pulse_type"][qb])
                        self.idle_q(sequencer, qb=qb, time=20)
                        sequencer.sync_channels_time(channels_excluding_fluxch)
                    self.idle_q(sequencer, qb=qb, time=25)
                    sequencer.sync_channels_time(channels_excluding_fluxch)

                    ##############################GENERATE RAMP - snap qbs to lattice, then adb ramp in others ###########################################
                    ff_vec_list = self.expt_cfg["ff_vec_list"]
                    delay_list = self.expt_cfg["delay_list"]
                    ff_len_list = copy.deepcopy(self.expt_cfg["ff_len_list"])
                    self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
                                           flux_vec_list = ff_vec_list, delay_list = delay_list, freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"])

                    sequencer.sync_channels_time(self.channels)
                    ############################## readout ###########################################
                    sequencer.sync_channels_time(self.channels)
                    self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
                    sequencer.sync_channels_time(self.channels)

                    # Tomography rotation (adjusting pulses so that they are at the large stagger)
                    for qb in self.on_rds:
                        setup = self.lattice_cfg["qubit"]["setup"][qb]
                        if gate == "X":
                            self.gen_q(sequencer, qb=qb, len=lattice_og["pulse_info"][setup]["half_pi_len"][qb],
                                       amp=lattice_og["pulse_info"][setup]['Y90_amp'][qb], phase=np.pi / 2,
                                       pulse_type=lattice_og["pulse_info"]['pulse_type'][qb],
                                       iq_overwrite = lattice_og["pulse_info"][setup]["iq_freq"][qb],
                                       Q_phase_overwrite = lattice_og["pulse_info"][setup]["Q_phase"][qb] )
                        elif gate == "Y":
                            self.gen_q(sequencer, qb=qb, len=lattice_og["pulse_info"][setup]["half_pi_len"][qb],
                                       amp=lattice_og["pulse_info"][setup]['X90_amp'][qb], phase=0,
                                       pulse_type=lattice_og["pulse_info"]['pulse_type'][qb],
                                       iq_overwrite=lattice_og["pulse_info"][setup]["iq_freq"][qb],
                                       Q_phase_overwrite=lattice_og["pulse_info"][setup]["Q_phase"][qb])
                        elif gate == "Z":
                            pass
                    self.readout_pxi(sequencer, self.on_rds, overlap=False)
                    sequencer.sync_channels_time(self.channels)

                    ############################## generate compensation ###########################################

                    self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                           flux_vec_list=ff_vec_list, delay_list=delay_list, freq_list=[0]*7, flip_amp=True, modulate_qb=self.expt_cfg["modulate_qb"])
                    sequencer.end_sequence()

        ############################## pi cal  ###########################################
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            self.lattice_cfg = copy.deepcopy(lattice_og)
            self.pulse_info = copy.deepcopy(lattice_og['pulse_info'])

            if self.expt_cfg["twoqb_calibration"] == True:
                self.twoqb_cal_sequence(sequencer, pi_qbs=self.on_rds, pi_rds=self.on_rds)
            else:
                for rd_qb in self.on_rds:
                    self.pi_cal_sequence(sequencer, pi_qbs=[rd_qb], pi_rds=self.on_rds)




        return sequencer.complete(self, plot=True)

    def melting_single_readout_combo_flux(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ##################################GENERATE PI PULSES - Pi Q4################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if "ef_qbs" in self.expt_cfg.keys():
                for i, qb in enumerate(self.expt_cfg["ef_qbs"]):
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.pi_q_ef(sequencer, qb=qb, phase=0, pulse_type=self.pulse_info["ef_pulse_type"][qb])
                    self.idle_q(sequencer, time=20)
                    sequencer.sync_channels_time(self.channels)

            self.idle_q(sequencer, time=100)
            sequencer.sync_channels_time(self.channels)

            ##############################GENERATE RAMP  ramp to lattice###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            ff_len_list = copy.deepcopy(self.expt_cfg["ff_len_list"])
            ff_len_list[-1] = [ff_len_list[-1][ii] + evolution_t for ii in range(len(ff_len_list[-1]))]
            self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
                                   flux_vec_list = ff_vec_list, delay_list = self.expt_cfg["delay_list"], freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"])

            sequencer.sync_channels_time(self.channels)

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=self.expt_cfg["delay_list"], freq_list=[0]*7,
                                   flip_amp=True, modulate_qb=self.expt_cfg["modulate_qb"])

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_single_readout_combo_flux_mxg(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ##################################GENERATE PI PULSES - Pi Q4################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if "ef_qbs" in self.expt_cfg.keys():
                for i, qb in enumerate(self.expt_cfg["ef_qbs"]):
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.pi_q_ef(sequencer, qb=qb, phase=0, pulse_type=self.pulse_info["ef_pulse_type"][qb])
                    self.idle_q(sequencer, time=20)
                    sequencer.sync_channels_time(self.channels)

            self.idle_q(sequencer, time=100)
            sequencer.sync_channels_time(self.channels)

            ##############################GENERATE RAMP  ramp to lattice###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            ff_len_list = copy.deepcopy(self.expt_cfg["ff_len_list"])
            ff_len_list[-1] = [ff_len_list[-1][ii] + evolution_t for ii in range(len(ff_len_list[-1]))]
            self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
                                   flux_vec_list = ff_vec_list, delay_list = self.expt_cfg["delay_list"], freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"])

            sequencer.append('stab_I', Square(max_amp=0, flat_len=self.expt_cfg["delay_list"][-1]+self.lattice_cfg['ff_info']['ff_exp_ramp_len'][0],
                                              ramp_sigma_len=0.001, cutoff_sigma=2, freq=0,
                                              phase=0))
            sequencer.append('stab_I', Square(max_amp=1, flat_len=evolution_t,
                                              ramp_sigma_len=0.001, cutoff_sigma=2, freq=0,
                                              phase=0))
            sequencer.sync_channels_time(self.channels)

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=self.expt_cfg["delay_list"], freq_list=[0]*7,
                                   flip_amp=True, modulate_qb=self.expt_cfg["modulate_qb"])

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_mb_ramsey_combo_flux(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        ram_qb = self.expt_cfg["ram_qb"]
        ef = self.expt_cfg["ef"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ##################################GENERATE PI PULSES - Pi Q4################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if ef:
                self.half_pi_q_ef(sequencer, ram_qb, pulse_type=self.pulse_info['ef_pulse_type'][ram_qb])
            else:
                self.half_pi_q(sequencer, ram_qb, pulse_type=self.pulse_info['pulse_type'][ram_qb])
            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)

            self.idle_q(sequencer, time=100)
            sequencer.sync_channels_time(self.channels)

            ##############################GENERATE RAMP ###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            ff_len_list = copy.deepcopy(self.expt_cfg["ff_len_list"])
            ff_len_list[-1] = [ff_len_list[-1][ii] + evolution_t for ii in range(len(ff_len_list[-1]))]
            self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
                                   flux_vec_list = ff_vec_list, delay_list = self.expt_cfg["delay_list"], freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["mod_qb"], mod_amp_list = self.expt_cfg["mod_amp"])

            sequencer.sync_channels_time(self.channels)

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            if ef:
                self.half_pi_q_ef(sequencer, ram_qb, pulse_type=self.pulse_info['ef_pulse_type'][ram_qb],
                               phase=2 * np.pi * evolution_t * self.expt_cfg['ramsey_freq'])
            else:
                if "ram_rd_qb" in self.expt_cfg.keys():
                    ram_rd_qb = self.expt_cfg["ram_rd_qb"]
                else:
                    ram_rd_qb = ram_qb
                self.half_pi_q(sequencer, ram_rd_qb, pulse_type=self.pulse_info['pulse_type'][ram_rd_qb],
                               phase=2 * np.pi * evolution_t * self.expt_cfg['ramsey_freq'])

            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=self.expt_cfg["delay_list"], freq_list=[0]*7,
                                   flip_amp=True, modulate_qb=self.expt_cfg["modulate_qb"])

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_mb_ramsey_combo_flux_uband(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        ram_qb = self.expt_cfg["ram_qb"]
        filled_qbs = self.expt_cfg["filled_qbs"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ##################################GENERATE PI PULSES - Pi Q4################################################
            for i, qb in enumerate(filled_qbs):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.pi_q_ef(sequencer, qb, pulse_type=self.pulse_info['ef_pulse_type'][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            self.half_pi_q_ef(sequencer, ram_qb, pulse_type=self.pulse_info['ef_pulse_type'][ram_qb])
            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)

            self.idle_q(sequencer, time=100)
            sequencer.sync_channels_time(self.channels)

            ##############################GENERATE RAMP ###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            ff_len_list = copy.deepcopy(self.expt_cfg["ff_len_list"])
            ff_len_list[-1] = [ff_len_list[-1][ii] + evolution_t for ii in range(len(ff_len_list[-1]))]
            self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
                                   flux_vec_list = ff_vec_list, delay_list = self.expt_cfg["delay_list"], freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["mod_qb"], mod_amp_list = self.expt_cfg["mod_amp"])

            sequencer.sync_channels_time(self.channels)

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)

            if "ram_rd_qb" in self.expt_cfg.keys():
                ram_rd_qb = self.expt_cfg["ram_rd_qb"]
            else:
                ram_rd_qb = ram_qb
            self.half_pi_q_ef(sequencer, ram_qb, pulse_type=self.pulse_info['ef_pulse_type'][ram_qb],
                              phase=2 * np.pi * evolution_t * self.expt_cfg['ramsey_freq'])

            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=self.expt_cfg["delay_list"], freq_list=[0]*7,
                                   flip_amp=True, modulate_qb=self.expt_cfg["modulate_qb"])

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)


    def melting_mb_ramsey_combo_flux_linramp(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        ram_qb = self.expt_cfg["ram_qb"]
        ef = self.expt_cfg["ef"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ##################################GENERATE PI PULSES - Pi Q4################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if ef:
                self.half_pi_q_ef(sequencer, ram_qb, pulse_type=self.pulse_info['ef_pulse_type'][ram_qb])
            else:
                self.half_pi_q(sequencer, ram_qb, pulse_type=self.pulse_info['pulse_type'][ram_qb])
            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)

            self.idle_q(sequencer, time=100)
            sequencer.sync_channels_time(self.channels)

            ##############################GENERATE RAMP ###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            # last two pulses are getting evol time growing
            ff_len_list = copy.deepcopy(self.expt_cfg["ff_len_list"])
            ff_len_list[-1] = [ff_len_list[-1][ii] + evolution_t for ii in range(len(ff_len_list[-1]))]
            ff_len_list[-2] = [ff_len_list[-2][ii] + evolution_t for ii in range(len(ff_len_list[-2]))]
            self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
                                   flux_vec_list = ff_vec_list, delay_list = self.expt_cfg["delay_list"], freq_list=[0]*7,
                                   flip_amp=False, modulate_qb=self.expt_cfg["mod_qb"], mod_amp_list = self.expt_cfg["mod_amp"], mod_pulse_list=self.expt_cfg["mod_pulse_list"])

            sequencer.sync_channels_time(self.channels)

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            if ef:
                self.half_pi_q_ef(sequencer, ram_qb, pulse_type=self.pulse_info['ef_pulse_type'][ram_qb],
                               phase=2 * np.pi * evolution_t * self.expt_cfg['ramsey_freq'])
            else:
                if "ram_rd_qb" in self.expt_cfg.keys():
                    ram_rd_qb = self.expt_cfg["ram_rd_qb"]
                else:
                    ram_rd_qb = self.expt_cfg["ram_qb"]
                self.half_pi_q(sequencer, ram_rd_qb, pulse_type=self.pulse_info['pulse_type'][ram_rd_qb],
                               phase=2 * np.pi * evolution_t * self.expt_cfg['ramsey_freq'])
            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=self.expt_cfg["delay_list"], freq_list=[0]*7,
                                   flip_amp=True, modulate_qb=self.expt_cfg["mod_qb"],mod_amp_list=self.expt_cfg["mod_amp"], mod_pulse_list=self.expt_cfg["mod_pulse_list"])

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_combo_flux_linramp(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ##################################GENERATE PI PULSES - Pi Q4################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)

            self.idle_q(sequencer, time=100)
            sequencer.sync_channels_time(self.channels)

            ##############################GENERATE RAMP ###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            # last two pulses are getting evol time growing
            ff_len_list = copy.deepcopy(self.expt_cfg["ff_len_list"])
            ff_len_list[-1] = [ff_len_list[-1][ii] + evolution_t for ii in range(len(ff_len_list[-1]))]
            ff_len_list[-2] = [ff_len_list[-2][ii] + evolution_t for ii in range(len(ff_len_list[-2]))]
            self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
                                   flux_vec_list = ff_vec_list, delay_list = self.expt_cfg["delay_list"], freq_list=[0]*7,
                                   flip_amp=False, modulate_qb=self.expt_cfg["mod_qb"], mod_amp_list = self.expt_cfg["mod_amp"], mod_pulse_list=self.expt_cfg["mod_pulse_list"])

            sequencer.sync_channels_time(self.channels)

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)

            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=self.expt_cfg["delay_list"], freq_list=[0]*7,
                                   flip_amp=True, modulate_qb=self.expt_cfg["mod_qb"],mod_amp_list=self.expt_cfg["mod_amp"], mod_pulse_list=self.expt_cfg["mod_pulse_list"])

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)


    def melting_JJinterferometry_single_readout(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            self.idle_q(sequencer, time=100)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs MELT + QUENCH ###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            expramplen = self.lattice_cfg['ff_info']['ff_exp_ramp_len'][0]
            t_wait = self.expt_cfg["t_wait"]
            t_coupling = self.expt_cfg["t_coupling"]
            N_coupling = self.expt_cfg["N_coupling"]

            # if we turn on the JJ coupling twice
            if N_coupling ==2:
                ff_len_list = [[t_wait + N_coupling * t_coupling + evolution_t]*8, \
                               [t_coupling]*8,
                               [evolution_t]*8,
                               [t_coupling]*8]
                delay_list = [0, expramplen + t_wait,
                              expramplen + t_wait + t_coupling,
                              expramplen + t_wait + t_coupling + evolution_t]

            # if we turn on the JJ coupling only once
            if N_coupling ==1:
                ff_len_list = [[t_wait + N_coupling * t_coupling + evolution_t] * 8, \
                               [t_coupling] * 8,
                               [evolution_t] * 8]
                delay_list = [0, expramplen + t_wait,
                              expramplen + t_wait + t_coupling]

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"])

            sequencer.sync_channels_time(self.channels)

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0]*7,flip_amp=True, modulate_qb=self.expt_cfg["modulate_qb"])

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_mb_NOON_ramseyv3(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        X90_qb = self.expt_cfg["X90_qb"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb]) # generate ancilla X90 gate
            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs MELT + QUENCH ###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            ff_len_list = [[ff_len + (step<4) * evolution_t]*8 for step, ff_len  in enumerate(self.expt_cfg["ff_len_list"])]
            delay_list = [d + (dd>3)*evolution_t for dd, d in enumerate(self.expt_cfg["delay_list"])]

            # we are not modulating but we still need to set these vectors to have the right dimensions...
            # freq_list = [[0]*7]*7
            # ff_ac_amp_list = [[0]*7]*7
            # self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
            #                        flux_vec_list = ff_vec_list, delay_list = self.expt_cfg["delay_list"], freq_list = freq_list, amp_list=ff_ac_amp_list, flip_amp=False)
            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list = self.expt_cfg["mod_amp"])

            ############################## second Ramsey pulse ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb],
                               phase=2 * np.pi * evolution_t * self.expt_cfg['ramsey_freq'])

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,freq_list=[0]*7,flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list = self.expt_cfg["mod_amp"])

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_mb_NOON_ramseyv2(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        X90_qb = self.expt_cfg["X90_qb"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb]) # generate ancilla X90 gate
                self.idle_q(sequencer, qb = X90_qb, time=20)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs MELT + QUENCH ###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            ff_len_list = [[ff_len + (step<3) *evolution_t]*8 for step, ff_len  in enumerate(self.expt_cfg["ff_len_list"])]
            delay_list = [d + (dd>2)*evolution_t for dd, d in enumerate(self.expt_cfg["delay_list"])]

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list = self.expt_cfg["mod_amp"])

            ############################## second Ramsey pulse ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb],
                               phase=2 * np.pi *evolution_t * self.expt_cfg['ramsey_freq'])

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,freq_list=[0]*7,flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list = self.expt_cfg["mod_amp"])

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_mb_NOON_echo(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        X90_qb = self.expt_cfg["X90_qb"]
        echo_qb = self.expt_cfg["echo_qb"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb]) # generate ancilla X90 gate
                self.idle_q(sequencer, qb = X90_qb, time=20)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs MELT + QUENCH ###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            ff_len_list1 = [[ff_len ] * 8 for step, ff_len in enumerate(self.expt_cfg["ff_len_list"])]
            delay_list1 = [d for dd, d in enumerate(self.expt_cfg["delay_list"])]
            ff_len_list2 = [[ff_len + (step<3) *evolution_t]*8 for step, ff_len  in enumerate(self.expt_cfg["ff_len_list"])]
            delay_list2 = [d + (dd>2)*evolution_t for dd, d in enumerate(self.expt_cfg["delay_list"])]

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list1, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list1,
                                   freq_list=[0] * 7, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=self.expt_cfg["mod_amp"])

            ################### echo Pi pulse ##################
            self.idle_q(sequencer, qb=X90_qb, time=25)
            sequencer.sync_channels_time(self.channels)
            
            if echo_qb is not None:
                self.idle_q(sequencer, qb=echo_qb, time=25)
                sequencer.sync_channels_time(self.channels)

                setup = self.lattice_cfg["qubit"]["setup"][echo_qb]
                self.gen_q(sequencer, qb=echo_qb, len=self.pulse_info[setup]["pi_len"][echo_qb],
                           amp=self.pulse_info[setup]["pi_amp"][echo_qb],
                           pulse_type=self.pulse_info["pulse_type"][echo_qb])

                self.idle_q(sequencer, qb=echo_qb, time=25)
                sequencer.sync_channels_time(self.channels)

            self.idle_q(sequencer, qb=X90_qb, time=25)
            sequencer.sync_channels_time(self.channels)

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list2, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list2,
                                   freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list = self.expt_cfg["mod_amp"])

            ############################## second Ramsey pulse ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb],
                               phase=2 * np.pi *evolution_t * self.expt_cfg['ramsey_freq'])

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list1, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list1,freq_list=[0]*7,flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list = self.expt_cfg["mod_amp"])

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list2,
                                   pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list2, freq_list=[0] * 7, flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=self.expt_cfg["mod_amp"])

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def mb_NOON_ramsey_echov2(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        X90_qb = self.expt_cfg["X90_qb"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"],
                                     self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb,
                               pulse_type=self.pulse_info['pulse_type'][X90_qb])  # generate ancilla X90 gate
            self.idle_q(sequencer, qb = X90_qb, time=20)
            sequencer.sync_channels_time(self.channels)

            ############################## second Ramsey pulse ###########################################
            waitTime = 2 * self.expt_cfg["ramplen"] + evolution_t + 10
            self.idle_q(sequencer, qb = X90_qb, time=waitTime)
            channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
            sequencer.sync_channels_time(channels_excluding_fluxch)

            if X90_qb is not None:
                self.pi_q_ancilla(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb])
                sequencer.sync_channels_time(channels_excluding_fluxch)

            ############################## GENERATE RAMPs MELT + QUENCH ###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            ff_len_list = [[ff_len + (step < 3) * (2*evolution_t + self.expt_cfg['echo_time'])] * 8 for step, ff_len in
                           enumerate(self.expt_cfg["ff_len_list"])]
            delay_list = [d + (dd > 2) * (2*evolution_t + self.expt_cfg['echo_time']) for dd, d in enumerate(self.expt_cfg["delay_list"])]

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * 7, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"],
                                   mod_amp_list=self.expt_cfg["mod_amp"])

            ############################## second Ramsey pulse ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb],
                               phase=2 * np.pi * 2*evolution_t * self.expt_cfg['ramsey_freq'])

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list, freq_list=[0] * 7, flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=self.expt_cfg["mod_amp"])

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs=[rd_qb], pi_rds=self.on_rds)

        return sequencer.complete(self, plot=True)

    def mb_NOON_ramseyModv2(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        X90_qb = self.expt_cfg["X90_qb"]
        lattice_og = copy.deepcopy(self.lattice_cfg)

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            ################################## GENERATE PI/2 PULSES inside FLUX ################################################
            self.idle_q(sequencer, time=200)
            channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
            sequencer.sync_channels_time(channels_excluding_fluxch)

            if X90_qb is not None:
                self.half_pi_q_ancilla(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb]) # generate ancilla X90 gate
            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(channels_excluding_fluxch)

            ########################## second pi/2 pulse ##########################
            waitTime = 4*self.expt_cfg["ramplen"] + 2*self.expt_cfg['mod_piLen'] + evolution_t + 300
            self.idle_q(sequencer, time=waitTime)
            sequencer.sync_channels_time(channels_excluding_fluxch)

            if X90_qb is not None:
                self.half_pi_q_ancilla(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb],
                               phase=2 * np.pi * evolution_t * self.expt_cfg['ramsey_freq'])
            sequencer.sync_channels_time(channels_excluding_fluxch)

            ############################## GENERATE RAMPs MELT + QUENCH ###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            # delay_list = self.expt_cfg["delay_list"]
            # delay_list[3] = delay_list[3] + evolution_t

            ff_len_list = [[ff_len + (step==2) * evolution_t]*8 for step, ff_len  in enumerate(self.expt_cfg["ff_len_list"])]
            delay_list = [d + (dd==3)*evolution_t for dd, d in enumerate(self.expt_cfg["delay_list"])]

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * 7, flip_amp=False,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=self.expt_cfg["mod_amp"],
                                   mod_pulse_list=self.expt_cfg["mod_pulse_list"])

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            # self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
            #                        flux_vec_list=ff_vec_list, delay_list=delay_list,freq_list=[0]*7,flip_amp=True,
            #                        modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list = self.expt_cfg["mod_amp"])

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * 7, flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=self.expt_cfg["mod_amp"],
                                   mod_pulse_list=self.expt_cfg["mod_pulse_list"])

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()

        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def mb_NOON_ramseyMod_singlegate(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        X90_qb = self.expt_cfg["X90_qb"]
        X90_qb_after = self.expt_cfg["X90_qb_after"]
        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb]) # generate ancilla X90 gate
                self.idle_q(sequencer, qb = X90_qb, time=50)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs MELT + QUENCH ###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            delay_list = self.expt_cfg["delay_list"]
            # delay_list[3] = delay_list[3] + evolution_t

            ff_len_list = [[ff_len + (step==2) * evolution_t]*8 for step, ff_len  in enumerate(self.expt_cfg["ff_len_list"])]
            # delay_list = [d + (dd==3)*evolution_t for dd, d in enumerate(self.expt_cfg["delay_list"])]

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * 7, flip_amp=False,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=self.expt_cfg["mod_amp"],
                                   mod_pulse_list=self.expt_cfg["mod_pulse_list"])

            ############################## second Ramsey pulse ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            # self.idle_q(sequencer, time=100)
            sequencer.sync_channels_time(self.channels)
            if X90_qb_after is not None:
                self.half_pi_q(sequencer, X90_qb_after, pulse_type=self.pulse_info['pulse_type'][X90_qb_after],
                               phase=2 * np.pi * evolution_t * self.expt_cfg['ramsey_freq'])

            # sequencer.sync_channels_time(self.channels)
            # self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux']-100)

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            # self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
            #                        flux_vec_list=ff_vec_list, delay_list=delay_list,freq_list=[0]*7,flip_amp=True,
            #                        modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list = self.expt_cfg["mod_amp"])

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * 7, flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=self.expt_cfg["mod_amp"],
                                   mod_pulse_list=self.expt_cfg["mod_pulse_list"])

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def mb_NOON_ramseyMod_echo(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        X90_qb = self.expt_cfg["X90_qb"]
        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb]) # generate ancilla X90 gate
                self.idle_q(sequencer, qb = X90_qb, time=50)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs MELT + QUENCH ###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            delay_list = self.expt_cfg["delay_list"]
            # delay_list[3] = delay_list[3] + evolution_t

            ff_len_list = [[ff_len]*8 for step, ff_len  in enumerate(self.expt_cfg["ff_len_list"])]
            # delay_list = [d + (dd==3)*evolution_t for dd, d in enumerate(self.expt_cfg["delay_list"])]

            ############################ FIRST GATE ###############################
            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * 7, flip_amp=False,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=self.expt_cfg["mod_amp"],
                                   mod_pulse_list=self.expt_cfg["mod_pulse_list"])

            ############################ RAMSEY IDLE TIMES ###############################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, qb=X90_qb, time=50)
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, qb=X90_qb, time=evolution_t)

            ################### echo Pi pulse ##################
            if X90_qb is not None:
                setup = self.lattice_cfg["qubit"]["setup"][X90_qb]
                self.gen_q(sequencer, qb=X90_qb, len=self.pulse_info[setup]["pi_len"][X90_qb],
                           amp=self.pulse_info[setup]["pi_amp"][X90_qb],
                           pulse_type=self.pulse_info["pulse_type"][X90_qb])
                # self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            # self.idle_q(sequencer, qb=X90_qb, time=2*evolution_t)
            # sequencer.sync_channels_time(self.channels)
            # 
            # ################### echo Pi pulse ##################
            # if X90_qb is not None:
            #     setup = self.lattice_cfg["qubit"]["setup"][X90_qb]
            #     self.gen_q(sequencer, qb=X90_qb, len=self.pulse_info[setup]["pi_len"][X90_qb],
            #                amp=self.pulse_info[setup]["pi_amp"][X90_qb],
            #                pulse_type=self.pulse_info["pulse_type"][X90_qb])
            #     # self.idle_q(sequencer, time=20)
            #     sequencer.sync_channels_time(self.channels)

            self.idle_q(sequencer, qb=X90_qb, time=evolution_t)
            sequencer.sync_channels_time(self.channels)
            # self.idle_q(sequencer, qb=X90_qb, time=50)
            # sequencer.sync_channels_time(self.channels)

            ############################ SECOND GATE ###############################
            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * 7, flip_amp=False,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=self.expt_cfg["mod_amp"],
                                   mod_pulse_list=self.expt_cfg["mod_pulse_list"])

            ############################## second Ramsey pulse ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb],
                               phase=2 * np.pi * evolution_t * self.expt_cfg['ramsey_freq'])

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * 7, flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=self.expt_cfg["mod_amp"],
                                   mod_pulse_list=self.expt_cfg["mod_pulse_list"])
            sequencer.sync_channels_time(self.channels)

            self.idle_q(sequencer, time= 2*evolution_t)
            sequencer.sync_channels_time(self.channels)

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * 7, flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=self.expt_cfg["mod_amp"],
                                   mod_pulse_list=self.expt_cfg["mod_pulse_list"])
            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()

        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def mb_NOON_ramsey_echo(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        echo_list = self.expt_cfg["Echo_qbs"]
        X90_qb = self.expt_cfg["X90_qb"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb]) # generate ancilla X90 gate
                self.idle_q(sequencer, qb = X90_qb, time=50)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs MELT + QUENCH ###########################################
            ff_vec_list1 = self.expt_cfg["ff_vec_list_gate"]
            delay_list1 = self.expt_cfg["delay_list_gate"]
            ff_len_list1 = [[ff_len]*8 for step, ff_len  in enumerate(self.expt_cfg["ff_len_list_gate"])]

            ff_vec_list2 = self.expt_cfg["ff_vec_list_gateundo"]
            delay_list2 = self.expt_cfg["delay_list_gateundo"]
            ff_len_list2 = [[ff_len] * 8 for step, ff_len in enumerate(self.expt_cfg["ff_len_list_gateundo"])]

            ############################ FIRST GATE ###############################
            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list1, pulse_type_list=self.expt_cfg["pulse_type_list_gate"],
                                   flux_vec_list=ff_vec_list1, delay_list=delay_list1,
                                   freq_list=[0] * 7, flip_amp=False,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=self.expt_cfg["mod_amp"],
                                   mod_pulse_list=self.expt_cfg["mod_pulse_list"])

            ############################ RAMSEY IDLE TIMES ###############################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, qb=X90_qb, time=50)
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, qb=X90_qb, time=evolution_t)

            ################### echo Pi pulse ##################
            if X90_qb is not None:
                setup = self.lattice_cfg["qubit"]["setup"][X90_qb]
                self.gen_q(sequencer, qb=X90_qb, len=self.pulse_info[setup]["pi_len"][X90_qb],
                           amp=self.pulse_info[setup]["pi_amp"][X90_qb],
                           pulse_type=self.pulse_info["pulse_type"][X90_qb])
                # self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            for i, qb in enumerate(echo_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=10)
                sequencer.sync_channels_time(self.channels)

            self.idle_q(sequencer, qb=X90_qb, time=evolution_t)
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, qb=X90_qb, time=50)
            sequencer.sync_channels_time(self.channels)

            ############################ SECOND GATE ###############################
            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list2, pulse_type_list=self.expt_cfg["pulse_type_list_gateundo"],
                                   flux_vec_list=ff_vec_list2, delay_list=delay_list2,
                                   freq_list=[0] * 7, flip_amp=False,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=self.expt_cfg["mod_amp"],
                                   mod_pulse_list=self.expt_cfg["mod_pulse_list"])

            ############################## second Ramsey pulse ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, qb=X90_qb, time=50)
            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb],
                               phase=2 * np.pi * evolution_t * self.expt_cfg['ramsey_freq'])
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux']-50)

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list1, pulse_type_list=self.expt_cfg["pulse_type_list_gate"],
                                   flux_vec_list=ff_vec_list1, delay_list=delay_list1,
                                   freq_list=[0] * 7, flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=self.expt_cfg["mod_amp"],
                                   mod_pulse_list=self.expt_cfg["mod_pulse_list"])
            sequencer.sync_channels_time(self.channels)

            self.idle_q(sequencer, time= 2*evolution_t)
            sequencer.sync_channels_time(self.channels)

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list2, pulse_type_list=self.expt_cfg["pulse_type_list_gateundo"],
                                   flux_vec_list=ff_vec_list2, delay_list=delay_list2,
                                   freq_list=[0] * 7, flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=self.expt_cfg["mod_amp"],
                                   mod_pulse_list=self.expt_cfg["mod_pulse_list"])
            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()

        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def mb_NOON_ramseyMod_repeat_echo(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        X90_qb = self.expt_cfg["X90_qb"]
        num_gates = self.expt_cfg["num_gates"]
        echo_qb = self.expt_cfg["echo_qb"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb]) # generate ancilla X90 gate
                self.idle_q(sequencer, qb = X90_qb, time=50)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs before echo ###########################################
            ff_vec = self.expt_cfg["ff_vec_list"]  # [snap_vect, rexp_vector, idle, rexp_vector]
            ff_vec_list1 = [ff_vec[0]]
            pulse_type_list1 = ['square']
            ff_len_list1 = [[10] * 8]
            delay_list1 = [0]
            mod_pulse_list1 = [False]
            mod_amp_list1 = [self.expt_cfg["mod_amp"]]

            for reps in range(num_gates):
                ff_vec_list1.append(ff_vec[1])
                ff_vec_list1.append(ff_vec[3])
                for jj in range(2):
                    pulse_type_list1.append('rexp')
                    mod_pulse_list1.append(True)
                    mod_amp_list1.append(self.expt_cfg["mod_amp"])
                    ff_len_list1.append([self.expt_cfg["mod_piLen"]] * 8)
                    delay_list1.append(10 + (2 * reps + jj) * (2*self.expt_cfg["ramp_len"] + self.expt_cfg["mod_piLen"]))

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list1, pulse_type_list=pulse_type_list1,
                                   flux_vec_list=ff_vec_list1, delay_list=delay_list1,
                                   freq_list=[0] * (2 * num_gates + 1), flip_amp=False,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=mod_amp_list1,
                                   mod_pulse_list=mod_pulse_list1)
            sequencer.sync_channels_time(self.channels)

            ################### echo Pi pulse ##################
            self.idle_q(sequencer, qb=X90_qb, time=25)
            sequencer.sync_channels_time(self.channels)
            if echo_qb is not None:
                self.idle_q(sequencer, qb=echo_qb, time=25)
                sequencer.sync_channels_time(self.channels)

                setup = self.lattice_cfg["qubit"]["setup"][echo_qb]
                self.gen_q(sequencer, qb=echo_qb, len=self.pulse_info[setup]["pi_len"][echo_qb],
                           amp=self.pulse_info[setup]["pi_amp"][echo_qb],
                           pulse_type=self.pulse_info["pulse_type"][echo_qb])

                self.idle_q(sequencer, qb=echo_qb, time=25)
                sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, qb=X90_qb, time=25)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs after echo ###########################################
            ff_vec = self.expt_cfg["ff_vec_list"]  # [snap_vect, rexp_vector, idle, rexp_vector]
            ff_vec_list2 = [ff_vec[0]]
            pulse_type_list2 = ['square']
            ff_len_list2 = [[10] * 8]
            delay_list2 = [0]
            mod_pulse_list2 = [False]
            mod_amp_list2 = [self.expt_cfg["mod_amp"]]

            for reps in range(num_gates-1):
                ff_vec_list2.append(ff_vec[1])
                ff_vec_list2.append(ff_vec[3])
                for jj in range(2):
                    pulse_type_list2.append('rexp')
                    mod_pulse_list2.append(True)
                    mod_amp_list2.append(self.expt_cfg["mod_amp"])
                    ff_len_list2.append([self.expt_cfg["mod_piLen"]] * 8)
                    delay_list2.append(10 + (2 * reps + jj) * (2 * self.expt_cfg["ramp_len"] + self.expt_cfg["mod_piLen"]))

            ########## ramsey sequence with dwell time inbetween gates ##########
            ff_vec_list2.append(ff_vec[1])
            ff_vec_list2.append(ff_vec[2])
            ff_vec_list2.append(ff_vec[3])
            pulse_type_list2.append('rexp'); mod_pulse_list2.append(True); mod_amp_list2.append(self.expt_cfg["mod_amp"])
            pulse_type_list2.append('square'); mod_pulse_list2.append(False); mod_amp_list2.append(self.expt_cfg["mod_amp"])
            pulse_type_list2.append('rexp'); mod_pulse_list2.append(True); mod_amp_list2.append(self.expt_cfg["mod_amp"])

            ff_len_list2.append([self.expt_cfg["mod_piLen"]] * 8)
            ff_len_list2.append([1 + evolution_t] * 8)
            ff_len_list2.append([self.expt_cfg["mod_piLen"]] * 8)
            delay_list2.append(10 + (2 * (num_gates-1)) * (2 * self.expt_cfg["ramp_len"] + self.expt_cfg["mod_piLen"]))
            delay_list2.append(10 + (2 * (num_gates - 1)+1) * (2 * self.expt_cfg["ramp_len"] + self.expt_cfg["mod_piLen"]))
            delay_list2.append(11 + evolution_t + (2 * (num_gates - 1)+1) * (2 * self.expt_cfg["ramp_len"] + self.expt_cfg["mod_piLen"]))


            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list2, pulse_type_list=pulse_type_list2,
                                   flux_vec_list=ff_vec_list2, delay_list=delay_list2,
                                   freq_list=[0] * (2 * num_gates + 2), flip_amp=False,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=mod_amp_list2,
                                   mod_pulse_list=mod_pulse_list2)
            sequencer.sync_channels_time(self.channels)

            ############################## second Ramsey pulse ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb],
                               phase=2 * np.pi * evolution_t * self.expt_cfg['ramsey_freq'])

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list1, pulse_type_list=pulse_type_list1,
                                   flux_vec_list=ff_vec_list1, delay_list=delay_list1,
                                   freq_list=[0] * (2 * num_gates + 1), flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=mod_amp_list1,
                                   mod_pulse_list=mod_pulse_list1)
            sequencer.sync_channels_time(self.channels)

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list2, pulse_type_list=pulse_type_list2,
                                   flux_vec_list=ff_vec_list2, delay_list=delay_list2,
                                   freq_list=[0] * (2 * num_gates + 2), flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=mod_amp_list2,
                                   mod_pulse_list=mod_pulse_list2)
            sequencer.sync_channels_time(self.channels)

            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def mb_NOON_ramseyMod(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        X90_qb = self.expt_cfg["X90_qb"]
        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb]) # generate ancilla X90 gate
            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs MELT + QUENCH ###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            # delay_list = self.expt_cfg["delay_list"]
            # delay_list[3] = delay_list[3] + evolution_t

            ff_len_list = [[ff_len + (step==2) * evolution_t]*8 for step, ff_len  in enumerate(self.expt_cfg["ff_len_list"])]
            delay_list = [d + (dd==3)*evolution_t for dd, d in enumerate(self.expt_cfg["delay_list"])]

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * 7, flip_amp=False,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=self.expt_cfg["mod_amp"],
                                   mod_pulse_list=self.expt_cfg["mod_pulse_list"])

            ############################## second Ramsey pulse ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb],
                               phase=2 * np.pi * evolution_t * self.expt_cfg['ramsey_freq'])

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            # self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
            #                        flux_vec_list=ff_vec_list, delay_list=delay_list,freq_list=[0]*7,flip_amp=True,
            #                        modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list = self.expt_cfg["mod_amp"])

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * 7, flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=self.expt_cfg["mod_amp"],
                                   mod_pulse_list=self.expt_cfg["mod_pulse_list"])

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_mb_NOON_ramsey(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        X90_qb = self.expt_cfg["X90_qb"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb]) # generate aux X90 gate
            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs MELT + QUENCH ###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            ff_len_list = [[ff_len+evolution_t]*8 for ff_len in self.expt_cfg["ff_len_list"]]
            # we are not modulating but we still need to set these vectors to have the right dimensions...
            # freq_list = [[0]*7]*7
            # ff_ac_amp_list = [[0]*7]*7
            # self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
            #                        flux_vec_list = ff_vec_list, delay_list = self.expt_cfg["delay_list"], freq_list = freq_list, amp_list=ff_ac_amp_list, flip_amp=False)
            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=self.expt_cfg["delay_list"],
                                   freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list = self.expt_cfg["mod_amp"])

            ############################## second Ramsey pulse ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=10)
            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb],
                               phase=2 * np.pi * evolution_t * self.expt_cfg['ramsey_freq'])
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])


            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=self.expt_cfg["delay_list"],freq_list=[0]*7,flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list = self.expt_cfg["mod_amp"])

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def mb_NOON_Mod_repeat(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        X90_qb = self.expt_cfg["X90_qb"]

        for num_gates in range(self.expt_cfg["num_gates"]+1):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb]) # generate ancilla X90 gate
            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs MELT + QUENCH ###########################################
            ff_vec = self.expt_cfg["ff_vec_list"] # [snap_vect, melt_vector]
            ff_vec_list = [ff_vec[0]]
            pulse_type_list = ['square']
            mod_pulse_list = [False]
            mod_amp_list = [self.expt_cfg["mod_amp"]]

            ff_len_list = [[10]*8]
            delay_list = [0]

            for reps in range(num_gates):
                ff_vec_list.append(ff_vec[1]) # add MB gate flux vector
                pulse_type_list.append('rexp')
                mod_pulse_list.append(True)
                ff_len_list.append([self.expt_cfg["gate_len"]]*8) # modulation time (ramp time included in the rexp)
                delay_list.append( 10 + (2*reps) * (2 * self.expt_cfg["ramp_len"] + self.expt_cfg["gate_len"]))
                mod_amp_list.append(self.expt_cfg["mod_amp"])

                ff_vec_list.append(ff_vec[1])  # add MB gate flux vector
                pulse_type_list.append('rexp')
                mod_pulse_list.append(True)
                ff_len_list.append([self.expt_cfg["gate_len"]] * 8)  # modulation time (ramp time included in the rexp)
                delay_list.append(10 + (2*reps+1) * (2 * self.expt_cfg["ramp_len"] + self.expt_cfg["gate_len"]))
                mod_amp_list.append(self.expt_cfg["mod_amp"])

            # ff_len_list = [[ff_len + (step == 2) * evolution_t] * 8 for step, ff_len in enumerate(self.expt_cfg["ff_len_list"])]
            # delay_list = [d + (dd == 3) * evolution_t for dd, d in enumerate(self.expt_cfg["delay_list"])]

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=pulse_type_list,
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * (2*num_gates+1), flip_amp=False,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=mod_amp_list,
                                   mod_pulse_list=mod_pulse_list)

            sequencer.sync_channels_time(self.channels)

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=pulse_type_list,
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * (2*num_gates+1), flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=mod_amp_list,
                                   mod_pulse_list=mod_pulse_list)


            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_mb_NOON_modulate(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        X90_qb = self.expt_cfg["X90_qb"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb]) # generate ancilla X90 gate
            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs MELT + QUENCH ###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            ff_len_list = [[ff_len+evolution_t]*8 for ff_len in self.expt_cfg["ff_len_list"]]
            # we are not modulating but we still need to set these vectors to have the right dimensions...
            # freq_list = [[0]*7]*7
            # ff_ac_amp_list = [[0]*7]*7
            # self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
            #                        flux_vec_list = ff_vec_list, delay_list = self.expt_cfg["delay_list"], freq_list = freq_list, amp_list=ff_ac_amp_list, flip_amp=False)
            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=self.expt_cfg["delay_list"],
                                   freq_list=[0]*7, flip_amp=False,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list = self.expt_cfg["mod_amp"],
                                   mod_pulse_list=self.expt_cfg["mod_pulse_list"])

            sequencer.sync_channels_time(self.channels)

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=self.expt_cfg["delay_list"],freq_list=[0]*7,flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list = self.expt_cfg["mod_amp"], mod_pulse_list=self.expt_cfg["mod_pulse_list"])


            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_mb_NOON(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        X90_qb = self.expt_cfg["X90_qb"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb]) # generate ancilla X90 gate
            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs MELT + QUENCH ###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            ff_len_list = [[ff_len+evolution_t]*8 for ff_len in self.expt_cfg["ff_len_list"]]
            # we are not modulating but we still need to set these vectors to have the right dimensions...
            # freq_list = [[0]*7]*7
            # ff_ac_amp_list = [[0]*7]*7
            # self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
            #                        flux_vec_list = ff_vec_list, delay_list = self.expt_cfg["delay_list"], freq_list = freq_list, amp_list=ff_ac_amp_list, flip_amp=False)
            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=self.expt_cfg["delay_list"],
                                   freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list = self.expt_cfg["mod_amp"])

            sequencer.sync_channels_time(self.channels)

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=self.expt_cfg["delay_list"],freq_list=[0]*7,flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list = self.expt_cfg["mod_amp"])

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_mb_NOON_repeat(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        X90_qb = self.expt_cfg["X90_qb"]

        for num_gates in range(self.expt_cfg["num_gates"] + 1):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb]) # generate ancilla X90 gate
                self.idle_q(sequencer, qb = X90_qb, time=20)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs MELT + QUENCH ###########################################
            ff_vec = self.expt_cfg["ff_vec_list"] # [snap_vect, melt_vector, rev_melt_vector]
            ff_vec_list = [ff_vec[0]]
            pulse_type_list = ['square']
            ff_len_list = [[10] * 8]
            delay_list = [0]

            mod_pulse_list = [False]
            mod_amp_list = [self.expt_cfg["mod_amp"]]

            for reps in range(num_gates):
                ff_vec_list.append(ff_vec[1])
                ff_vec_list.append(ff_vec[2])
                ff_vec_list.append(ff_vec[3])
                ff_vec_list.append(ff_vec[4])  # add MB gate flux vector
                pulse_type_list.append('exp')
                pulse_type_list.append('exp_flip_tau')
                pulse_type_list.append('exp')
                pulse_type_list.append('exp_flip_tau')

                for jj in range(4):
                    mod_pulse_list.append(False)
                    mod_amp_list.append(self.expt_cfg["mod_amp"])
                    ff_len_list.append([0] * 8)
                    delay_list.append(10 + (4*reps + jj) * self.expt_cfg["ramp_len"])

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=pulse_type_list,
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * (4*num_gates+1), flip_amp=False,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=mod_amp_list,
                                   mod_pulse_list=mod_pulse_list)

            sequencer.sync_channels_time(self.channels)

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=pulse_type_list,
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * (4*num_gates + 1), flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=mod_amp_list,
                                   mod_pulse_list=mod_pulse_list)

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_mb_NOON_repeat_ramsey(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        X90_qb = self.expt_cfg["X90_qb"]
        num_gates = self.expt_cfg["num_gates"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb]) # generate ancilla X90 gate
                self.idle_q(sequencer, qb = X90_qb, time=20)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs MELT + QUENCH ###########################################
            ff_vec = self.expt_cfg["ff_vec_list"] # [snap_vect, melt_vector, rev_melt_vector]
            ff_vec_list = [ff_vec[0]]
            pulse_type_list = ['square']
            ff_len_list = [[10] * 8]
            delay_list = [0]

            mod_pulse_list = [False]
            mod_amp_list = [self.expt_cfg["mod_amp"]]

            for reps in range(num_gates):
                ff_vec_list.append(ff_vec[1])
                ff_vec_list.append(ff_vec[2])
                ff_vec_list.append(ff_vec[3])
                ff_vec_list.append(ff_vec[4])  # add MB gate flux vector
                pulse_type_list.append('exp')
                pulse_type_list.append('exp_flip_tau')
                pulse_type_list.append('exp')
                pulse_type_list.append('exp_flip_tau')

                for jj in range(4):
                    mod_pulse_list.append(False)
                    mod_amp_list.append(self.expt_cfg["mod_amp"])

                    if reps == (num_gates-1):
                        ff_len_list.append([0 + (jj<2) *evolution_t] * 8)
                        delay_list.append(10 + (4 * reps + jj) * self.expt_cfg["ramp_len"] + (jj>1)*evolution_t )
                    else:
                        ff_len_list.append([0] * 8)
                        delay_list.append(10 + (4*reps + jj) * self.expt_cfg["ramp_len"])

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=pulse_type_list,
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * (4*num_gates+1), flip_amp=False,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=mod_amp_list,
                                   mod_pulse_list=mod_pulse_list)

            sequencer.sync_channels_time(self.channels)

            ############################## second Ramsey pulse ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb],
                               phase=2 * np.pi * evolution_t * self.expt_cfg['ramsey_freq'])

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=pulse_type_list,
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * (4*num_gates + 1), flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=mod_amp_list,
                                   mod_pulse_list=mod_pulse_list)

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_mb_NOON_repeat_echo(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]
        X90_qb = self.expt_cfg["X90_qb"]
        echo_qb = self.expt_cfg["echo_qb"]
        num_gates = self.expt_cfg["num_gates"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb]) # generate ancilla X90 gate
                self.idle_q(sequencer, qb = X90_qb, time=20)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs before echo ###########################################
            ff_vec = self.expt_cfg["ff_vec_list"]  # [snap_vect, melt_vector, rev_melt_vector]
            ff_vec_list = [ff_vec[0]]
            pulse_type_list = ['square']
            ff_len_list = [[10] * 8]
            delay_list = [0]
            mod_pulse_list = [False]
            mod_amp_list = [self.expt_cfg["mod_amp"]]

            for reps in range(num_gates):
                ff_vec_list.append(ff_vec[1])
                ff_vec_list.append(ff_vec[2])
                ff_vec_list.append(ff_vec[3])
                ff_vec_list.append(ff_vec[4])  # add MB gate flux vector
                pulse_type_list.append('exp')
                pulse_type_list.append('exp_flip_tau')
                pulse_type_list.append('exp')
                pulse_type_list.append('exp_flip_tau')
                for jj in range(4):
                    mod_pulse_list.append(False)
                    mod_amp_list.append(self.expt_cfg["mod_amp"])
                    ff_len_list.append([0] * 8)
                    delay_list.append(10 + (4 * reps + jj) * self.expt_cfg["ramp_len"])

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=pulse_type_list,
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * (4 * num_gates + 1), flip_amp=False,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=mod_amp_list,
                                   mod_pulse_list=mod_pulse_list)
            sequencer.sync_channels_time(self.channels)

            ################### echo Pi pulse ##################
            self.idle_q(sequencer, qb=X90_qb, time=25)
            sequencer.sync_channels_time(self.channels)
            if echo_qb is not None:
                self.idle_q(sequencer, qb=echo_qb, time=25)
                sequencer.sync_channels_time(self.channels)

                setup = self.lattice_cfg["qubit"]["setup"][echo_qb]
                self.gen_q(sequencer, qb=echo_qb, len=self.pulse_info[setup]["pi_len"][echo_qb],
                           amp=self.pulse_info[setup]["pi_amp"][echo_qb],
                           pulse_type=self.pulse_info["pulse_type"][echo_qb])

                self.idle_q(sequencer, qb=echo_qb, time=25)
                sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, qb=X90_qb, time=25)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs after echo ###########################################
            ff_vec = self.expt_cfg["ff_vec_list"] # [snap_vect, melt_vector, rev_melt_vector]
            ff_vec_list = [ff_vec[0]]
            pulse_type_list = ['square']
            ff_len_list = [[10] * 8]
            delay_list = [0]
            mod_pulse_list = [False]
            mod_amp_list = [self.expt_cfg["mod_amp"]]

            for reps in range(num_gates):
                ff_vec_list.append(ff_vec[1])
                ff_vec_list.append(ff_vec[2])
                ff_vec_list.append(ff_vec[3])
                ff_vec_list.append(ff_vec[4])  # add MB gate flux vector
                pulse_type_list.append('exp')
                pulse_type_list.append('exp_flip_tau')
                pulse_type_list.append('exp')
                pulse_type_list.append('exp_flip_tau')
                for jj in range(4):
                    mod_pulse_list.append(False)
                    mod_amp_list.append(self.expt_cfg["mod_amp"])
                    if reps == (num_gates-1):
                        ff_len_list.append([0 + (jj<2) *evolution_t] * 8)
                        delay_list.append(10 + (4 * reps + jj) * self.expt_cfg["ramp_len"] + (jj>1)*evolution_t )
                    else:
                        ff_len_list.append([0] * 8)
                        delay_list.append(10 + (4*reps + jj) * self.expt_cfg["ramp_len"])

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=pulse_type_list,
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * (4*num_gates+1), flip_amp=False,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=mod_amp_list,
                                   mod_pulse_list=mod_pulse_list)
            sequencer.sync_channels_time(self.channels)

            ############################## second Ramsey pulse ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            if X90_qb is not None:
                self.half_pi_q(sequencer, X90_qb, pulse_type=self.pulse_info['pulse_type'][X90_qb],
                               phase=2 * np.pi * evolution_t * self.expt_cfg['ramsey_freq'])

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            ff_vec = self.expt_cfg["ff_vec_list"]  # [snap_vect, melt_vector, rev_melt_vector]
            ff_vec_list = [ff_vec[0]]
            pulse_type_list = ['square']
            ff_len_list = [[10] * 8]
            delay_list = [0]
            mod_pulse_list = [False]
            mod_amp_list = [self.expt_cfg["mod_amp"]]

            for reps in range(num_gates):
                ff_vec_list.append(ff_vec[1])
                ff_vec_list.append(ff_vec[2])
                ff_vec_list.append(ff_vec[3])
                ff_vec_list.append(ff_vec[4])  # add MB gate flux vector
                pulse_type_list.append('exp')
                pulse_type_list.append('exp_flip_tau')
                pulse_type_list.append('exp')
                pulse_type_list.append('exp_flip_tau')
                for jj in range(4):
                    mod_pulse_list.append(False)
                    mod_amp_list.append(self.expt_cfg["mod_amp"])
                    ff_len_list.append([0] * 8)
                    delay_list.append(10 + (4 * reps + jj) * self.expt_cfg["ramp_len"])

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=pulse_type_list,
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * (4 * num_gates + 1), flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=mod_amp_list,
                                   mod_pulse_list=mod_pulse_list)
            sequencer.sync_channels_time(self.channels)

            ff_vec = self.expt_cfg["ff_vec_list"]  # [snap_vect, melt_vector, rev_melt_vector]
            ff_vec_list = [ff_vec[0]]
            pulse_type_list = ['square']
            ff_len_list = [[10] * 8]
            delay_list = [0]
            mod_pulse_list = [False]
            mod_amp_list = [self.expt_cfg["mod_amp"]]

            for reps in range(num_gates):
                ff_vec_list.append(ff_vec[1])
                ff_vec_list.append(ff_vec[2])
                ff_vec_list.append(ff_vec[3])
                ff_vec_list.append(ff_vec[4])  # add MB gate flux vector
                pulse_type_list.append('exp')
                pulse_type_list.append('exp_flip_tau')
                pulse_type_list.append('exp')
                pulse_type_list.append('exp_flip_tau')
                for jj in range(4):
                    mod_pulse_list.append(False)
                    mod_amp_list.append(self.expt_cfg["mod_amp"])
                    if reps == (num_gates - 1):
                        ff_len_list.append([0 + (jj < 2) * evolution_t] * 8)
                        delay_list.append(10 + (4 * reps + jj) * self.expt_cfg["ramp_len"] + (jj > 1) * evolution_t)
                    else:
                        ff_len_list.append([0] * 8)
                        delay_list.append(10 + (4 * reps + jj) * self.expt_cfg["ramp_len"])

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=pulse_type_list,
                                   flux_vec_list=ff_vec_list, delay_list=delay_list,
                                   freq_list=[0] * (4 * num_gates + 1), flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list=mod_amp_list,
                                   mod_pulse_list=mod_pulse_list)
            sequencer.sync_channels_time(self.channels)

            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_quench_single_readout(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs MELT + QUENCH ###########################################
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            ff_len_list = [[ff_len+evolution_t]*8 for ff_len in self.expt_cfg["ff_len_list"]]
            # we are not modulating but we still need to set these vectors to have the right dimensions...
            # freq_list = [[0]*7]*7
            # ff_ac_amp_list = [[0]*7]*7
            # self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
            #                        flux_vec_list = ff_vec_list, delay_list = self.expt_cfg["delay_list"], freq_list = freq_list, amp_list=ff_ac_amp_list, flip_amp=False)
            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=self.expt_cfg["delay_list"],
                                   freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list = self.expt_cfg["mod_amp"])

            sequencer.sync_channels_time(self.channels)

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
                                   flux_vec_list=ff_vec_list, delay_list=self.expt_cfg["delay_list"],freq_list=[0]*7,flip_amp=True,
                                   modulate_qb=self.expt_cfg["modulate_qb"], mod_amp_list = self.expt_cfg["mod_amp"])

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_modulateprep_single_readout(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ################################## GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            self.idle_q(sequencer, time=100)
            sequencer.sync_channels_time(self.channels)

            ############################## GENERATE RAMPs MELT + QUENCH ###########################################

            # in order to do a combined exp ramp+modulation+square, ff_pulse_combined needed to be changed
            # didn't want to break other experiments so I avoided using it in this experiment specifically

            ff_vec_list = self.expt_cfg["ff_vec_list"]
            ff_len_list = self.expt_cfg["ff_len_list"]
            pulse_type_list = self.expt_cfg["pulse_type_list"]
            delay_list = self.expt_cfg["delay_list"]
            freq_list = [0] * 7
            flip_amp = False
            modulate_qb = self.expt_cfg["modulate_qb"]
            mod_amp_list = [0]*7

            for qb in range(len(ff_vec_list[0])):  # just get nb of qbs/channels
                pulse_list = []
                # exp ramp + modulation
                pulse = self.return_ff_pulse(qb, ff_vec_list[0][qb], [ff_len_list[0]]*7, pulse_type_list[0], freq_list[qb], flip_amp, modulate_qb)
                pulse_list.append(pulse)
                # square pulse (NO modulation)
                pulse = self.return_ff_pulse(qb, ff_vec_list[1][qb], [ff_len_list[1] + evolution_t]*7, pulse_type_list[1], freq_list[qb],flip_amp, [])
                pulse_list.append(pulse)
                sequencer.append_composite('ff_Q%s' % qb, pulse_list, delay_list)

            sequencer.sync_channels_time(self.channels)

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################

            flip_amp = True
            for qb in range(len(ff_vec_list[0])):  # just get nb of qbs/channels
                pulse_list = []
                # exp ramp + modulation
                pulse = self.return_ff_pulse(qb, ff_vec_list[0][qb], [ff_len_list[0]]*7, pulse_type_list[0], freq_list[qb], flip_amp, modulate_qb)
                pulse_list.append(pulse)
                # square pulse (NO modulation)
                pulse = self.return_ff_pulse(qb, ff_vec_list[1][qb], [ff_len_list[1] + evolution_t]*7, pulse_type_list[1], freq_list[qb], flip_amp, [])
                pulse_list.append(pulse)
                sequencer.append_composite('ff_Q%s' % qb, pulse_list, delay_list)

            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()
        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_single_readout_modulate_2setups(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ##################################GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)
            if self.expt_cfg["ef"]==True:
                qb=5
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.pi_q_ef(sequencer, qb, pulse_type = 'gauss')

            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=100)
            ##############################GENERATE RAMP###########################################
            flux_vec = self.expt_cfg["ff_vec"]
            self.ff_pulse(sequencer, ff_len = [evolution_t]*8, pulse_type = self.expt_cfg["ramp_type"], flux_vec= flux_vec,
                          flip_amp=False, modulate_qb = self.expt_cfg["mod_qb"])

            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            ############################ Optional Basis rotation for X measurement ###############
            if self.expt_cfg["pauliX"]:
                for qb in self.on_rds:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["half_pi_len"][qb],
                               amp=self.pulse_info[setup]['Y90_amp'][qb], phase=np.pi / 2,
                               pulse_type=self.pulse_info['pulse_type'][qb])
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################
            flux_vec = self.expt_cfg["ff_vec"]
            self.ff_pulse(sequencer, [evolution_t]*8, self.expt_cfg["ramp_type"], flux_vec=flux_vec, flip_amp=True, modulate_qb = self.expt_cfg["mod_qb"])
            sequencer.end_sequence()

        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)
        return sequencer.complete(self, plot=True)

    def melting_correlators_full_ramp_2setups(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]

        for evolution_t in range(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ##################################GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=100)
            ##############################GENERATE RAMP###########################################
            sequencer.sync_channels_time(self.channels)
            ff_vec_list = self.expt_cfg["ff_vec_list"]
            delay_list = self.expt_cfg["delay_list"]
            ff_len_list = self.expt_cfg["ff_len_list"]

            ff_len_list[-1] = [evolution_t]*7
            pulse_type_list = self.expt_cfg["ff_pulse_type_list"]
            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=pulse_type_list,
                                   flux_vec_list=ff_vec_list, delay_list=delay_list, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"])
            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################
            self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=pulse_type_list,
                                   flux_vec_list=ff_vec_list, delay_list=delay_list, freq_list=[0]*7, flip_amp=True, modulate_qb=self.expt_cfg["modulate_qb"])
            sequencer.end_sequence()

        # if self.expt_cfg["pi_calibration"]==True:
        #     for rd_qb in self.on_rds:
        #         if self.expt_cfg["pi_cal_2rds"]==True:
        #             self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)
        #         else:
        #             self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = [rd_qb])

        if self.expt_cfg["twoqb_calibration"]==True:
            self.twoqb_cal_sequence(sequencer, pi_qbs=self.on_rds, pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def melting_correlatorsXX_full_ramp_2setups(self, sequencer):
        qb_list = self.expt_cfg["Mott_qbs"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ##################################GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=100)
            ##############################GENERATE RAMP###########################################
            flux_vec = self.expt_cfg["ff_vec"]
            self.ff_pulse(sequencer, ff_len = [evolution_t]*8, pulse_type = self.expt_cfg["ramp_type"], flux_vec= flux_vec, flip_amp=False)
            sequencer.sync_channels_time(self.channels)

            ############################## readout ###########################################
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            ##### Basis rotation to measure X component
            for qb in self.on_rds:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["half_pi_len"][qb],
                            amp=self.pulse_info[setup]['Y90_amp'][qb],phase=np.pi/2, pulse_type=self.pulse_info['pulse_type'][qb])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################
            flux_vec = self.expt_cfg["ff_vec"]
            self.ff_pulse(sequencer, [evolution_t]*8, pulse_type=self.expt_cfg["ramp_type"], flux_vec=flux_vec, flip_amp=True)
            sequencer.end_sequence()

        if self.expt_cfg["twoqb_calibration"]==True:
            self.twoqb_cal_sequence(sequencer, pi_qbs=self.on_rds, pi_rds = self.on_rds)

        return sequencer.complete(self, plot=True)

    def sideband_t1(self, sequencer):
        """Drives sb_qb only with a flat_top gaussian flux pulse. Amp given by ff_amp, freq of pulse given by sb_freq, gaussian edges given by sb_sig.
        all other qubits except for sb_qb are not ff modulated. no compensation pulse since its a sine"""
        sb_qb = self.on_qbs[0]
        setup = self.lattice_cfg["qubit"]["setup"][sb_qb]
        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ##################################GENERATE PI PULSES ################################################
            pulse_info = self.lattice_cfg["pulse_info"]
            self.gen_q(sequencer, qb=sb_qb, len=pulse_info[setup]["pi_len"][sb_qb],
                       amp=pulse_info[setup]["pi_amp"][sb_qb], add_freq=0, phase=0,
                       pulse_type=pulse_info["pulse_type"][sb_qb])
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=100)
            ##############################GENERATE SIDEBAND###########################################
            flux_vec = [0]*8
            flux_vec[sb_qb] = self.expt_cfg["ff_amp"]
            self.lattice_cfg['ff_info']['ff_ramp_sigma_len'][sb_qb] = self.expt_cfg["sb_sig"]
            self.ff_pulse(sequencer, ff_len = [evolution_t]*8, pulse_type = "square", flux_vec= flux_vec, freq=self.expt_cfg["sb_freq"],flip_amp=False)
            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def pi_cal_sequence(self, sequencer, pi_qbs = None, pi_rds = None):
        #note: in pi_cal_sequence, defualt pi cal is done with self.on_qbs and self.on_rds. This can be changed as desired

        try:
            ef = self.expt_cfg["ef_cal"]
        except:
            ef = False

        if pi_qbs==None:
            pi_qbs= self.on_qbs
        if pi_rds==None:
            pi_rds = self.on_rds
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
        sequencer.sync_channels_time(self.channels)
        self.readout_pxi(sequencer, pi_rds, overlap=False)
        sequencer.sync_channels_time(self.channels)
        sequencer.end_sequence()

        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
        for qb in pi_qbs:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            self.pi_q(sequencer, qb=qb, phase=0, pulse_type=self.pulse_info["pulse_type"][qb])
        sequencer.sync_channels_time(self.channels)
        if 'wait_time' in self.expt_cfg.keys():
            self.idle_q(sequencer, time=self.expt_cfg['wait_time'])
            sequencer.sync_channels_time(self.channels)
        self.readout_pxi(sequencer, pi_rds, overlap=False)
        sequencer.sync_channels_time(self.channels)
        sequencer.end_sequence()

        if ef:
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            for qb in pi_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.pi_q(sequencer, qb=qb, phase=0, pulse_type=self.pulse_info["pulse_type"][qb])
                self.pi_q_ef(sequencer, qb=qb, phase=0, pulse_type=self.pulse_info["pulse_type"][qb])
            sequencer.sync_channels_time(self.channels)
            if 'wait_time' in self.expt_cfg.keys():
                self.idle_q(sequencer, time=self.expt_cfg['wait_time'])
                sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, pi_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()

    def twoqb_cal_sequence(self, sequencer, pi_qbs = None, pi_rds = None):
        #note: in pi_cal_sequence, defualt pi cal is done with self.on_qbs and self.on_rds. This can be changed as desired
        if pi_qbs==None:
            pi_qbs= self.on_qbs
        if pi_rds==None:
            pi_rds = self.on_rds

        for cal_qbs in [[], [pi_qbs[1]], [pi_qbs[0]], pi_qbs]:
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
            for qb in cal_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.pi_q(sequencer, qb=qb, phase=0, pulse_type=self.pulse_info["pulse_type"][qb])
            sequencer.sync_channels_time(self.channels)
            if 'wait_time' in self.expt_cfg.keys():
                self.idle_q(sequencer, time=self.expt_cfg['wait_time'])
                sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, pi_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)
            sequencer.end_sequence()


    def pi_cal(self, sequencer):
        self.pi_cal_sequence(sequencer)
        return sequencer.complete(self, plot=True)

    def ff_pi_cal(self, sequencer):
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
        sequencer.sync_channels_time(self.channels)
        if self.expt_cfg["ff_len"] == "auto":
            ff_len = np.asarray(
                self.lattice_cfg['ff_info']["ff_len"] + self.lattice_cfg['ff_info']['ff_pulse_padding'])
        else:
            ff_len = self.expt_cfg["ff_len"]

        self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                      flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
        sequencer.sync_channels_time(self.channels)
        self.idle_q(sequencer, time=self.expt_cfg["wait_post_flux"])
        #channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
        #self.readout_pxi(sequencer, self.on_rds, synch_channels=channels_excluding_fluxch)
        self.readout_pxi(sequencer, self.on_rds)
        sequencer.sync_channels_time(self.channels)
        self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                      flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)
        sequencer.end_sequence()

        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
        for qb in self.on_qbs:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            self.pi_q(sequencer, qb=qb, phase=0, pulse_type=self.pulse_info["pulse_type"][qb])
        sequencer.sync_channels_time(self.channels)
        if self.expt_cfg["ff_len"] == "auto":
            ff_len = np.asarray(
                self.lattice_cfg['ff_info']["ff_len"] + self.lattice_cfg['ff_info']['ff_pulse_padding'])
        else:
            ff_len = self.expt_cfg["ff_len"]

        self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                      flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
        sequencer.sync_channels_time(self.channels)
        self.idle_q(sequencer, time=self.expt_cfg["wait_post_flux"])
        #self.readout_pxi(sequencer, self.on_rds, synch_channels=channels_excluding_fluxch)
        self.readout_pxi(sequencer, self.on_rds)
        sequencer.sync_channels_time(self.channels)
        self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                      flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)
        sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def ff_sweep_j(self, sequencer, perc_flux_vec):
        qb_list = self.expt_cfg["Mott_qbs"]

        for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            ##################################GENERATE PI PULSES ################################################
            for i, qb in enumerate(qb_list):
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb], phase=0,
                           pulse_type=self.pulse_info["pulse_type"][qb])
                self.idle_q(sequencer, time=20)

            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=100)
            ##############################GENERATE RAMP###########################################
            flux_vec = np.array(self.expt_cfg["ff_vec"]) * perc_flux_vec
            self.ff_pulse(sequencer, [evolution_t]*8, pulse_type = self.expt_cfg["ramp_type"], flux_vec= flux_vec, flip_amp=False)
            ############################## readout ###########################################
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds, overlap=False)
            sequencer.sync_channels_time(self.channels)

            ############################## generate compensation ###########################################
            self.ff_pulse(sequencer, [evolution_t]*8, pulse_type=self.expt_cfg["ramp_type"], flux_vec=flux_vec, flip_amp=True)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)


    def pulse_probe_iq(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer=sequencer, qb=qb, len=self.expt_cfg['pulse_length'],
                           amp=self.expt_cfg['amp'], add_freq=dfreq, phase=0,
                           pulse_type=self.pulse_info['pulse_type'][qb])
                self.idle_q(sequencer, time=self.expt_cfg['delay'])
            self.readout_pxi(sequencer, self.on_rds,overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def pulse_probe_iq_flatfreq(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]

                sequencer.append('charge%s_I' % setup, FreqSquare(max_amp=self.expt_cfg['amp'],
                                                                 flat_freq_Ghz=self.expt_cfg['flat_len_Ghz'],
                                                                  pulse_len=self.expt_cfg['pulse_len'],
                                                                  conv_gauss_sig_Ghz=self.expt_cfg['conv_gauss_sig_Ghz'],
                                                                  freq=self.pulse_info[setup]['iq_freq'][qb]+dfreq,
                                                                  phase=0))
                sequencer.append('charge%s_Q' % setup, FreqSquare(max_amp=self.expt_cfg['amp'],
                                                                 flat_freq_Ghz=self.expt_cfg['flat_len_Ghz'],
                                                                  pulse_len=self.expt_cfg['pulse_len'],
                                                                  conv_gauss_sig_Ghz=self.expt_cfg['conv_gauss_sig_Ghz'],
                                                                  freq=self.pulse_info[setup]['iq_freq'][qb]+dfreq,
                                                                phase=0 + self.pulse_info[setup]['Q_phase'][qb]))

                #self.idle_q(sequencer, time=self.expt_cfg['delay'])
            self.readout_pxi(sequencer, self.on_rds,overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def pulse_probe_iq_pi(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            for qb in self.expt_cfg["pi_qb"]:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                # self.pi_q(self, sequencer, qb=qb, phase=0, pulse_type=self.pulse_info['pulse_type'][qb])
                # Throws an error for "multiple qb?" Replace with gen_q pi pulse
                self.gen_q(sequencer=sequencer, qb=qb, len=self.lattice_cfg['pulse_info'][setup]['pi_len'][qb],
                           amp=self.lattice_cfg['pulse_info'][setup]['pi_amp'][qb], add_freq=0, phase=0,
                           pulse_type=self.pulse_info['pulse_type'][qb])

            # synchronize - pi pulse before PPIQ
            sequencer.sync_channels_time(self.channels)

            for qb in self.expt_cfg["ppiq_qb"]:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer=sequencer, qb=qb, len=self.expt_cfg['pulse_length'],
                           amp=self.expt_cfg['amp'], add_freq=dfreq, phase=0,
                           pulse_type=self.pulse_info['pulse_type'][qb])
                self.idle_q(sequencer, time=self.expt_cfg['delay'])

            self.readout_pxi(sequencer, self.on_rds,overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def pulse_probe_ef_iq_pi(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            for qb in self.expt_cfg["pi_qb"]:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer=sequencer, qb=qb, len=self.lattice_cfg['pulse_info'][setup]['pi_len'][qb],
                           amp=self.lattice_cfg['pulse_info'][setup]['pi_amp'][qb], add_freq=0, phase=0,
                           pulse_type=self.pulse_info['pulse_type'][qb])

                # synchronize - pi pulse before PPIQ
                self.idle_q(sequencer, time=20)
                sequencer.sync_channels_time(self.channels)

            for qb in self.expt_cfg["ppiq_qb"]:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb, len=self.expt_cfg['pulse_length'],amp = self.pulse_info[setup]['pi_ef_amp'][qb],phase=0,pulse_type=self.pulse_info['pulse_type'][qb],add_freq=dfreq+self.lattice_cfg['qubit']['anharmonicity'][qb])
                self.idle_q(sequencer, time=self.expt_cfg['delay'])

            self.readout_pxi(sequencer, self.on_rds,overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ff_pulse_probe_ef_iq_pi_mxg(self, sequencer):

        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

        for qb in self.expt_cfg["pi_qb"]:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            self.gen_q(sequencer=sequencer, qb=qb, len=self.lattice_cfg['pulse_info'][setup]['pi_len'][qb],
                       amp=self.lattice_cfg['pulse_info'][setup]['pi_amp'][qb], add_freq=0, phase=0,
                       pulse_type=self.pulse_info['pulse_type'][qb])

            # synchronize - pi pulse before PPIQ
            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)

        pre_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch
        # add IQ pulse
        self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
        channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
        sequencer.sync_channels_time(channels_excluding_fluxch)

        #"EF" pulse
        sequencer.append('stab_I', Square(max_amp=1, flat_len=self.expt_cfg['qb_pulse_length'],
                                          ramp_sigma_len=0.001, cutoff_sigma=2, freq=0,
                                          phase=0))

        sequencer.sync_channels_time(channels_excluding_fluxch)
        post_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch

        if self.expt_cfg["ff_len"] == "auto":
            ff_len = [post_flux_time - pre_flux_time] * 8
        else:
            ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"])
        self.ff_pulse(sequencer, ff_len=ff_len, flux_vec=self.expt_cfg['ff_vec'],
                      pulse_type=self.expt_cfg['ff_pulse_type'])

        # synch all channels then do readout
        self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
        self.readout_pxi(sequencer, self.on_rds, overlap=False)

        # add compensation
        sequencer.sync_channels_time(self.channels)
        self.ff_pulse(sequencer, ff_len=ff_len, flux_vec=self.expt_cfg['ff_vec'],
                      pulse_type=self.expt_cfg['ff_pulse_type'],
                      flip_amp=True)

        sequencer.end_sequence()

        return sequencer.complete(self, plot=True)


    def ff_resonator_spectroscopy(self, sequencer):
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
        if self.expt_cfg["pulse_inside_flux"]:
            pre_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch
            self.idle_q(sequencer, time=self.lattice_cfg["ff_info"]["ff_settling_time"])
            channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
            self.readout_pxi(sequencer, self.on_rds, overlap=False, synch_channels=channels_excluding_fluxch)

            # add flux to pulse
            sequencer.sync_channels_time(channels_excluding_fluxch)
            post_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch

            if self.expt_cfg["ff_len"] == "auto":
                ff_len = [post_flux_time - pre_flux_time +self.lattice_cfg['ff_info']['ff_pulse_padding']] * 8
            else:
                ff_len = np.asarray(
                    np.array(self.lattice_cfg['ff_info']["ff_len"]) + self.lattice_cfg['ff_info']['ff_pulse_padding'])
            self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                          flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
            self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                          flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)

        else:
            sequencer.sync_channels_time(self.channels)
            if self.expt_cfg["ff_len"] == "auto":
                ff_len = np.asarray(
                    np.array(self.lattice_cfg['ff_info']["ff_len"])) + self.lattice_cfg['ff_info']['ff_pulse_padding']
            else:
                ff_len = self.expt_cfg["ff_len"]

            self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                          flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.lattice_cfg["ff_info"]["ff_settling_time"])
            self.readout_pxi(sequencer, self.on_rds, overlap=False, synch_channels=self.channels)
            sequencer.sync_channels_time(self.channels)
            self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                          flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)

        sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ff_resonator_spectroscopy_pi(self, sequencer):
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

        if self.expt_cfg["pulse_inside_flux"]:
            pre_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
            for qb in self.expt_cfg["on_qbs"]:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb], phase=0,
                           pulse_type=self.pulse_info["pulse_type"][qb])
            self.idle_q(sequencer, time=self.expt_cfg["delay"])
            channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
            self.readout_pxi(sequencer, self.on_rds, overlap=False, synch_channels=channels_excluding_fluxch)

            # add flux to pulse
            sequencer.sync_channels_time(channels_excluding_fluxch)
            post_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch

            if self.expt_cfg["ff_len"] == "auto":
                ff_len = [post_flux_time - pre_flux_time + self.lattice_cfg['ff_info']['ff_pulse_padding']] * 8
            else:
                ff_len = np.asarray(
                    self.lattice_cfg['ff_info']["ff_len"] + self.lattice_cfg['ff_info']['ff_pulse_padding'])
            self.ff_square_and_comp(sequencer, ff_len=ff_len)
        else:
            sequencer.sync_channels_time(self.channels)
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            for qb in self.expt_cfg["on_qbs"]:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                #qb_iq_freq_dif = self.lattice_cfg["qubit"]["freq"][qb] - self.lattice_cfg["qubit"]["freq"][res_nb]
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                           amp=self.pulse_info[setup]["pi_amp"][qb],phase=0,
                           pulse_type=self.pulse_info["pulse_type"][qb])
            self.idle_q(sequencer, time=self.expt_cfg["delay"])
            sequencer.sync_channels_time(self.channels)
            if self.expt_cfg["ff_len"] == "auto":
                ff_len = np.asarray(
                    self.lattice_cfg['ff_info']["ff_len"]) + self.lattice_cfg['ff_info']['ff_pulse_padding']
            else:
                ff_len = self.expt_cfg["ff_len"]

            self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                          flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
            sequencer.sync_channels_time(self.channels)
            self.idle_q(sequencer, time=self.lattice_cfg["ff_info"]["ff_settling_time"])
            self.readout_pxi(sequencer, self.on_rds, overlap=False, synch_channels=self.channels)
            sequencer.sync_channels_time(self.channels)
            self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                          flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)

        sequencer.end_sequence()
        return sequencer.complete(self, plot=True)

    def ff_pulse_probe_iq(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
            pre_flux_time = sequencer.get_time('digtzr_trig') #could just as well be any ch

            #add IQ pulse
            self.idle_q(sequencer, time=self.lattice_cfg['ff_info']['ff_settling_time'])
            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer=sequencer, qb=qb, len=self.expt_cfg['qb_pulse_length'], amp=self.expt_cfg['qb_amp'], add_freq=dfreq, phase=0, pulse_type=self.pulse_info['pulse_type'][qb])
            self.idle_q(sequencer, time=self.expt_cfg['delay'])

            if self.expt_cfg["rd_inside_flux"]:
                #synch all channels except flux before adding readout, then do readout
                channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
                self.readout_pxi(sequencer, self.on_rds, overlap=False, synch_channels=channels_excluding_fluxch)

                #add flux to pulse
                sequencer.sync_channels_time(channels_excluding_fluxch)
                post_flux_time = sequencer.get_time('digtzr_trig') #could just as well be any ch

                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = [post_flux_time-pre_flux_time]*8
                else:
                    ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"])
                self.ff_pulse(sequencer, ff_len=ff_len,flux_vec =self.expt_cfg['ff_vec'], pulse_type= self.expt_cfg['ff_pulse_type'])
                self.ff_pulse(sequencer, ff_len=ff_len, flux_vec=self.expt_cfg['ff_vec'], pulse_type=self.expt_cfg['ff_pulse_type'],
                              flip_amp=True)
                sequencer.end_sequence()
            else:
                channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
                sequencer.sync_channels_time(channels_excluding_fluxch)
                post_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch

                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = [post_flux_time - pre_flux_time] * 8
                else:
                    ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"])
                self.ff_pulse(sequencer, ff_len=ff_len, flux_vec=self.expt_cfg['ff_vec'],
                              pulse_type=self.expt_cfg['ff_pulse_type'])


                # synch all channels then do readout
                self.idle_q(sequencer, time=self.lattice_cfg['ff_info']['ff_settling_time'])
                self.readout_pxi(sequencer, self.on_rds, overlap=False)

                # add compensation
                sequencer.sync_channels_time(self.channels)
                self.ff_pulse(sequencer, ff_len=ff_len, flux_vec=self.expt_cfg['ff_vec'],
                              pulse_type=self.expt_cfg['ff_pulse_type'],
                              flip_amp=True)
                sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ff_pulse_probe_ef_iq(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
            pre_flux_time = sequencer.get_time('digtzr_trig') #could just as well be any ch

            #add IQ pulse
            self.idle_q(sequencer, time=self.lattice_cfg['ff_info']['ff_settling_time'])
            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                self.gen_q(sequencer, qb, len=self.expt_cfg['pulse_length'],amp = self.pulse_info[setup]['pi_ef_amp'][qb],phase=0,pulse_type=self.pulse_info['pulse_type'][qb],add_freq=dfreq+self.lattice_cfg['qubit']['anharmonicity'][qb])
            self.idle_q(sequencer, time=self.expt_cfg['delay'])

            if self.expt_cfg["rd_inside_flux"]:
                #synch all channels except flux before adding readout, then do readout
                channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
                self.readout_pxi(sequencer, self.on_rds, overlap=False, synch_channels=channels_excluding_fluxch)

                #add flux to pulse
                sequencer.sync_channels_time(channels_excluding_fluxch)
                post_flux_time = sequencer.get_time('digtzr_trig') #could just as well be any ch

                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = [post_flux_time-pre_flux_time]*8
                else:
                    ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"])
                self.ff_pulse(sequencer, ff_len=ff_len,flux_vec =self.expt_cfg['ff_vec'], pulse_type= self.expt_cfg['ff_pulse_type'])
                self.ff_pulse(sequencer, ff_len=ff_len, flux_vec=self.expt_cfg['ff_vec'], pulse_type=self.expt_cfg['ff_pulse_type'],
                              flip_amp=True)
                sequencer.end_sequence()
            else:
                channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
                sequencer.sync_channels_time(channels_excluding_fluxch)
                post_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch

                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = [post_flux_time - pre_flux_time] * 8
                else:
                    ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"])
                self.ff_pulse(sequencer, ff_len=ff_len, flux_vec=self.expt_cfg['ff_vec'],
                              pulse_type=self.expt_cfg['ff_pulse_type'])


                # synch all channels then do readout
                self.idle_q(sequencer, time=self.lattice_cfg['ff_info']['ff_settling_time'])
                self.readout_pxi(sequencer, self.on_rds, overlap=False)

                # add compensation
                sequencer.sync_channels_time(self.channels)
                self.ff_pulse(sequencer, ff_len=ff_len, flux_vec=self.expt_cfg['ff_vec'],
                              pulse_type=self.expt_cfg['ff_pulse_type'],
                              flip_amp=True)
                sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ff_pulse_probe_iq_pi(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
            pre_flux_time = sequencer.get_time('digtzr_trig') #could just as well be any ch
            channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]

            #add IQ pulse
            self.idle_q(sequencer, time=self.lattice_cfg['ff_info']['ff_settling_time'])
            sequencer.sync_channels_time(channels_excluding_fluxch)
            for qb in self.expt_cfg["pi_qb"]:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.pi_q(sequencer=sequencer, qb=qb, phase=0, pulse_type=self.pulse_info['pulse_type'][qb])
                sequencer.sync_channels_time(channels_excluding_fluxch)
            for qb in self.expt_cfg["ppiq_qb"]:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer=sequencer, qb=qb, len=self.expt_cfg['qb_pulse_length'], amp=self.expt_cfg['qb_amp'], add_freq=dfreq, phase=0, pulse_type=self.pulse_info['pulse_type'][qb])
            self.idle_q(sequencer, time=self.expt_cfg['delay'])

            if self.expt_cfg["rd_inside_flux"]:
                #synch all channels except flux before adding readout, then do readout

                self.readout_pxi(sequencer, self.on_rds, overlap=False, synch_channels=channels_excluding_fluxch)

                #add flux to pulse
                sequencer.sync_channels_time(channels_excluding_fluxch)
                post_flux_time = sequencer.get_time('digtzr_trig') #could just as well be any ch

                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = [post_flux_time-pre_flux_time]*8
                else:
                    ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"])
                self.ff_pulse(sequencer, ff_len=ff_len,flux_vec =self.expt_cfg['ff_vec'], pulse_type= self.expt_cfg['ff_pulse_type'])
                self.ff_pulse(sequencer, ff_len=ff_len, flux_vec=self.expt_cfg['ff_vec'], pulse_type=self.expt_cfg['ff_pulse_type'],
                              flip_amp=True)
                sequencer.end_sequence()
            else:
                channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
                sequencer.sync_channels_time(channels_excluding_fluxch)
                post_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch

                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = [post_flux_time - pre_flux_time] * 8
                else:
                    ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"])
                self.ff_pulse(sequencer, ff_len=ff_len, flux_vec=self.expt_cfg['ff_vec'],
                              pulse_type=self.expt_cfg['ff_pulse_type'])


                # synch all channels then do readout
                self.idle_q(sequencer, time=self.lattice_cfg['ff_info']['ff_settling_time'])
                self.readout_pxi(sequencer, self.on_rds, overlap=False)

                # add compensation
                sequencer.sync_channels_time(self.channels)
                self.ff_pulse(sequencer, ff_len=ff_len, flux_vec=self.expt_cfg['ff_vec'],
                              pulse_type=self.expt_cfg['ff_pulse_type'],
                              flip_amp=True)
                sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ff_ramp_cal_ppiq(self, sequencer):
        delt = self.expt_cfg['delt']
        pulse_type = self.expt_cfg['pulse_type']
        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            if delt<0:
                #ppiq first if dt less than zero
                for qb in self.on_qbs:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.gen_q(sequencer=sequencer, qb=qb, len=self.expt_cfg['qb_pulse_length'],
                               amp=self.expt_cfg['qb_amp'], add_freq=dfreq, phase=0, pulse_type=self.pulse_info['pulse_type'][qb])

                # wait time dt, the apply flux pulse
                for qb in range(8):
                    sequencer.append('ff_Q%s' % qb, Idle(time=-delt))

                if self.expt_cfg["ff_len"] == "auto":
                    readout_len_tot = 0
                    for rd in self.on_rds:
                        readout_len_tot += self.lattice_cfg['readout']['length'][rd]
                    ff_len = 8 * [
                        self.expt_cfg['qb_pulse_length'] + delt + readout_len_tot]
                else:
                    ff_len = self.lattice_cfg['ff_info']["ff_len"]

                # add flux pulse
                area_vec = self.ff_pulse(sequencer, ff_len= ff_len, pulse_type= pulse_type, flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)

            else:
                #flux pulse first if dt greater than zero
                if self.expt_cfg["ff_len"] == "auto":
                    readout_len_tot = 0
                    for rd in self.on_rds:
                        readout_len_tot += self.lattice_cfg['readout']['length'][rd]
                    ff_len = 8 * [
                        self.expt_cfg['qb_pulse_length'] + delt + readout_len_tot]
                else:
                    ff_len = self.lattice_cfg['ff_info']["ff_len"]

                #add flux pulse
                area_vec = self.ff_pulse(sequencer, ff_len= ff_len, pulse_type= pulse_type, flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)

                #wait time dt, the apply ppiq
                self.idle_q(sequencer, time=delt)
                for qb in self.on_qbs:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.gen_q(sequencer=sequencer, qb=qb, len=self.expt_cfg['qb_pulse_length'], amp=self.expt_cfg['qb_amp'], add_freq=dfreq, phase=0, pulse_type=self.pulse_info['pulse_type'][qb])

            #synch all channels except flux before adding readout, then do readout
            channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
            self.readout_pxi(sequencer, self.on_rds, overlap=False, synch_channels=channels_excluding_fluxch)

            #add compensation flux pulse
            fluxch = [ch for ch in self.channels if 'ff' in ch]
            sequencer.sync_channels_time(fluxch + ['readoutA']+['readoutB'])
            if self.lattice_cfg["ff_info"]["ff_comp_sym"]:
                self.ff_pulse(sequencer, ff_len= ff_len, pulse_type= pulse_type, flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)
            else:
                self.ff_comp(sequencer, area_vec)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ff_ramp_down_cal_ppiq(self, sequencer):
        delt = self.expt_cfg['delt']
        pulse_type = self.expt_cfg['pulse_type']
        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            # add fast flux pulse
            sequencer.sync_channels_time(self.channels)
            if self.expt_cfg["ff_len"] == "auto":
                readout_len_tot = 0
                for rd in self.on_rds:
                    readout_len_tot += self.lattice_cfg['readout']['length'][rd]
                ff_len = 8 * [
                    self.expt_cfg['qb_pulse_length'] + delt + readout_len_tot]
            else:
                ff_len = self.expt_cfg["ff_len"]
            area_vec = self.ff_pulse(sequencer, ff_len=ff_len, pulse_type=pulse_type, flux_vec=self.expt_cfg["ff_vec"],
                                     flip_amp=False)

            #add ppiq and readout
            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.idle_q(sequencer=sequencer, qb=qb, time=ff_len[0]+delt)
                self.gen_q(sequencer=sequencer, qb=qb, len=self.expt_cfg['qb_pulse_length'],
                           amp=self.expt_cfg['qb_amp'], add_freq=dfreq, phase=0, pulse_type=self.pulse_info["pulse_type"][qb])

            self.readout_pxi(sequencer, self.on_rds, overlap=False, synch_channels=self.channels)
            sequencer.sync_channels_time(self.channels)

            #add compensation flux pulse
            if self.lattice_cfg["ff_info"]["ff_comp_sym"]:
                self.ff_pulse(sequencer, ff_len= ff_len, pulse_type= pulse_type, flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)
            else:
                self.ff_comp(sequencer, area_vec)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ff_track_traj(self, sequencer, perc_flux_vec):
        ramp_type = self.expt_cfg['ramp_type']
        flux_vec = np.array(self.expt_cfg["ff_vec"]) * perc_flux_vec
        for dfreq in np.arange(self.expt_cfg['qbf_start'], self.expt_cfg['qbf_stop'], self.expt_cfg['qbf_step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            #flux pulse first if dt greater than zero
            if self.expt_cfg["ff_len"] == "auto":
                readout_len_tot = 0
                for rd in self.on_rds:
                    readout_len_tot += self.lattice_cfg['readout']['length'][rd]
                ff_len = 8 * [
                    self.expt_cfg['qb_pulse_length'] + delt + readout_len_tot]
            else:
                ff_len = self.lattice_cfg['ff_info']["ff_len"]

            #add flux pulse
            area_vec = self.ff_pulse(sequencer, ff_len= ff_len, pulse_type= ramp_type, flux_vec=flux_vec, flip_amp=False)

            #wait time dt, the apply ppiq
            self.idle_q(sequencer, time=self.expt_cfg["delay_post_ramp"])
            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer=sequencer, qb=qb, len=self.expt_cfg['qb_pulse_length'], amp=self.expt_cfg['qb_amp'], add_freq=dfreq, phase=0, pulse_type=self.pulse_info[setup]["pulse_type"][qb])

            #synch all channels except flux before adding readout, then do readout
            channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
            self.readout_pxi(sequencer, self.on_rds, overlap=False, synch_channels=channels_excluding_fluxch)

            #add compensation flux pulse
            fluxch = [ch for ch in self.channels if 'ff' in ch]
            sequencer.sync_channels_time(fluxch + ['readoutA']+["readoutB"])
            if self.lattice_cfg["ff_info"]["ff_comp_sym"]:
                area_vec = self.ff_pulse(sequencer, ff_len= ff_len, pulse_type= ramp_type, flux_vec=flux_vec, flip_amp=True)
            else:
                self.ff_comp(sequencer, area_vec)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ff_rabi(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            if self.expt_cfg["Qpulse_in_flux"]:
                pre_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch
                # add rabi pulse
                for qb in self.on_qbs:
                    self.idle_q(sequencer, qb=qb, time=self.lattice_cfg['ff_info']['ff_settling_time'])
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.gen_q(sequencer, qb, len=rabi_len, amp=self.expt_cfg['amp'], phase=0,
                               pulse_type=self.expt_cfg['pulse_type'])
                #synch all channels except flux before adding readout, then do readout
                channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
                self.readout_pxi(sequencer, self.on_rds, overlap=False, synch_channels=channels_excluding_fluxch)

                #add flux to pulse
                sequencer.sync_channels_time(channels_excluding_fluxch)
                post_flux_time = sequencer.get_time('digtzr_trig') #could just as well be any ch

                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = [post_flux_time-pre_flux_time]*8
                else:
                    ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"])
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)

            if self.expt_cfg["rd_outside_flux"]:
                pre_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch
                # add rabi pulse
                for qb in self.on_qbs:
                    self.idle_q(sequencer, qb=qb, time=self.lattice_cfg['ff_info']['ff_settling_time'])
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.gen_q(sequencer, qb, len=rabi_len, amp=self.expt_cfg['amp'], phase=0,
                               pulse_type=self.expt_cfg['pulse_type'])
                    self.idle_q(sequencer, qb=qb, time=100)
                #synch all channels except flux before adding readout, then do readout
                channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
                #add flux to pulse
                sequencer.sync_channels_time(channels_excluding_fluxch)
                post_flux_time = sequencer.get_time('digtzr_trig') #could just as well be any ch

                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = [post_flux_time-pre_flux_time]*8
                else:
                    ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"])
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)

                sequencer.sync_channels_time(self.channels)
                self.idle_q(sequencer, qb=qb, time=self.lattice_cfg['ff_info']['ff_settling_time'])
                self.readout_pxi(sequencer, self.on_rds, overlap=False, synch_channels=self.channels)
                sequencer.sync_channels_time(self.channels)

                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)

            else:
                # add rabi pulse
                for qb in self.on_qbs:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.gen_q(sequencer, qb, len=rabi_len, amp=self.expt_cfg['amp'], phase=0,
                               pulse_type=self.expt_cfg['pulse_type'])

                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = np.asarray(
                        self.lattice_cfg['ff_info']["ff_len"] + self.lattice_cfg['ff_info']['ff_pulse_padding'])
                else:
                    ff_len = self.expt_cfg["ff_len"]

                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"], flux_vec= self.expt_cfg["ff_vec"], flip_amp=False)
                sequencer.sync_channels_time(self.channels)
                self.idle_q(sequencer, time=self.lattice_cfg["ff_info"]["ff_settling_time"])
                self.readout_pxi(sequencer, self.on_rds, overlap=False, synch_channels=self.channels)
                sequencer.sync_channels_time(self.channels)
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)


            #end sequence
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ff_rabi_pi(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
            channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]

            if self.expt_cfg["Qpulse_in_flux"]:
                pre_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch
                # add rabi pulse
                self.idle_q(sequencer, time=self.lattice_cfg['ff_info']['ff_settling_time'])
                sequencer.sync_channels_time(channels_excluding_fluxch)
                for qb in self.expt_cfg["pi_qb"]:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.pi_q(sequencer, qb, phase=0, pulse_type=self.expt_cfg['pulse_type'])
                    self.idle_q(sequencer, qb=qb, time=20)
                    sequencer.sync_channels_time(channels_excluding_fluxch)
                for qb in self.expt_cfg["rabi_qb"]:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.gen_q(sequencer, qb, len=rabi_len, amp=self.expt_cfg['amp'], phase=0,
                               pulse_type=self.expt_cfg['pulse_type'])
                #synch all channels except flux before adding readout, then do readout

                self.readout_pxi(sequencer, self.on_rds, overlap=False, synch_channels=channels_excluding_fluxch)

                #add flux to pulse
                sequencer.sync_channels_time(channels_excluding_fluxch)
                post_flux_time = sequencer.get_time('digtzr_trig') #could just as well be any ch

                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = [post_flux_time-pre_flux_time]*8
                else:
                    ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"])
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)

            if self.expt_cfg["rd_outside_flux"]:
                pre_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch
                # add rabi pulse
                self.idle_q(sequencer, time=self.lattice_cfg['ff_info']['ff_settling_time'])
                sequencer.sync_channels_time(channels_excluding_fluxch)
                for qb in self.expt_cfg["pi_qb"]:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.pi_q(sequencer, qb, phase=0, pulse_type=self.expt_cfg['pulse_type'])
                    self.idle_q(sequencer, qb=qb, time=20)
                    sequencer.sync_channels_time(channels_excluding_fluxch)
                for qb in self.expt_cfg["rabi_qb"]:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.gen_q(sequencer, qb, len=rabi_len, amp=self.expt_cfg['amp'], phase=0,
                               pulse_type=self.expt_cfg['pulse_type'])
                #synch all channels except flux before adding readout, then do readout
                channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
                #add flux to pulse
                sequencer.sync_channels_time(channels_excluding_fluxch)
                post_flux_time = sequencer.get_time('digtzr_trig') #could just as well be any ch

                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = [post_flux_time-pre_flux_time]*8
                else:
                    ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"])
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)

                sequencer.sync_channels_time(self.channels)
                self.idle_q(sequencer, qb=qb, time=self.lattice_cfg['ff_info']['ff_settling_time'])
                self.readout_pxi(sequencer, self.on_rds, overlap=False, synch_channels=self.channels)
                sequencer.sync_channels_time(self.channels)

                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)

            else:
                # add rabi pulse
                for qb in self.expt_cfg["pi_qb"]:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.pi_q(sequencer, qb, phase=0, pulse_type=self.expt_cfg['pulse_type'])
                    self.idle_q(sequencer, qb=qb, time=20)
                    sequencer.sync_channels_time(self.channels)
                for qb in self.expt_cfg["rabi_qb"]:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.gen_q(sequencer, qb, len=rabi_len, amp=self.expt_cfg['amp'], phase=0,
                               pulse_type=self.expt_cfg['pulse_type'])

                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = np.asarray(
                        self.lattice_cfg['ff_info']["ff_len"] + self.lattice_cfg['ff_info']['ff_pulse_padding'])
                else:
                    ff_len = self.expt_cfg["ff_len"]

                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"], flux_vec= self.expt_cfg["ff_vec"], flip_amp=False)
                sequencer.sync_channels_time(self.channels)
                self.idle_q(sequencer, time=self.lattice_cfg["ff_info"]["ff_settling_time"])
                self.readout_pxi(sequencer, self.on_rds, overlap=False, synch_channels=self.channels)
                sequencer.sync_channels_time(self.channels)
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)


            #end sequence
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)


    def ff_t1(self, sequencer):

        for t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=['A','B'],time=500)
            pre_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch

            for qb in self.on_qbs:
                self.idle_q(sequencer, qb=qb, time=self.lattice_cfg['ff_info']['ff_pulse_padding'])
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.pi_q(sequencer,qb,pulse_type=self.pulse_info['pulse_type'][qb])
                #self.gen_q(sequencer,qb,len=2000,amp=1,phase=0,pulse_type=self.pulse_info['pulse_type'][qb])
                self.idle_q(sequencer, qb=qb, time=t1_len)

            # synch all channels except flux before adding readout, then do readout
            channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
            self.readout_pxi(sequencer, self.on_rds, overlap=False,
                             synch_channels=channels_excluding_fluxch)

            #add flux to pulse
            sequencer.sync_channels_time(channels_excluding_fluxch)
            post_flux_time = sequencer.get_time('digtzr_trig') #could just as well be any ch

            if self.expt_cfg["ff_len"] == "auto":
                ff_len = [post_flux_time-pre_flux_time + self.lattice_cfg['ff_info']['ff_pulse_padding']]*8
            else:
                ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"]+ self.lattice_cfg['ff_info']['ff_pulse_padding'])
            self.ff_square_and_comp(sequencer, ff_len=ff_len)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ff_ramsey(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            if not self.expt_cfg["Q_pulse_in_flux"]:
                for qb in self.on_qbs:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.half_pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                    self.idle_q(sequencer, qb=qb, time=20)

                    sequencer.sync_channels_time(self.channels)

                flux_vec = self.expt_cfg["ff_vec"]
                self.ff_pulse(sequencer, ff_len=[ramsey_len] * 8, pulse_type=self.expt_cfg["ramp_type"], flux_vec=flux_vec,
                              flip_amp=False, modulate_qb=self.expt_cfg["mod_qb"])
                sequencer.sync_channels_time(self.channels)

                self.idle_q(sequencer, qb=qb, time=self.expt_cfg["wait_post_flux"])
                sequencer.sync_channels_time(self.channels)
                for qb in self.on_qbs:
                    self.half_pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb],phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])

            else:
                pre_flux_time = sequencer.get_time('digtzr_trig')
                for qb in self.on_qbs:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.idle_q(sequencer, qb=qb, time=self.expt_cfg["wait_post_flux"])
                    self.half_pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                    self.idle_q(sequencer, qb=qb, time=ramsey_len)
                    self.half_pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb],
                                   phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])
                channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
                sequencer.sync_channels_time(channels_excluding_fluxch)
                post_flux_time = sequencer.get_time('digtzr_trig')
                flux_vec = self.expt_cfg["ff_vec"]
                self.ff_pulse(sequencer, ff_len=[int(post_flux_time-pre_flux_time)] * 8, pulse_type=self.expt_cfg["ramp_type"],
                              flux_vec=flux_vec,
                              flip_amp=False, modulate_qb=self.expt_cfg["mod_qb"])

                sequencer.sync_channels_time(self.channels)

                self.idle_q(sequencer, qb=qb, time=self.expt_cfg["wait_post_flux"])
                sequencer.sync_channels_time(self.channels)

            # synch all channels except flux before adding readout, then do readout
            self.readout_pxi(sequencer, self.on_rds, overlap=False,
                             synch_channels=self.channels)
            sequencer.sync_channels_time(self.channels)

            # add flux to pulse
            self.ff_pulse(sequencer, ff_len=[ramsey_len] * 8, pulse_type=self.expt_cfg["ramp_type"], flux_vec=flux_vec,
                          flip_amp=True, modulate_qb=self.expt_cfg["mod_qb"])

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ff_ramsey_legacy(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
            pre_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch

            for qb in self.on_qbs:
                self.idle_q(sequencer, qb=qb, time=self.lattice_cfg['ff_info']['ff_pulse_padding'])
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.half_pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                self.idle_q(sequencer, qb=qb, time=ramsey_len)
                self.half_pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb],phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])

            # synch all channels except flux before adding readout, then do readout
            channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
            self.readout_pxi(sequencer, self.on_rds, overlap=False,
                             synch_channels=channels_excluding_fluxch)

            # add flux to pulse
            sequencer.sync_channels_time(channels_excluding_fluxch)
            post_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch

            if self.expt_cfg["ff_len"] == "auto":
                ff_len = [post_flux_time-pre_flux_time + self.lattice_cfg['ff_info']['ff_pulse_padding']]*8
            else:
                ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"]+ self.lattice_cfg['ff_info']['ff_pulse_padding'])
            self.ff_square_and_comp(sequencer, ff_len=ff_len)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)


    def ff_histogram(self, sequencer):
        for ii in range(self.expt_cfg['num_seq_sets']):
            if self.expt_cfg["Qpulse_in_flux"]:
                ######################
                # just readout
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=['A','B'], time=500)
                pre_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch
                self.idle_q(sequencer, time=self.lattice_cfg['ff_info']['ff_settling_time'])
                # synch all channels except flux before adding readout, then do readout
                channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
                self.readout_pxi(sequencer, self.on_rds, overlap=False, synch_channels=channels_excluding_fluxch)

                # add flux to pulse
                sequencer.sync_channels_time(channels_excluding_fluxch)
                post_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch

                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = [post_flux_time - pre_flux_time] * 8
                else:
                    ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"])
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)
                sequencer.end_sequence()

                ###################
                # with pi, for e
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=['A','B'], time=500)
                pre_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch
                for qb in self.on_qbs:
                    self.idle_q(sequencer, qb=qb, time=self.lattice_cfg['ff_info']['ff_settling_time'])
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                # synch all channels except flux before adding readout, then do readout
                channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
                self.readout_pxi(sequencer, self.on_rds, overlap=False, synch_channels=channels_excluding_fluxch)

                # add flux to pulse
                sequencer.sync_channels_time(channels_excluding_fluxch)
                post_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch

                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = [post_flux_time - pre_flux_time] * 8
                else:
                    ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"])
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)
                sequencer.end_sequence()

                ###################
                # with pi and pief pulse
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=['A','B'], time=500)
                pre_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch
                for qb in self.on_qbs:
                    self.idle_q(sequencer, qb=qb, time=self.lattice_cfg['ff_info']['ff_settling_time'])
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                    self.pi_q_ef(sequencer, qb, pulse_type=self.pulse_info['ef_pulse_type'][qb])
                # synch all channels except flux before adding readout, then do readout
                channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
                self.readout_pxi(sequencer, self.on_rds, overlap=False, synch_channels=channels_excluding_fluxch)

                # add flux to pulse
                sequencer.sync_channels_time(channels_excluding_fluxch)
                post_flux_time = sequencer.get_time('digtzr_trig')  # could just as well be any ch

                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = [post_flux_time - pre_flux_time] * 8
                else:
                    ff_len = np.asarray(self.lattice_cfg['ff_info']["ff_len"])
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)
                sequencer.end_sequence()
            else:
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=['A','B'], time=500)
                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = np.asarray(
                        self.lattice_cfg['ff_info']["ff_len"] + self.lattice_cfg['ff_info']['ff_pulse_padding'])
                else:
                    ff_len = self.expt_cfg["ff_len"]

                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
                sequencer.sync_channels_time(self.channels)
                self.idle_q(sequencer, time=self.lattice_cfg["ff_info"]["ff_settling_time"])
                self.readout_pxi(sequencer, self.on_rds)
                sequencer.sync_channels_time(self.channels)
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)
                sequencer.end_sequence()

                # with pi pulse (e state)
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
                for qb in self.on_qbs:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = np.asarray(
                        self.lattice_cfg['ff_info']["ff_len"] + self.lattice_cfg['ff_info']['ff_pulse_padding'])
                else:
                    ff_len = self.expt_cfg["ff_len"]

                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
                sequencer.sync_channels_time(self.channels)
                self.idle_q(sequencer, time=self.lattice_cfg["ff_info"]["ff_settling_time"])
                self.readout_pxi(sequencer, self.on_rds)
                sequencer.sync_channels_time(self.channels)
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)
                sequencer.end_sequence()

                # with pi pulse and ef pi pulse (f state)
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
                for qb in self.on_qbs:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                    self.pi_q_ef(sequencer, qb, pulse_type=self.pulse_info['ef_pulse_type'][qb])
                sequencer.sync_channels_time(self.channels)
                if self.expt_cfg["ff_len"] == "auto":
                    ff_len = np.asarray(
                        self.lattice_cfg['ff_info']["ff_len"] + self.lattice_cfg['ff_info']['ff_pulse_padding'])
                else:
                    ff_len = self.expt_cfg["ff_len"]

                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
                sequencer.sync_channels_time(self.channels)
                self.idle_q(sequencer, time=self.lattice_cfg["ff_info"]["ff_settling_time"])
                self.readout_pxi(sequencer, self.on_rds)
                sequencer.sync_channels_time(self.channels)
                self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                              flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)

                sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def fast_flux_pulse(self, sequencer):
        for i in range(self.expt_cfg['nb_reps']):
            sequencer.new_sequence(self)

            #add flux to pulse
            for target in self.expt_cfg['target_qb']:
                #since we haven't been synching the flux channels, this should still be back at the beginning
                sequencer.append('ff_Q%s' % target,
                                 Square(max_amp=self.expt_cfg['ff_amp'], flat_len=self.expt_cfg['ff_time'],
                                        ramp_sigma_len=self.expt_cfg['ff_ramp_sigma_len'], cutoff_sigma=2, freq=0,
                                        phase=0))

            #COMPENSATION PULSE
            for target in self.expt_cfg['target_qb']:
                sequencer.append('ff_Q%s' % target,
                                 Square(max_amp=-self.expt_cfg['ff_amp'],
                                        flat_len=self.expt_cfg['ff_time'],
                                        ramp_sigma_len=self.expt_cfg['ff_ramp_sigma_len'], cutoff_sigma=2, freq=0,
                                        phase=0))

            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def rabi(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=['A','B'],time=500)

            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer,qb,len=rabi_len,amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'])
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def test_stab(self, sequencer):

        for stab_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=['A','B'],time=500)

            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer,qb,len=stab_len,amp = 1,phase=0,pulse_type="square")

            sequencer.append('stab_I', Square(max_amp=1, flat_len=stab_len,
                                                          ramp_sigma_len=0.001, cutoff_sigma=2, freq=0,
                                                          phase=0))

            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def rabi_amp(self, sequencer):

        for rabi_amp in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=['A','B'],time=500)

            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer,qb,len=self.expt_cfg["len"],amp = rabi_amp,phase=0,pulse_type=self.expt_cfg['pulse_type'])
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def rabi_pi(self, sequencer):

        pi_qbs = self.expt_cfg["pi_qbs"]
        rabi_qbs = self.expt_cfg["rabi_qbs"]

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=['A','B'],time=500)

            pulse_info = self.lattice_cfg["pulse_info"]

            for qb in pi_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.pi_q(sequencer,qb,phase=0,pulse_type=self.expt_cfg['pulse_type'])
            self.idle_q(sequencer, time=20)
            sequencer.sync_channels_time(self.channels)

            for i, qb in enumerate(rabi_qbs):
                pulse_info = self.lattice_cfg["pulse_info"]
                self.gen_q(sequencer, qb=qb, len=rabi_len,
                           amp=pulse_info[setup]["pi_amp"][qb], phase=0,
                           pulse_type=pulse_info["pulse_type"][qb])
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def rabi_chevron(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['len_start'], self.expt_cfg['len_stop'], self.expt_cfg['len_step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=['A','B'],time=500)

            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.gen_q(sequencer,qb,len=rabi_len,amp = self.expt_cfg['amp'],phase=0,pulse_type=self.expt_cfg['pulse_type'])
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)


    def t1(self, sequencer):

        if self.expt_cfg["t1_len_array"]=="auto":
            t1_len_array = np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])
        else:
            t1_len_array = self.expt_cfg["t1_len_array"]

        for t1_len in t1_len_array:
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=['A','B'],time=500)

            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.pi_q(sequencer,qb,pulse_type=self.pulse_info['pulse_type'][qb])
            self.idle_q(sequencer, time=t1_len)

            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ramsey(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.half_pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                self.idle_q(sequencer, qb=qb, time=ramsey_len)
                self.half_pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb],phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])

            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ramsey_pi(self, sequencer):
        pi_qbs = self.expt_cfg["pi_qbs"]
        ram_qbs = self.expt_cfg["ram_qbs"]

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            for qb in pi_qbs:
                self.pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                self.pi_q_ef(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                self.idle_q(sequencer, qb=qb, time=20)
            sequencer.sync_channels_time(self.channels)

            for qb in ram_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.half_pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                self.idle_q(sequencer, qb=qb, time=ramsey_len)
                self.half_pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb],phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])

            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def echo(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.half_pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                for echo_id in range(self.expt_cfg['echo_times']):
                    self.idle_q(sequencer, qb=qb, time=ramsey_len/(float(2*self.expt_cfg['echo_times'])))
                    if self.expt_cfg['cp']:
                        self.pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                    elif self.expt_cfg['cpmg']:
                        self.pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb],phase=np.pi/2.0)
                    self.idle_q(sequencer, qb=qb, time=ramsey_len / (float(2 * self.expt_cfg['echo_times'])))
                self.half_pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb],
                               phase=2 * np.pi * ramsey_len*self.expt_cfg['ramsey_freq'])

            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)




    def qb_tomo_test(self, sequencer):

        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
        for qb in self.on_qbs:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            for nb in range(self.expt_cfg["nb_pulses"]):
                self.gen_q(sequencer, qb=qb, len = self.pulse_info[setup]["half_pi_len"][qb], amp = self.expt_cfg["amp"], phase = self.expt_cfg["phase"], pulse_type = self.pulse_info['pulse_type'][qb])
                self.idle_q(sequencer, qb=qb, time=10)

            #self.half_pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
            self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["half_pi_len"][qb], amp=self.pulse_info[setup]['Y90_amp'][qb],phase=np.pi/2, pulse_type=self.pulse_info['pulse_type'][qb])
            self.readout_pxi(sequencer, self.on_rds)
        sequencer.end_sequence()

        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
        for qb in self.on_qbs:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            for nb in range(self.expt_cfg["nb_pulses"]):
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["half_pi_len"][qb], amp=self.expt_cfg["amp"],phase=self.expt_cfg["phase"], pulse_type=self.pulse_info['pulse_type'][qb])
                self.idle_q(sequencer, qb=qb, time=10)

            #self.half_pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb], phase = np.pi/2)
            self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["half_pi_len"][qb],amp=self.pulse_info[setup]['X90_amp'][qb], phase= 0, pulse_type=self.pulse_info['pulse_type'][qb])
            self.readout_pxi(sequencer, self.on_rds)
        sequencer.end_sequence()

        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
        for qb in self.on_qbs:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            for nb in range(self.expt_cfg["nb_pulses"]):
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["half_pi_len"][qb], amp=self.expt_cfg["amp"],phase=self.expt_cfg["phase"], pulse_type=self.pulse_info['pulse_type'][qb])
                self.idle_q(sequencer, qb=qb, time=10)
            self.readout_pxi(sequencer, self.on_rds)
        sequencer.end_sequence()

        return sequencer.complete(self)

    def qb_tomo_test_ff(self, sequencer):

        #### X! ###############
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)

        # prep state
        for qb in self.on_qbs:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            for nb in range(self.expt_cfg["nb_pulses"]):
                self.gen_q(sequencer, qb=qb, len = self.pulse_info[setup]["half_pi_len"][qb], amp = self.expt_cfg["amp"], phase = self.expt_cfg["phase"], pulse_type = self.pulse_info['pulse_type'][qb])
                self.idle_q(sequencer, qb=qb, time=10)


        # add flux to pulse
        sequencer.sync_channels_time(self.channels)
        ff_len = np.asarray(self.expt_cfg["ff_len"])
        self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                      flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
        sequencer.sync_channels_time(self.channels)
        self.idle_q(sequencer, time=self.lattice_cfg['ff_info']['ff_settling_time'])
        sequencer.sync_channels_time(self.channels)

        # Tomography rotation
        for qb in self.on_rds:
            self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["half_pi_len"][qb], amp=self.pulse_info[setup]['Y90_amp'][qb],phase=np.pi/2, pulse_type=self.pulse_info['pulse_type'][qb])
            self.readout_pxi(sequencer, self.on_rds)

        # compensation
        sequencer.sync_channels_time(self.channels)
        self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                      flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)


        sequencer.end_sequence()

        #### Y! ###############
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)

        # prep state
        for qb in self.on_qbs:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            for nb in range(self.expt_cfg["nb_pulses"]):
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["half_pi_len"][qb], amp=self.expt_cfg["amp"],
                           phase=self.expt_cfg["phase"], pulse_type=self.pulse_info['pulse_type'][qb])
                self.idle_q(sequencer, qb=qb, time=10)

        # add flux to pulse
        sequencer.sync_channels_time(self.channels)
        ff_len = np.asarray(self.expt_cfg["ff_len"])
        self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                      flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
        sequencer.sync_channels_time(self.channels)
        self.idle_q(sequencer, time=self.lattice_cfg['ff_info']['ff_settling_time'])
        sequencer.sync_channels_time(self.channels)

        # Tomography rotation
        for qb in self.on_rds:
            self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["half_pi_len"][qb],
                       amp=self.pulse_info[setup]['X90_amp'][qb], phase=0,
                       pulse_type=self.pulse_info['pulse_type'][qb])
            self.readout_pxi(sequencer, self.on_rds)

        # compensation
        sequencer.sync_channels_time(self.channels)
        self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                      flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)

        sequencer.end_sequence()

        #### Z! ###############
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)

        # prep state
        for qb in self.on_qbs:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            for nb in range(self.expt_cfg["nb_pulses"]):
                self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["half_pi_len"][qb], amp=self.expt_cfg["amp"],
                           phase=self.expt_cfg["phase"], pulse_type=self.pulse_info['pulse_type'][qb])
                self.idle_q(sequencer, qb=qb, time=10)

        # add flux to pulse
        sequencer.sync_channels_time(self.channels)
        ff_len = np.asarray(self.expt_cfg["ff_len"])
        self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                      flux_vec=self.expt_cfg["ff_vec"], flip_amp=False)
        sequencer.sync_channels_time(self.channels)
        self.idle_q(sequencer, time=self.lattice_cfg['ff_info']['ff_settling_time'])
        sequencer.sync_channels_time(self.channels)

        # Tomography rotation
        for qb in self.on_rds:
            self.readout_pxi(sequencer, self.on_rds)

        # compensation
        sequencer.sync_channels_time(self.channels)
        self.ff_pulse(sequencer, ff_len, pulse_type=self.expt_cfg["ff_pulse_type"],
                      flux_vec=self.expt_cfg["ff_vec"], flip_amp=True)

        sequencer.end_sequence()

        return sequencer.complete(self)

    def ff_qb_tomo(self, sequencer):
        ###################
        # Tomography after melting
        ###################

        FF_vec_orig = copy.copy(self.expt_cfg["ff_vec_list"][-1])
        for det in self.expt_cfg["detfactor_list"]:
            self.expt_cfg["ff_vec_list"][-1] = list(np.array(FF_vec_orig)*det)
            for gate in ["X", "Y", "Z"]:
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)

                # prep state
                for qb in self.expt_cfg["Mott_qbs"]:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
                               amp=self.pulse_info[setup]["pi_amp"][qb], phase=0,
                               pulse_type=self.pulse_info["pulse_type"][qb])


                # add flux to pulse
                sequencer.sync_channels_time(self.channels)
                ff_vec_list = self.expt_cfg["ff_vec_list"]
                delay_list = self.expt_cfg["delay_list"]
                ff_len_list = self.expt_cfg["ff_len_list"]
                pulse_type_list = self.expt_cfg["ff_pulse_type_list"]
                self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = pulse_type_list,
                                       flux_vec_list = ff_vec_list, delay_list = delay_list, freq_list=[0]*7, flip_amp=False, modulate_qb=self.expt_cfg["modulate_qb"])

                sequencer.sync_channels_time(self.channels)
                self.idle_q(sequencer, time=self.lattice_cfg['ff_info']['ff_settling_time'])
                sequencer.sync_channels_time(self.channels)

                # Tomography rotation
                for qb in self.on_rds:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    if gate =="X":
                        self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["half_pi_len"][qb], amp=self.pulse_info[setup]['Y90_amp'][qb],phase=np.pi/2, pulse_type=self.pulse_info['pulse_type'][qb])
                    elif gate=="Y":
                        self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["half_pi_len"][qb],
                                   amp=self.pulse_info[setup]['X90_amp'][qb], phase=0,
                                   pulse_type=self.pulse_info['pulse_type'][qb])
                    elif gate=="Z":
                        pass

                    self.readout_pxi(sequencer, self.on_rds)

                # compensation
                sequencer.sync_channels_time(self.channels)
                self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=pulse_type_list,
                                       flux_vec_list=ff_vec_list, delay_list=delay_list, freq_list=[0]*7, flip_amp=True, modulate_qb=self.expt_cfg["modulate_qb"])


                sequencer.end_sequence()

        if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
            for rd_qb in self.on_rds:
                self.pi_cal_sequence(sequencer, pi_qbs = [rd_qb], pi_rds = self.on_rds)
        return sequencer.complete(self)

    def wurst_sweep(self,sequencer):
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
        for qb in self.on_qbs:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            iq_freq = self.lattice_cfg['pulse_info'][setup]['iq_freq'][qb]
            # iq_freq = 0
            add_freq = 0
            phase = 0

            sequencer.append('charge%s_I' % setup,
                             WURST20_I(max_amp=self.expt_cfg['maxamp'],
                                       bandwidth = self.expt_cfg['pulse_bw'],
                                       t_length=self.expt_cfg['pulse_length'],
                                       freq=iq_freq + add_freq,
                                       phase=phase))
            sequencer.append('charge%s_Q' % setup,
                             WURST20_Q(max_amp=self.expt_cfg['maxamp'],
                                       bandwidth=self.expt_cfg['pulse_bw'],
                                       t_length=self.expt_cfg['pulse_length'],
                                       freq=iq_freq + add_freq,
                                       phase=phase,
                                       phase_t0 = self.lattice_cfg['pulse_info'][setup]['Q_phase'][qb]))
        sequencer.sync_channels_time(self.channels)
        for qb in self.on_rds:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            self.readout_pxi(sequencer, self.on_rds)
        sequencer.end_sequence()
        return sequencer.complete(self)

    def wurst_sweep_amp(self,sequencer):
        for qb in self.expt_cfg["on_qbs"]:
            for amp in self.expt_cfg["ampvec"]:
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                iq_freq = self.lattice_cfg['pulse_info'][setup]['iq_freq'][qb]
                # iq_freq = 0
                add_freq = 0
                phase = 0

                sequencer.append('charge%s_I' % setup,
                                 WURST20_I(max_amp=amp,
                                           bandwidth = self.expt_cfg['pulse_bw'],
                                           t_length=self.expt_cfg['pulse_length'],
                                           freq=iq_freq + add_freq,
                                           phase=phase))
                sequencer.append('charge%s_Q' % setup,
                                 WURST20_Q(max_amp=amp,
                                           bandwidth=self.expt_cfg['pulse_bw'],
                                           t_length=self.expt_cfg['pulse_length'],
                                           freq=iq_freq + add_freq,
                                           phase=phase,
                                           phase_t0 = self.lattice_cfg['pulse_info'][setup]['Q_phase'][qb]))
                sequencer.sync_channels_time(self.channels)
                for qb in self.on_rds:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.readout_pxi(sequencer, self.on_rds)
                sequencer.end_sequence()
        return sequencer.complete(self)


    def wurst_x_sweep(self,sequencer):
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
        for qb in self.on_qbs:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            iq_freq = self.lattice_cfg['pulse_info'][setup]['iq_freq'][qb]
            # iq_freq = 0
            add_freq = 0
            phase = 0

            sequencer.append('charge%s_I' % setup,
                             WURST20_I_X(max_amp=self.expt_cfg['maxamp'],
                                       bandwidth = self.expt_cfg['pulse_bw'],
                                       t_length=self.expt_cfg['pulse_length'],
                                       freq=iq_freq + add_freq,
                                       phase=phase))
            sequencer.append('charge%s_Q' % setup,
                             WURST20_Q_X(max_amp=self.expt_cfg['maxamp'],
                                       bandwidth=self.expt_cfg['pulse_bw'],
                                       t_length=self.expt_cfg['pulse_length'],
                                       freq=iq_freq + add_freq,
                                       phase=phase,
                                       phase_t0 = self.lattice_cfg['pulse_info'][setup]['Q_phase'][qb]))
        sequencer.sync_channels_time(self.channels)
        for qb in self.on_rds:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            self.readout_pxi(sequencer, self.on_rds)
        sequencer.end_sequence()
        return sequencer.complete(self)

    def wurst_y_sweep(self,sequencer):
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
        for qb in self.on_qbs:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            iq_freq = self.lattice_cfg['pulse_info'][setup]['iq_freq'][qb]
            # iq_freq = 0
            add_freq = 0
            phase = 0

            sequencer.append('charge%s_I' % setup,
                             WURST20_I_X(max_amp=self.expt_cfg['maxamp'],
                                       bandwidth = self.expt_cfg['pulse_bw'],
                                       t_length=self.expt_cfg['pulse_length'],
                                       freq=iq_freq + add_freq,
                                       phase=phase))
            sequencer.append('charge%s_Q' % setup,
                             WURST20_Q_X(max_amp=self.expt_cfg['maxamp'],
                                       bandwidth=self.expt_cfg['pulse_bw'],
                                       t_length=self.expt_cfg['pulse_length'],
                                       freq=iq_freq + add_freq,
                                       phase=phase,
                                       phase_t0 = self.lattice_cfg['pulse_info'][setup]['Q_phase'][qb]+ (3.14 / 4)) )
        sequencer.sync_channels_time(self.channels)
        for qb in self.on_rds:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            self.readout_pxi(sequencer, self.on_rds)
        sequencer.end_sequence()
        return sequencer.complete(self)

    def bir_4_sweep(self,sequencer):
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
        for qb in self.on_qbs:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            iq_freq = self.lattice_cfg['pulse_info'][setup]['iq_freq'][qb]
            # iq_freq = 0
            add_freq = 0
            phase = 0

            sequencer.append('charge%s_I' % setup,
                             BIR_4(max_amp=self.expt_cfg['maxamp'],
                                       bandwidth = self.expt_cfg['pulse_bw'],
                                       t_length=self.expt_cfg['pulse_length'],
                                       theta = self.expt_cfg['theta'],
                                       freq=iq_freq + add_freq,
                                       phase=phase))
            sequencer.append('charge%s_Q' % setup,
                             BIR_4(max_amp=self.expt_cfg['maxamp'],
                                       bandwidth=self.expt_cfg['pulse_bw'],
                                       t_length=self.expt_cfg['pulse_length'],
                                       theta=self.expt_cfg['theta'],
                                       freq=iq_freq + add_freq,
                                       phase=phase,
                                       phase_t0 = self.lattice_cfg['pulse_info'][setup]['Q_phase'][qb]))

        sequencer.sync_channels_time(self.channels)
        for qb in self.on_rds:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            self.readout_pxi(sequencer, self.on_rds)
        sequencer.end_sequence()
        return sequencer.complete(self)

    def bir_4_sweep_amp(self,sequencer):
        for qb in self.on_qbs:
            for amp in self.expt_cfg['amplist']:
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                iq_freq = self.lattice_cfg['pulse_info'][setup]['iq_freq'][qb]
                # iq_freq = 0
                add_freq = 0
                phase = 0

                sequencer.append('charge%s_I' % setup,
                                 BIR_4(max_amp=amp,
                                           bandwidth = self.expt_cfg['pulse_bw'],
                                           t_length=self.expt_cfg['pulse_length'],
                                           theta = self.expt_cfg['theta'],
                                           freq=iq_freq + add_freq,
                                           phase=phase))
                sequencer.append('charge%s_Q' % setup,
                                 BIR_4(max_amp=amp,
                                           bandwidth=self.expt_cfg['pulse_bw'],
                                           t_length=self.expt_cfg['pulse_length'],
                                           theta=self.expt_cfg['theta'],
                                           freq=iq_freq + add_freq,
                                           phase=phase,
                                           phase_t0 = self.lattice_cfg['pulse_info'][setup]['Q_phase'][qb]))

                sequencer.sync_channels_time(self.channels)
                for qb in self.on_rds:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.readout_pxi(sequencer, self.on_rds)
                sequencer.end_sequence()
        return sequencer.complete(self)

    def bir_4_hyp_sweep(self,sequencer):
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
        for qb in self.on_qbs:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            iq_freq = self.lattice_cfg['pulse_info'][setup]['iq_freq'][qb]
            # iq_freq = 0
            add_freq = 0
            phase = 0

            sequencer.append('charge%s_I' % setup,
                             BIR_4_hyp(max_amp=self.expt_cfg['maxamp'],
                                       bandwidth = self.expt_cfg['pulse_bw'],
                                       t_length=self.expt_cfg['pulse_length'],
                                       theta = self.expt_cfg['theta'],
                                       kappa = self.expt_cfg['kappa'],
                                       beta = self.expt_cfg['beta'],
                                       freq=iq_freq + add_freq,
                                       phase=phase))
            sequencer.append('charge%s_Q' % setup,
                             BIR_4_hyp(max_amp=self.expt_cfg['maxamp'],
                                       bandwidth=self.expt_cfg['pulse_bw'],
                                       t_length=self.expt_cfg['pulse_length'],
                                       theta=self.expt_cfg['theta'],
                                       kappa=self.expt_cfg['kappa'],
                                       beta=self.expt_cfg['beta'],
                                       freq=iq_freq + add_freq,
                                       phase=phase,
                                       phase_t0 = self.lattice_cfg['pulse_info'][setup]['Q_phase'][qb]))

        sequencer.sync_channels_time(self.channels)
        for qb in self.on_rds:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            self.readout_pxi(sequencer, self.on_rds)
        sequencer.end_sequence()
        return sequencer.complete(self)

    def bir_4_hyp_sweep_amp_theta(self,sequencer):
        for qb in self.on_qbs:
            for theta in self.expt_cfg['thetalist']:
                for amp in self.expt_cfg['amplist']:
                    sequencer.new_sequence(self)
                    self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    iq_freq = self.lattice_cfg['pulse_info'][setup]['iq_freq'][qb]
                    # iq_freq = 0
                    add_freq = 0
                    phase = 0
                    sequencer.append('charge%s_I' % setup,
                                     BIR_4_hyp(max_amp=amp,
                                               bandwidth = self.expt_cfg['pulse_bw'],
                                               t_length=self.expt_cfg['pulse_length'],
                                               theta = theta,
                                               kappa=self.expt_cfg['kappa'],
                                               beta=self.expt_cfg['beta'],
                                               freq=iq_freq + add_freq,
                                               phase=phase))
                    sequencer.append('charge%s_Q' % setup,
                                     BIR_4_hyp(max_amp=amp,
                                               bandwidth=self.expt_cfg['pulse_bw'],
                                               t_length=self.expt_cfg['pulse_length'],
                                               theta=theta,
                                               kappa=self.expt_cfg['kappa'],
                                               beta=self.expt_cfg['beta'],
                                               freq=iq_freq + add_freq,
                                               phase=phase,
                                               phase_t0 = self.lattice_cfg['pulse_info'][setup]['Q_phase'][qb]))

                    sequencer.sync_channels_time(self.channels)
                    for qb in self.on_rds:
                        setup = self.lattice_cfg["qubit"]["setup"][qb]
                        self.readout_pxi(sequencer, self.on_rds)
                    sequencer.end_sequence()
        return sequencer.complete(self)

    def bir_4_sweep4pulses(self,sequencer):
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
        for qb in self.on_qbs:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            iq_freq = self.lattice_cfg['pulse_info'][setup]['iq_freq'][qb]
            iq_freq = 0
            add_freq = 0
            phase = 0

            sequencer.append('charge%s_I' % setup,
                             BIR_4(max_amp=self.expt_cfg['maxamp'],
                                       bandwidth = self.expt_cfg['pulse_bw'],
                                       t_length=self.expt_cfg['pulse_length'],
                                       theta = self.expt_cfg['thetaarray'][0],
                                       freq=iq_freq + add_freq,
                                       phase=phase))
            sequencer.append('charge%s_I' % setup,
                             BIR_4(max_amp=self.expt_cfg['maxamp'],
                                   bandwidth=self.expt_cfg['pulse_bw'],
                                   t_length=self.expt_cfg['pulse_length'],
                                   theta=self.expt_cfg['thetaarray'][1],
                                   freq=iq_freq + add_freq,
                                   phase=phase))
            sequencer.append('charge%s_I' % setup,
                             BIR_4(max_amp=self.expt_cfg['maxamp'],
                                   bandwidth=self.expt_cfg['pulse_bw'],
                                   t_length=self.expt_cfg['pulse_length'],
                                   theta=self.expt_cfg['thetaarray'][2],
                                   freq=iq_freq + add_freq,
                                   phase=phase))
            sequencer.append('charge%s_I' % setup,
                             BIR_4(max_amp=self.expt_cfg['maxamp'],
                                   bandwidth=self.expt_cfg['pulse_bw'],
                                   t_length=self.expt_cfg['pulse_length'],
                                   theta=self.expt_cfg['thetaarray'][3],
                                   freq=iq_freq + add_freq,
                                   phase=phase))
            sequencer.append('charge%s_Q' % setup,
                             BIR_4(max_amp=self.expt_cfg['maxamp'],
                                       bandwidth=self.expt_cfg['pulse_bw'],
                                       t_length=self.expt_cfg['pulse_length'],
                                       theta=self.expt_cfg['thetaarray'][0],
                                       freq=iq_freq + add_freq,
                                       phase=phase,
                                       phase_t0 = self.lattice_cfg['pulse_info'][setup]['Q_phase'][qb]))
            sequencer.append('charge%s_Q' % setup,
                             BIR_4(max_amp=self.expt_cfg['maxamp'],
                                   bandwidth=self.expt_cfg['pulse_bw'],
                                   t_length=self.expt_cfg['pulse_length'],
                                   theta=self.expt_cfg['thetaarray'][1],
                                   freq=iq_freq + add_freq,
                                   phase=phase,
                                   phase_t0=self.lattice_cfg['pulse_info'][setup]['Q_phase'][qb]))
            sequencer.append('charge%s_Q' % setup,
                             BIR_4(max_amp=self.expt_cfg['maxamp'],
                                   bandwidth=self.expt_cfg['pulse_bw'],
                                   t_length=self.expt_cfg['pulse_length'],
                                   theta=self.expt_cfg['thetaarray'][2],
                                   freq=iq_freq + add_freq,
                                   phase=phase,
                                   phase_t0=self.lattice_cfg['pulse_info'][setup]['Q_phase'][qb]))
            sequencer.append('charge%s_Q' % setup,
                             BIR_4(max_amp=self.expt_cfg['maxamp'],
                                   bandwidth=self.expt_cfg['pulse_bw'],
                                   t_length=self.expt_cfg['pulse_length'],
                                   theta=self.expt_cfg['thetaarray'][3],
                                   freq=iq_freq + add_freq,
                                   phase=phase,
                                   phase_t0=self.lattice_cfg['pulse_info'][setup]['Q_phase'][qb]))

        sequencer.sync_channels_time(self.channels)
        for qb in self.on_rds:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            self.readout_pxi(sequencer, self.on_rds)
        sequencer.end_sequence()
        return sequencer.complete(self)

    def bir_4_wurst(self,sequencer):
        sequencer.new_sequence(self)
        self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
        for qb in self.on_qbs:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            iq_freq = self.lattice_cfg['pulse_info'][setup]['iq_freq'][qb]
            iq_freq = 0
            add_freq = 0
            phase = 0

            sequencer.append('charge%s_I' % setup,
                             BIR_4_wurst(max_amp=self.expt_cfg['maxamp'],
                                       bandwidth = self.expt_cfg['pulse_bw'],
                                       t_length=self.expt_cfg['pulse_length'],
                                       freq=iq_freq + add_freq,
                                       phase=phase))
            sequencer.append('charge%s_Q' % setup,
                             BIR_4_wurst(max_amp=self.expt_cfg['maxamp'],
                                       bandwidth=self.expt_cfg['pulse_bw'],
                                       t_length=self.expt_cfg['pulse_length'],
                                       freq=iq_freq + add_freq,
                                       phase=phase,
                                       phase_t0 = self.lattice_cfg['pulse_info'][setup]['Q_phase'][qb]))
        sequencer.sync_channels_time(self.channels)
        for qb in self.on_rds:
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            self.readout_pxi(sequencer, self.on_rds)
        sequencer.end_sequence()
        return sequencer.complete(self)

    def bb1_sweep_amp_theta(self,sequencer,qb = 0,phase = 0):
        for qb in self.on_qbs:
            for theta_angle in self.expt_cfg['thetalist']:
                for amp in self.expt_cfg['amplist']:
                    sequencer.new_sequence(self)
                    self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    theta = (np.pi/180)*theta_angle
                    phi = np.arccos(-1*theta / (4*np.pi))
                    phase = phi
                    if self.expt_cfg['pulse_type'] == "square":
                        sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                      flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                                      ramp_sigma_len=0.001, cutoff_sigma=2,
                                                                      freq=self.pulse_info[setup]['iq_freq'][qb], phase=phase))
                        sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                      flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                                      ramp_sigma_len=0.001, cutoff_sigma=2,
                                                                      freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                      phase=phase + self.pulse_info[setup]['Q_phase'][qb]))
                    elif self.expt_cfg['pulse_type'] == "gauss":
                        sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                     sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                                     cutoff_sigma=2,
                                                                     freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                     phase=phase))
                        sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                     sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                                     cutoff_sigma=2,
                                                                     freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                     phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                         qb]))
                    phase = 3*phi
                    if self.expt_cfg['pulse_type'] == "square":
                        sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                      flat_len=2*self.pulse_info[setup]['pi_len'][qb],
                                                                      ramp_sigma_len=0.001, cutoff_sigma=2,
                                                                      freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                      phase=phase))
                        sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                      flat_len=2*self.pulse_info[setup]['pi_len'][qb],
                                                                      ramp_sigma_len=0.001, cutoff_sigma=2,
                                                                      freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                      phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                          qb]))
                    elif self.expt_cfg['pulse_type'] == "gauss":
                        sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                     sigma_len=2*self.pulse_info[setup]['pi_len'][qb],
                                                                     cutoff_sigma=2,
                                                                     freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                     phase=phase))
                        sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                     sigma_len=2*self.pulse_info[setup]['pi_len'][qb],
                                                                     cutoff_sigma=2,
                                                                     freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                     phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                         qb]))
                    phase = phi
                    if self.expt_cfg['pulse_type'] == "square":
                        sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                      flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                                      ramp_sigma_len=0.001, cutoff_sigma=2,
                                                                      freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                      phase=phase))
                        sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                      flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                                      ramp_sigma_len=0.001, cutoff_sigma=2,
                                                                      freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                      phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                          qb]))
                    elif self.expt_cfg['pulse_type'] == "gauss":
                        sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                     sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                                     cutoff_sigma=2,
                                                                     freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                     phase=phase))
                        sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                     sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                                     cutoff_sigma=2,
                                                                     freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                     phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                         qb]))

                    sequencer.sync_channels_time(self.channels)
                    for qb in self.on_rds:
                        setup = self.lattice_cfg["qubit"]["setup"][qb]
                        self.readout_pxi(sequencer, self.on_rds)
                    sequencer.end_sequence()
        return sequencer.complete(self)

    # def corpse(self,sequencer,qb = 0,phase = 0,pulse_type = 'square'):
    #     setup = self.lattice_cfg["qubit"]["setup"][qb]
    #     theta = (np.pi / 180) * self.expt_cfg['theta']
    #     phi = np.arccos(-1 * theta / (4 * np.pi))
    #     phase = phi
    #     sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
    #                                                   flat_len=self.pulse_info[setup]['pi_len'][qb],
    #                                                   ramp_sigma_len=0.001, cutoff_sigma=2,
    #                                                   freq=self.pulse_info[setup]['iq_freq'][qb], phase=phase))
    #     sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
    #                                                   flat_len=self.pulse_info[setup]['pi_len'][qb],
    #                                                   ramp_sigma_len=0.001, cutoff_sigma=2,
    #                                                   freq=self.pulse_info[setup]['iq_freq'][qb],
    #                                                   phase=phase + self.pulse_info[setup]['Q_phase'][qb]))
    #     phase = 3 * phi
    #     sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
    #                                                   flat_len=2 * self.pulse_info[setup]['pi_len'][qb],
    #                                                   ramp_sigma_len=0.001, cutoff_sigma=2,
    #                                                   freq=self.pulse_info[setup]['iq_freq'][qb], phase=phase))
    #     sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
    #                                                   flat_len=2 * self.pulse_info[setup]['pi_len'][qb],
    #                                                   ramp_sigma_len=0.001, cutoff_sigma=2,
    #                                                   freq=self.pulse_info[setup]['iq_freq'][qb],
    #                                                   phase=phase + self.pulse_info[setup]['Q_phase'][qb]))
    #     phase = phi
    #     sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
    #                                                   flat_len=self.pulse_info[setup]['pi_len'][qb],
    #                                                   ramp_sigma_len=0.001, cutoff_sigma=2,
    #                                                   freq=self.pulse_info[setup]['iq_freq'][qb], phase=phase))
    #     sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
    #                                                   flat_len=self.pulse_info[setup]['pi_len'][qb],
    #                                                   ramp_sigma_len=0.001, cutoff_sigma=2,
    #                                                   freq=self.pulse_info[setup]['iq_freq'][qb],
    #                                                   phase=phase + self.pulse_info[setup]['Q_phase'][qb]))

    def knill_sweep_amp(self,sequencer):
        for qb in self.on_qbs:
            for amp in self.expt_cfg['amplist']:
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                phase = np.pi/6
                if self.expt_cfg['pulse_type'] == "square":
                    sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                  flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                                  ramp_sigma_len=0.001, cutoff_sigma=2,
                                                                  freq=self.pulse_info[setup]['iq_freq'][qb], phase=phase))
                    sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                  flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                                  ramp_sigma_len=0.001, cutoff_sigma=2,
                                                                  freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                  phase=phase + self.pulse_info[setup]['Q_phase'][qb]))
                elif self.expt_cfg['pulse_type'] == "gauss":
                    sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                 sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                                 cutoff_sigma=2,
                                                                 freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                 phase=phase))
                    sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                 sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                                 cutoff_sigma=2,
                                                                 freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                 phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                     qb]))
                phase = 0
                if self.expt_cfg['pulse_type'] == "square":
                    sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                  flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                                  ramp_sigma_len=0.001, cutoff_sigma=2,
                                                                  freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                  phase=phase))
                    sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                  flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                                  ramp_sigma_len=0.001, cutoff_sigma=2,
                                                                  freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                  phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                      qb]))
                elif self.expt_cfg['pulse_type'] == "gauss":
                    sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                 sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                                 cutoff_sigma=2,
                                                                 freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                 phase=phase))
                    sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                 sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                                 cutoff_sigma=2,
                                                                 freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                 phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                     qb]))
                phase = np.pi/2
                if self.expt_cfg['pulse_type'] == "square":
                    sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                  flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                                  ramp_sigma_len=0.001, cutoff_sigma=2,
                                                                  freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                  phase=phase))
                    sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                  flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                                  ramp_sigma_len=0.001, cutoff_sigma=2,
                                                                  freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                  phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                      qb]))
                elif self.expt_cfg['pulse_type'] == "gauss":
                    sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                 sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                                 cutoff_sigma=2,
                                                                 freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                 phase=phase))
                    sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                 sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                                 cutoff_sigma=2,
                                                                 freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                 phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                     qb]))
                phase = 0
                if self.expt_cfg['pulse_type'] == "square":
                    sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                  flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                                  ramp_sigma_len=0.001, cutoff_sigma=2,
                                                                  freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                  phase=phase))
                    sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                  flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                                  ramp_sigma_len=0.001, cutoff_sigma=2,
                                                                  freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                  phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                      qb]))
                elif self.expt_cfg['pulse_type'] == "gauss":
                    sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                 sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                                 cutoff_sigma=2,
                                                                 freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                 phase=phase))
                    sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                 sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                                 cutoff_sigma=2,
                                                                 freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                 phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                     qb]))
                phase = np.pi / 6
                if self.expt_cfg['pulse_type'] == "square":
                    sequencer.append('charge%s_I' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                  flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                                  ramp_sigma_len=0.001, cutoff_sigma=2,
                                                                  freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                  phase=phase))
                    sequencer.append('charge%s_Q' % setup, Square(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                  flat_len=self.pulse_info[setup]['pi_len'][qb],
                                                                  ramp_sigma_len=0.001, cutoff_sigma=2,
                                                                  freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                  phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                      qb]))
                elif self.expt_cfg['pulse_type'] == "gauss":
                    sequencer.append('charge%s_I' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                 sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                                 cutoff_sigma=2,
                                                                 freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                 phase=phase))
                    sequencer.append('charge%s_Q' % setup, Gauss(max_amp=self.pulse_info[setup]['pi_amp'][qb],
                                                                 sigma_len=self.pulse_info[setup]['pi_len'][qb],
                                                                 cutoff_sigma=2,
                                                                 freq=self.pulse_info[setup]['iq_freq'][qb],
                                                                 phase=phase + self.pulse_info[setup]['Q_phase'][
                                                                     qb]))
                sequencer.sync_channels_time(self.channels)
                for qb in self.on_rds:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.readout_pxi(sequencer, self.on_rds)
                sequencer.end_sequence()
        return sequencer.complete(self)


    def t1rho(self,sequencer):

        for t1rho_len in np.arange(self.expt_cfg['start'],self.expt_cfg['stop'],self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.half_pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                #self.idle_q(sequencer, time= self.expt_cfg['ramsey_zero_cross'])
                #self.idle_q(sequencer, time=self.pulse_info[setup]['half_pi_len'])
                self.gen_q(sequencer, qb, len=t1rho_len, amp=self.expt_cfg['amp'], phase=np.pi/2,
                           pulse_type=self.expt_cfg['pulse_type'])
                #self.idle_q(sequencer, time=self.expt_cfg['ramsey_zero_cross'])
                #self.idle_q(sequencer, time=self.pulse_info[setup]['half_pi_len'])
                self.half_pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb],
                               phase=0)


                self.readout_pxi(sequencer, self.on_rds)
                sequencer.end_sequence()

        return sequencer.complete(self, plot=True)


    def pulse_probe_ef_iq(self, sequencer):

        for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                self.gen_q(sequencer, qb, len=self.expt_cfg['pulse_length'],amp = self.pulse_info[setup]['pi_ef_amp'][qb],phase=0,pulse_type=self.pulse_info['pulse_type'][qb],add_freq=dfreq+self.lattice_cfg['qubit']['anharmonicity'][qb])
            self.readout_pxi(sequencer, self.on_rds,overlap=False)

            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ef_rabi(self, sequencer):

        for rabi_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer,on_qubits=['A','B'],time=500)

            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                self.gen_q(sequencer,qb,len=rabi_len,amp = self.expt_cfg['amp'],phase=0,pulse_type=self.pulse_info['ef_pulse_type'][qb],add_freq = self.lattice_cfg['qubit']['anharmonicity'][qb])
                if self.expt_cfg['pi_calibration']:
                    self.pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                    self.pi_q_ef(sequencer, qb, pulse_type=self.pulse_info['ef_pulse_type'][qb])

            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ef_t1(self, sequencer):
        # t1 for the e and f level

        for ef_t1_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]

                self.pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                self.pi_q_ef(sequencer, qb, pulse_type=self.pulse_info['ef_pulse_type'][qb])
                self.idle_q(sequencer, time=ef_t1_len)

            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ef_ramsey(self, sequencer):
        # ef ramsey sequences

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                self.half_pi_q_ef(sequencer, qb, pulse_type=self.pulse_info['ef_pulse_type'][qb])
                self.idle_q(sequencer, qb=qb, time=ramsey_len)
                self.half_pi_q_ef(sequencer, qb, pulse_type=self.pulse_info['ef_pulse_type'][qb],
                               phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])
                if self.expt_cfg['pi_calibration']:
                    self.pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                    self.pi_q_ef(sequencer, qb, pulse_type=self.pulse_info['ef_pulse_type'][qb])

            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def ef_echo(self, sequencer):

        for ramsey_len in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)

            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                self.pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                self.half_pi_q_ef(sequencer, qb, pulse_type=self.pulse_info['ef_pulse_type'][qb])
                for echo_id in range(self.expt_cfg['echo_times']):
                    self.idle_q(sequencer, qb=qb, time=ramsey_len / (float(2 * self.expt_cfg['echo_times'])))
                    if self.expt_cfg['cp']:
                        self.pi_q_ef(sequencer, qb, pulse_type=self.pulse_info['ef_pulse_type'][qb])
                    elif self.expt_cfg['cpmg']:
                        self.pi_q_ef(sequencer, qb, pulse_type=self.pulse_info['ef_pulse_type'][qb],
                                  phase=np.pi / 2.0)
                    self.idle_q(sequencer, qb=qb, time=ramsey_len / (float(2 * self.expt_cfg['echo_times'])))
                self.half_pi_q_ef(sequencer, qb, pulse_type=self.pulse_info['ef_pulse_type'][qb],
                               phase=2 * np.pi * ramsey_len * self.expt_cfg['ramsey_freq'])

            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def histogram(self, sequencer):
        # vacuum rabi sequences
        for ii in range(self.expt_cfg['num_seq_sets']):

            for qb in self.on_qbs:
                setup = self.lattice_cfg["qubit"]["setup"][qb]

                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A","B"][0], time=500)
                self.readout_pxi(sequencer,self.on_rds)
                sequencer.end_sequence()

                # with pi pulse (e state)
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
                for qb in self.on_qbs:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                self.readout_pxi(sequencer, self.on_rds)
                sequencer.end_sequence()

                # with pi pulse and ef pi pulse (f state)
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
                for qb in self.on_qbs:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    self.pi_q(sequencer, qb, pulse_type=self.pulse_info['pulse_type'][qb])
                    self.pi_q_ef(sequencer, qb, pulse_type=self.pulse_info['ef_pulse_type'][qb])
                self.readout_pxi(sequencer, self.on_rds)

                sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def histogram_crosstalk(self, sequencer):
        # vacuum rabi sequences
        for ii in range(self.expt_cfg['num_seq_sets']):

            #g
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

            # with pi pulse for actual rd qb(e state)
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            qb = self.on_rds[0]
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            pulse_info = self.lattice_cfg["pulse_info"]
            self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                       amp=pulse_info[setup]["pi_amp"][qb], 
                       pulse_type=pulse_info["pulse_type"][qb])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

            # with pi pulse (e state)
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            qb = self.expt_cfg["cross_qb"]
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            pulse_info = self.lattice_cfg["pulse_info"]
            self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                       amp=pulse_info[setup]["pi_amp"][qb],
                       pulse_type=pulse_info["pulse_type"][qb])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

            # with two pi pulses
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            for qb in [self.on_rds[0], self.expt_cfg["cross_qb"]]:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                pulse_info = self.lattice_cfg["pulse_info"]
                self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                           amp=pulse_info[setup]["pi_amp"][qb],
                           pulse_type=pulse_info["pulse_type"][qb])
                sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer,self.on_rds)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def histogram_crosstalk_combo(self, sequencer):
        # vacuum rabi sequences
        for ii in range(self.expt_cfg['num_seq_sets']):

            #g
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

            # with pi pulse for actual rd qb(e state)
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            qb = self.on_rds[0]
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            pulse_info = self.lattice_cfg["pulse_info"]
            self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                       amp=pulse_info[setup]["pi_amp"][qb],
                       pulse_type=pulse_info["pulse_type"][qb])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

            # with pi pulse (e state)
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            qb = self.expt_cfg["cross_qb"]
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            pulse_info = self.lattice_cfg["pulse_info"]
            self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                       amp=pulse_info[setup]["pi_amp"][qb],
                       pulse_type=pulse_info["pulse_type"][qb])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

            # with two pi pulses
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            for qb in [self.on_rds[0], self.expt_cfg["cross_qb"]]:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                pulse_info = self.lattice_cfg["pulse_info"]
                self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                           amp=pulse_info[setup]["pi_amp"][qb],
                           pulse_type=pulse_info["pulse_type"][qb])
                sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer,self.on_rds)
            sequencer.end_sequence()

            # with two pi pulses flip order
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            for qb in [self.expt_cfg["cross_qb"], self.on_rds[0]]:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                pulse_info = self.lattice_cfg["pulse_info"]
                self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                           amp=pulse_info[setup]["pi_amp"][qb],
                           pulse_type=pulse_info["pulse_type"][qb])
                sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer,self.on_rds)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def histogram_crosstalk_combo_ef(self, sequencer):
        # vacuum rabi sequences
        for ii in range(self.expt_cfg['num_seq_sets']):

            # g
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

            # with pi pulse for actual rd qb(e state)
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            qb = self.on_rds[0]
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            pulse_info = self.lattice_cfg["pulse_info"]
            self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                       amp=pulse_info[setup]["pi_amp"][qb],
                       pulse_type=pulse_info["pulse_type"][qb])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

            # with pi ef pulse for actual rd qb(f state)
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            qb = self.on_rds[0]
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            pulse_info = self.lattice_cfg["pulse_info"]

            self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                       amp=pulse_info[setup]["pi_amp"][qb],
                       pulse_type=pulse_info["pulse_type"][qb])
            self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_ef_len"][qb],
                       amp=pulse_info[setup]["pi_ef_amp"][qb],
                       pulse_type=pulse_info["ef_pulse_type"][qb], add_freq=self.lattice_cfg['qubit']['anharmonicity'][qb] )
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

            # with pi pulse crossqb (e state)
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            qb = self.expt_cfg["cross_qb"]
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            pulse_info = self.lattice_cfg["pulse_info"]
            self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                       amp=pulse_info[setup]["pi_amp"][qb],
                       pulse_type=pulse_info["pulse_type"][qb])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

            # with f pulse crossqb (f state)
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            qb = self.expt_cfg["cross_qb"]
            setup = self.lattice_cfg["qubit"]["setup"][qb]
            pulse_info = self.lattice_cfg["pulse_info"]
            self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                       amp=pulse_info[setup]["pi_amp"][qb],
                       pulse_type=pulse_info["pulse_type"][qb])
            self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_ef_len"][qb],
                       amp=pulse_info[setup]["pi_ef_amp"][qb],
                       pulse_type=pulse_info["ef_pulse_type"][qb],
                       add_freq=self.lattice_cfg['qubit']['anharmonicity'][qb])
            sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

            # with two pi pulses
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            for qb in [self.on_rds[0], self.expt_cfg["cross_qb"]]:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                pulse_info = self.lattice_cfg["pulse_info"]
                self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                           amp=pulse_info[setup]["pi_amp"][qb],
                           pulse_type=pulse_info["pulse_type"][qb])
                sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

            # with two pi pulses flip order
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            for qb in [self.expt_cfg["cross_qb"], self.on_rds[0]]:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                pulse_info = self.lattice_cfg["pulse_info"]
                self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                           amp=pulse_info[setup]["pi_amp"][qb],
                           pulse_type=pulse_info["pulse_type"][qb])
                sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

            # with pief, pi cross
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            for qb in [self.on_rds[0], self.expt_cfg["cross_qb"]]:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                pulse_info = self.lattice_cfg["pulse_info"]
                self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                           amp=pulse_info[setup]["pi_amp"][qb],
                           pulse_type=pulse_info["pulse_type"][qb])
                if qb==self.on_rds[0]:
                    self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_ef_len"][qb],
                               amp=pulse_info[setup]["pi_ef_amp"][qb],
                               pulse_type=pulse_info["ef_pulse_type"][qb],
                               add_freq=self.lattice_cfg['qubit']['anharmonicity'][qb])
                sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

            # with pi cross, pief
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            for qb in [self.expt_cfg["cross_qb"], self.on_rds[0]]:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                pulse_info = self.lattice_cfg["pulse_info"]
                self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                           amp=pulse_info[setup]["pi_amp"][qb],
                           pulse_type=pulse_info["pulse_type"][qb])
                if qb==self.on_rds[0]:
                    self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_ef_len"][qb],
                               amp=pulse_info[setup]["pi_ef_amp"][qb],
                               pulse_type=pulse_info["ef_pulse_type"][qb],
                               add_freq=self.lattice_cfg['qubit']['anharmonicity'][qb])
                sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

            # with pi, pief cross
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            for qb in [self.on_rds[0], self.expt_cfg["cross_qb"]]:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                pulse_info = self.lattice_cfg["pulse_info"]
                self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                           amp=pulse_info[setup]["pi_amp"][qb],
                           pulse_type=pulse_info["pulse_type"][qb])
                if qb==self.expt_cfg["cross_qb"]:
                    self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_ef_len"][qb],
                               amp=pulse_info[setup]["pi_ef_amp"][qb],
                               pulse_type=pulse_info["ef_pulse_type"][qb],
                               add_freq=self.lattice_cfg['qubit']['anharmonicity'][qb])
                sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

            # with pief cross, pi
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            for qb in [self.expt_cfg["cross_qb"], self.on_rds[0]]:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                pulse_info = self.lattice_cfg["pulse_info"]
                self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                           amp=pulse_info[setup]["pi_amp"][qb],
                           pulse_type=pulse_info["pulse_type"][qb])
                if qb==self.expt_cfg["cross_qb"]:
                    self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_ef_len"][qb],
                               amp=pulse_info[setup]["pi_ef_amp"][qb],
                               pulse_type=pulse_info["ef_pulse_type"][qb],
                               add_freq=self.lattice_cfg['qubit']['anharmonicity'][qb])
                sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

            # with two f pulses
            sequencer.new_sequence(self)
            self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
            for qb in [self.on_rds[0], self.expt_cfg["cross_qb"]]:
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                pulse_info = self.lattice_cfg["pulse_info"]
                self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                           amp=pulse_info[setup]["pi_amp"][qb],
                           pulse_type=pulse_info["pulse_type"][qb])
                self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_ef_len"][qb],
                           amp=pulse_info[setup]["pi_ef_amp"][qb],
                           pulse_type=pulse_info["ef_pulse_type"][qb],
                           add_freq=self.lattice_cfg['qubit']['anharmonicity'][qb])
                sequencer.sync_channels_time(self.channels)
            self.readout_pxi(sequencer, self.on_rds)
            sequencer.end_sequence()

        return sequencer.complete(self, plot=True)

    def histogram_crosstalk_combo_bir4(self, sequencer):
        # vacuum rabi sequences
        for ii in range(self.expt_cfg['num_seq_sets']):
            for bw in self.expt_cfg['bw_list']:
                #g
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
                self.readout_pxi(sequencer, self.on_rds)
                sequencer.end_sequence()

                # with pi pulse for actual rd qb(e state)
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
                qb = self.on_rds[0]
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                pulse_info = self.lattice_cfg["pulse_info"]
                sequencer.append('charge%s_I' % setup,
                                 BIR_4_hyp(max_amp=amp,
                                           bandwidth=bw,
                                           t_length=self.lattice_cfg['BIR4_hyp_pulse_len'],
                                           theta=self.lattice_cfg['BIR4_hyp_theta'],
                                           kappa=self.lattice_cfg['BIR4_hyp_kappa'],
                                           beta=self.lattice_cfg['BIR4_hyp_beta'],
                                           freq=iq_freq + add_freq,
                                           phase=phase))
                sequencer.append('charge%s_Q' % setup,
                                 BIR_4_hyp(max_amp=amp,
                                           bandwidth=bw,
                                           t_length=self.lattice_cfg['pulse_length'],
                                           theta=self.lattice_cfg['kappa'],
                                           kappa=self.lattice_cfg['kappa'],
                                           beta=self.lattice_cfg['beta'],
                                           freq=iq_freq + add_freq,
                                           phase=phase))
                sequencer.sync_channels_time(self.channels)
                self.readout_pxi(sequencer, self.on_rds)
                sequencer.end_sequence()

                # with pi pulse (e state)
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
                qb = self.expt_cfg["cross_qb"]
                setup = self.lattice_cfg["qubit"]["setup"][qb]
                pulse_info = self.lattice_cfg["pulse_info"]
                self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                           amp=pulse_info[setup]["pi_amp"][qb],
                           pulse_type=pulse_info["pulse_type"][qb])
                sequencer.sync_channels_time(self.channels)
                self.readout_pxi(sequencer, self.on_rds)
                sequencer.end_sequence()

                # with two pi pulses
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
                for qb in [self.on_rds[0], self.expt_cfg["cross_qb"]]:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    pulse_info = self.lattice_cfg["pulse_info"]
                    self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                               amp=pulse_info[setup]["pi_amp"][qb],
                               pulse_type=pulse_info["pulse_type"][qb])
                    sequencer.sync_channels_time(self.channels)
                self.readout_pxi(sequencer,self.on_rds)
                sequencer.end_sequence()

                # with two pi pulses flip order
                sequencer.new_sequence(self)
                self.pad_start_pxi(sequencer, on_qubits=["A", "B"], time=500)
                for qb in [self.expt_cfg["cross_qb"], self.on_rds[0]]:
                    setup = self.lattice_cfg["qubit"]["setup"][qb]
                    pulse_info = self.lattice_cfg["pulse_info"]
                    self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len"][qb],
                               amp=pulse_info[setup]["pi_amp"][qb],
                               pulse_type=pulse_info["pulse_type"][qb])
                    sequencer.sync_channels_time(self.channels)
                self.readout_pxi(sequencer,self.on_rds)
                sequencer.end_sequence()

        return sequencer.complete(self, plot=True)
    def get_experiment_sequences(self, experiment, **kwargs):
        vis = visdom.Visdom()
        vis.close()

        sequencer = Sequencer(self.channels, self.channels_awg, self.awg_info, self.channels_delay)
        self.expt_cfg = self.experiment_cfg[experiment]
        self.expt_params = self.expt_cfg
        self.on_qbs = self.expt_cfg["on_qbs"]
        self.qb_setups = [self.lattice_cfg["qubit"]["setup"][qb] for qb in self.on_qbs]
        if not ("on_rds" in self.expt_params.keys()) or self.expt_params["on_rds"] == [] or self.expt_params["on_rds"] == "auto":
            self.on_rds = self.on_qbs
        else:
            self.on_rds = self.expt_cfg["on_rds"]
        self.rd_setups = [self.lattice_cfg["readout"]["setup"][qb] for qb in self.on_rds]

        multiple_sequences = eval('self.' + experiment)(sequencer, **kwargs)

        return self.get_sequences(multiple_sequences)

    def get_sequences(self, multiple_sequences):
        seq_num = len(multiple_sequences)

        sequences = {}
        for channel in self.channels:
            channel_waveform = []
            for seq_id in range(seq_num):
                if "ff" in channel and (str(7) not in channel) and self.lattice_cfg["ff_info"]["ff_convolve"]:
                    for qb in range(7):
                        if str(qb) in channel:
                            kernel = np.load(self.lattice_cfg["ff_info"]["ff_kernel_filepath"] + self.lattice_cfg["ff_info"]["ff_kernel_filename"][qb])
                            channel_waveform.append(
                                np.concatenate(((np.convolve(multiple_sequences[seq_id][channel], kernel, 'same')
                                 ), [0])))
                else:
                    channel_waveform.append(multiple_sequences[seq_id][channel])

            sequences[channel] = np.array(channel_waveform)

        return sequences

    def get_awg_readout_time(self, readout_time_list):
        awg_readout_time_list = {}
        for awg in self.awg_info:
            awg_readout_time_list[awg] = (readout_time_list - self.awg_info[awg]['time_delay'])

        return awg_readout_time_list


if __name__ == "__main__":
    cfg = {
        "qubit": {"freq": 4.5}
    }

    hardware_cfg = {
        "channels": [
            "charge1", "flux1", "charge2", "flux2",
            "hetero1_I", "hetero1_Q", "hetero2_I", "hetero2_Q",
            "m8195a_trig", "readout1_trig", "readout2_trig", "alazar_trig"
        ],
        "channels_awg": {"charge1": "m8195a", "flux1": "m8195a", "charge2": "m8195a", "flux2": "m8195a",
                         "hetero1_I": "tek5014a", "hetero1_Q": "tek5014a", "hetero2_I": "tek5014a",
                         "hetero2_Q": "tek5014a",
                         "m8195a_trig": "tek5014a", "readout1_trig": "tek5014a", "readout2_trig": "tek5014a",
                         "alazar_trig": "tek5014a"},
        "awg_info": {"m8195a": {"dt": 0.0625, "min_increment": 16, "min_samples": 128, "time_delay": 110},
                     "tek5014a": {"dt": 0.83333333333, "min_increment": 16, "min_samples": 128, "time_delay": 0}}
    }

    ps = PulseSequences(cfg, hardware_cfg)

    ps.get_experiment_sequences('rabi')
    
    
    
# OLD EXPERIMENTS
#     def fast_flux_flux_fact_finding_mission(self, sequencer):
# 
#         for dfreq in np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step']):
#             sequencer.new_sequence(self)
# 
#             #add flux to pulse
#             for target in self.expt_cfg['target_qb']:
#                 #since we haven't been synching the flux channels, this should still be back at the beginning
#                 sequencer.append('ff_Q%s' % target,
#                                  Square(max_amp=self.expt_cfg['ff_amp'], flat_len=self.expt_cfg['ff_time'],
#                                         ramp_sigma_len=self.expt_cfg['ff_ramp_sigma_len'], cutoff_sigma=2, freq=0,
#                                         phase=0))
# 
#             #COMPENSATION PULSE
#             for target in self.expt_cfg['target_qb']:
#                 sequencer.append('ff_Q%s' % target,
#                                  Square(max_amp=-self.expt_cfg['ff_amp'],
#                                         flat_len=self.expt_cfg['ff_time'],
#                                         ramp_sigma_len=self.expt_cfg['ff_ramp_sigma_len'], cutoff_sigma=2, freq=0,
#                                         phase=0))
# 
#             sequencer.sync_channels_time(self.channels)
#             for qb in self.on_qbs:
#                 setup = self.lattice_cfg["qubit"]["setup"][qb]
#                 sequencer.append('charge%s_I' % setup,
#                                  Square(max_amp=self.expt_cfg['qb_amp'], flat_len=self.expt_cfg['qb_pulse_length'],
#                                         ramp_sigma_len=0.001, cutoff_sigma=2,
#                                         freq=self.pulse_info[setup]['iq_freq'] + dfreq,
#                                         phase=0))
# 
#                 sequencer.append('charge%s_Q' % setup,
#                                  Square(max_amp=self.expt_cfg['qb_amp'], flat_len=self.expt_cfg['qb_pulse_length'],
#                                         ramp_sigma_len=0.001, cutoff_sigma=2, freq= self.pulse_info[setup]['iq_freq'] + dfreq,
#                                         phase=self.pulse_info[setup]['Q_phase']))
#                 self.idle_q(sequencer, time=self.expt_cfg['delay'])
# 
#             #add readout and readout trig (ie dig) pulse
#             self.readout_pxi(sequencer, self.on_rds,overlap=False)
# 
#             sequencer.end_sequence()
# 
#         return sequencer.complete(self, plot=True)

    # 
    # def melting_multi_readout_full_ramp_Q3(self, sequencer):
    #     # Modify to always refer to iq_diff relative to 185MHz above Q3
    #     qb_list = self.expt_cfg["Mott_qbs"]
    #     lo_qb = 3
    # 
    #     setup = self.on_qubits[0]
    #     for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
    #         sequencer.new_sequence(self)
    #         self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
    # 
    #         ##################################GENERATE PI PULSES ################################################
    #         for i, qb in enumerate(qb_list):
    #             pulse_info = self.lattice_cfg["pulse_info"]
    #             qb_iq_freq_dif = self.lattice_cfg["qubit"]["freq"][qb] - self.lattice_cfg["qubit"]["freq"][3] # modify to always refer to Q3
    #             # qb_iq_freq_dif = 0
    #             self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len_melt"][qb],
    #                        amp=pulse_info[setup]["pi_amp"][qb], add_freq=qb_iq_freq_dif, phase=0,
    #                        pulse_type=pulse_info["pulse_type"][qb])
    #             self.idle_q(sequencer, time=20)
    # 
    #         sequencer.sync_channels_time(self.channels)
    #         self.idle_q(sequencer, time=100)
    #         ##############################GENERATE RAMP###########################################
    #         flux_vec = self.expt_cfg["ff_vec"]
    #         self.ff_pulse(sequencer, ff_len = [evolution_t]*8, pulse_type = self.expt_cfg["ramp_type"], flux_vec= flux_vec, flip_amp=False)
    #         ############################## readout ###########################################
    #         sequencer.sync_channels_time(self.channels)
    #         self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
    #         sequencer.sync_channels_time(self.channels)
    #         self.readout_pxi(sequencer, setup, overlap=False)
    #         sequencer.sync_channels_time(self.channels)
    # 
    #         ############################## generate compensation ###########################################
    #         flux_vec = self.expt_cfg["ff_vec"]
    #         self.ff_pulse(sequencer, [evolution_t]*8, pulse_type=self.expt_cfg["ramp_type"], flux_vec=flux_vec, flip_amp=True)
    #         sequencer.end_sequence()
    # 
    #     ############### Append Pi-Calibration pulses to Melt Expt Sequence.  Need Q1,Q3,Q5 IQ Freq Shapes ##########
    #     ## Iterate over different IQ values, each qubit driven
    #     rd_qb_list = self.expt_cfg["rd_qb"]
    #     for ii,rd_qb in enumerate(rd_qb_list):
    # 
    #         sequencer.new_sequence(self)
    #         self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
    #         sequencer.sync_channels_time(self.channels)
    #         self.readout_pxi(sequencer, setup, overlap=False)
    #         sequencer.sync_channels_time(self.channels)
    #         sequencer.end_sequence()
    # 
    #         sequencer.new_sequence(self)
    #         self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
    #         pulse_info = self.lattice_cfg["pulse_info"]
    #         qb_iq_freq_dif = self.lattice_cfg["qubit"]["freq"][rd_qb] - self.lattice_cfg["qubit"]["freq"][3]  # modify to always refer to Q3
    #         sequencer.sync_channels_time(self.channels)
    #         self.gen_q(sequencer, qb=qb, len=pulse_info[setup]["pi_len_melt"][rd_qb],
    #                    amp=pulse_info[setup]["pi_amp"][rd_qb], add_freq=qb_iq_freq_dif, phase=0,
    #                    pulse_type=pulse_info["pulse_type"][rd_qb])
    #         sequencer.sync_channels_time(self.channels)
    #         self.idle_q(sequencer, time=20)
    #         sequencer.sync_channels_time(self.channels)
    #         self.readout_pxi(sequencer, setup, overlap=False)
    #         sequencer.sync_channels_time(self.channels)
    #         sequencer.end_sequence()
    # 
    #     return sequencer.complete(self, plot=True)

    # def melting_single_readout_full_ramp_2setups_temp(self, sequencer):
    #     qb_list_1 = self.expt_cfg["Mott_qbs_1"]
    #     qb_list_2 = self.expt_cfg["Mott_qbs_2"]
    #     idle_post_pulse = 100
    #     wait_post_flux = 500
    #
    #     for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
    #         sequencer.new_sequence(self)
    #         self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
    #
    #         ##################################GENERATE PI PULSES - Pi Q1, 3, 5, wait for flux, pi Q4################################################
    #         #pi 1, 3, 5
    #         for i, qb in enumerate(qb_list_1):
    #             setup = self.lattice_cfg["qubit"]["setup"][qb]
    #             self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
    #                        amp=self.pulse_info[setup]["pi_amp"][qb],
    #                        pulse_type=self.pulse_info["pulse_type"][qb])
    #             self.idle_q(sequencer, time=20)
    #             sequencer.sync_channels_time(self.channels)
    #
    #         self.idle_q(sequencer, time=idle_post_pulse)
    #         sequencer.sync_channels_time(self.channels)
    #         # going to move Q4 to lattice
    #
    #
    #         #Qbs at lattice
    #         self.idle_q(sequencer, time=wait_post_flux)
    #         channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
    #         sequencer.sync_channels_time(channels_excluding_fluxch)
    #         for i, qb in enumerate(qb_list_2):
    #             setup = self.lattice_cfg["qubit"]["setup"][qb]
    #             self.gen_q(sequencer, qb=qb, len=self.expt_cfg["ff_pi_len"][qb],
    #                        amp=self.expt_cfg["ff_pi_amp"][qb],
    #                        pulse_type=self.pulse_info["pulse_type"][qb])
    #             self.idle_q(sequencer, time=20)
    #
    #         ##############################GENERATE RAMP - Q4 ramp to lattice and 3/5 positioned to not oscillate, idle for piQ4, then bring 1/3/5 to lat###########################################
    #
    #         pulsetime1 = 0
    #         pulsetime1 += wait_post_flux
    #         for i,qb in enumerate(qb_list_2):
    #             pulsetime1 += 4 * self.expt_cfg["ff_pi_len"][qb] + 20
    #             # pulsetime1 += 20 + self.lattice_cfg['pulse_info']['wurst_pulselen'][qb] # wurst pulse size
    #
    #         pulsetime1 += idle_post_pulse
    #         pulsetime2 = idle_post_pulse
    #         pulsetime3 = evolution_t
    #
    #         delay_list = [0, pulsetime1, pulsetime1+pulsetime2]
    #         ff_len_list = [[pulsetime1 + pulsetime2+pulsetime3 + 2*self.lattice_cfg['ff_info']['ff_exp_ramp_len'][0]]*8,
    #                        [pulsetime2+pulsetime3 + 2*self.lattice_cfg['ff_info']['ff_exp_ramp_len'][0]]*8,
    #                        [evolution_t]*8]
    #
    #         ff_vec_list = self.expt_cfg["ff_vec_list"]
    #         self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
    #                                flux_vec_list = ff_vec_list, delay_list = delay_list, freq_list=[0]*7, flip_amp=False)
    #
    #         sequencer.sync_channels_time(self.channels)
    #         ############################## readout ###########################################
    #         sequencer.sync_channels_time(self.channels)
    #         self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
    #         sequencer.sync_channels_time(self.channels)
    #         self.readout_pxi(sequencer, self.on_rds, overlap=False)
    #         sequencer.sync_channels_time(self.channels)
    #
    #         ############################## generate compensation ###########################################
    #
    #         self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
    #                                flux_vec_list=ff_vec_list, delay_list=delay_list, freq_list=[0]*7, flip_amp=True)
    #         sequencer.end_sequence()
    #
    #     return sequencer.complete(self, plot=True)



    # def melting_reversability_sweep_ramplen(self, sequencer):
    #
    #     qb_list_1 = self.expt_cfg["Mott_qbs_1"]
    #     qb_list_2 = self.expt_cfg["Mott_qbs_2"]
    #     idle_post_pulse = 100
    #     wait_post_flux = 500
    #     for expramplen in self.expt_cfg['ramplenlist']:
    #         self.lattice_cfg['ff_info']['ff_exp_ramp_len'] = list(np.ones(7) * expramplen)
    #         self.lattice_cfg['ff_info']['ff_exp_ramp_tau'] = list(np.ones(7) * expramplen * (0.5 / 5))
    #         for evolution_t in np.arange(self.expt_cfg["evolution_t_start"], self.expt_cfg["evolution_t_stop"], self.expt_cfg["evolution_t_step"]):
    #             sequencer.new_sequence(self)
    #             self.pad_start_pxi(sequencer, on_qubits=["A","B"], time=500)
    #
    #             ##################################GENERATE PI PULSES - Pi Q1, 3, 5, wait for flux, pi Q4################################################
    #             #pi 1, 3, 5
    #             for i, qb in enumerate(qb_list_1):
    #                 setup = self.lattice_cfg["qubit"]["setup"][qb]
    #                 self.gen_q(sequencer, qb=qb, len=self.pulse_info[setup]["pi_len"][qb],
    #                            amp=self.pulse_info[setup]["pi_amp"][qb],
    #                            pulse_type=self.pulse_info["pulse_type"][qb])
    #                 self.idle_q(sequencer, time=20)
    #                 sequencer.sync_channels_time(self.channels)
    #
    #             self.idle_q(sequencer, time=idle_post_pulse)
    #             sequencer.sync_channels_time(self.channels)
    #             # going to move Q4 to lattice
    #
    #
    #             #Qbs at lattice
    #             self.idle_q(sequencer, time=wait_post_flux)
    #             channels_excluding_fluxch = [ch for ch in self.channels if 'ff' not in ch]
    #             sequencer.sync_channels_time(channels_excluding_fluxch)
    #             for i, qb in enumerate(qb_list_2):
    #                 setup = self.lattice_cfg["qubit"]["setup"][qb]
    #                 self.gen_q(sequencer, qb=qb, len=self.expt_cfg["ff_pi_len"][qb],
    #                            amp=self.expt_cfg["ff_pi_amp"][qb],
    #                            pulse_type=self.pulse_info["pulse_type"][qb])
    #                 self.idle_q(sequencer, time=20)
    #
    #             ##############################GENERATE RAMP - Q4 ramp to lattice and 3/5 positioned to not oscillate, idle for piQ4, then bring 1/3/5 to lat###########################################
    #
    #             pulsetime1 = 0
    #             pulsetime1 += wait_post_flux
    #             for i,qb in enumerate(qb_list_2):
    #                 pulsetime1 += 4 * self.expt_cfg["ff_pi_len"][qb] + 20
    #                 # pulsetime1 += 20 + self.lattice_cfg['pulse_info']['wurst_pulselen'][qb] # wurst pulse size
    #
    #             pulsetime1 += idle_post_pulse
    #             pulsetime2 = idle_post_pulse
    #             pulsetime3 = evolution_t
    #
    #             delay_list = [0, pulsetime1, pulsetime1+pulsetime2]
    #             ff_len_list = [[pulsetime1 + pulsetime2+pulsetime3 + 2*(expramplen)]*8,
    #                            [pulsetime2+pulsetime3 + 2*(expramplen)]*8,
    #                            [evolution_t]*8]
    #
    #             ff_vec_list = self.expt_cfg["ff_vec_list"]
    #             self.ff_pulse_combined(sequencer, ff_len_list = ff_len_list, pulse_type_list = self.expt_cfg["pulse_type_list"],
    #                                    flux_vec_list = ff_vec_list, delay_list = delay_list, freq_list=[0]*7, flip_amp=False)
    #
    #             sequencer.sync_channels_time(self.channels)
    #             ############################## readout ###########################################
    #             sequencer.sync_channels_time(self.channels)
    #             self.idle_q(sequencer, time=self.expt_cfg['wait_post_flux'])
    #             sequencer.sync_channels_time(self.channels)
    #             self.readout_pxi(sequencer, self.on_rds, overlap=False)
    #             sequencer.sync_channels_time(self.channels)
    #
    #             ############################## generate compensation ###########################################
    #
    #             self.ff_pulse_combined(sequencer, ff_len_list=ff_len_list, pulse_type_list=self.expt_cfg["pulse_type_list"],
    #                                    flux_vec_list=ff_vec_list, delay_list=delay_list, freq_list=[0]*7, flip_amp=True)
    #             sequencer.end_sequence()
    #         if self.expt_cfg["pi_cal_seq"] and self.expt_cfg["pi_calibration"]:
    #             for rd_qb in self.on_rds:
    #                 self.pi_cal_sequence(sequencer, pi_qbs=[rd_qb], pi_rds=self.on_rds)
    #
    #     return sequencer.complete(self, plot=True)
