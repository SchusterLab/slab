__author__ = 'Nelson'

from slab.instruments.awg.PulseSequence import *
from numpy import arange, linspace, array
from slab.experiments.ExpLib.PulseSequenceBuilder import *
import time
from liveplot import LivePlotClient

class QubitPulseSequence(PulseSequence):
    '''
    Parent class for all the single qubit pulse sequences.
    '''
    def __init__(self, name, cfg, expt_cfg, define_points, define_parameters, define_pulses, **kwargs):

        self.expt_cfg = expt_cfg
        self.cfg = cfg
        define_points()
        define_parameters()
        sequence_length = len(self.expt_pts)

        # if "multimode" not in name.lower():
        #     cfg['awgs'][1]['upload'] = False
        # else:
        #     cfg['awgs'][1]['upload'] = True
        calibration_pts = []

        if (expt_cfg['use_g-e-f_calibration']):
            calibration_pts = range(3)
            sequence_length+=3
        elif (expt_cfg['use_pi_calibration']):
            calibration_pts = range(2)
            sequence_length+=2

        PulseSequence.__init__(self, name, cfg['awgs'], sequence_length)

        self.psb = PulseSequenceBuilder(cfg)
        self.pulse_sequence_matrix = []
        self.total_pulse_span_length_list = []
        self.total_flux_pulse_span_length_list = []

        self.flux_pulse_span_list = []

        for ii, pt in enumerate(self.expt_pts):
            # obtain pulse sequence for each experiment point
            define_pulses(pt)

            self.pulse_sequence_matrix.append(self.psb.get_pulse_sequence())
            self.total_pulse_span_length_list.append(self.psb.get_total_pulse_span_length())
            self.total_flux_pulse_span_length_list.append(self.psb.get_total_flux_pulse_span_length())

        if len(calibration_pts) > 0:

            for jj, pt in enumerate(calibration_pts):

                if self.name == 'rabi_thermalizer':
                    define_pulses(jj+len(self.expt_pts))

                else:
                    if jj ==0:
                        self.psb.idle(10)
                    if jj ==1:
                        self.psb.append('q','cal_pi', self.pulse_type)
                    if jj ==2:
                        self.psb.append('q','cal_pi', self.pulse_type)
                        self.psb.append('q', 'pi_q_ef', self.pulse_type)

                self.pulse_sequence_matrix.append(self.psb.get_pulse_sequence())
                self.total_pulse_span_length_list.append(self.psb.get_total_pulse_span_length())
                self.total_flux_pulse_span_length_list.append(self.psb.get_total_flux_pulse_span_length())

        max_length = self.psb.get_max_length(self.total_pulse_span_length_list)
        max_flux_length = self.psb.get_max_flux_length(self.total_flux_pulse_span_length_list)
        self.set_all_lengths(max_length)

        ###
        # heterodyne pulse - hack: max_length = 0
        temp_seq_matrix = self.pulse_sequence_matrix[:]
        self.pulse_sequence_matrix = []
        self.add_heterodyne_pulses()
        #self.add_flux_pulses(length=500)
        temp_seqs = self.psb.get_pulse_sequence()

        # clears ?
        dummy = self.psb.get_total_pulse_span_length()
        dummy = self.psb.get_total_flux_pulse_span_length()

        for ii in range(len(self.expt_pts)+len(calibration_pts)):
            self.pulse_sequence_matrix.append(temp_seq_matrix[ii] + temp_seqs) # concatenate hetero/flux pulse
        ###

        ###

        # flux pulse
        temp_seq_matrix = self.pulse_sequence_matrix[:]
        self.pulse_sequence_matrix = []
        for ii in range(len(self.expt_pts) + len(calibration_pts)):

            if self.name == 'rabi_thermalizer':
                define_pulses(ii, isFlux = True)
            else:
                self.add_flux_pulses(pulse_span_length = self.total_pulse_span_length_list[ii])
            temp_seqs = self.psb.get_pulse_sequence() # also clears seq
            self.flux_pulse_span_list.append(self.psb.get_total_pulse_span_length())
            dummy = self.psb.get_total_flux_pulse_span_length()

            self.pulse_sequence_matrix.append(temp_seq_matrix[ii] + temp_seqs)  # concatenate hetero/flux pulse

        max_flux_length = round_samples( max(self.flux_pulse_span_list)+ 2 * self.psb.start_end_buffer)
        self.set_all_lengths(max(max_flux_length,max_length))

        ###

    def add_flux_pulses(self, pulse_span_length):

        # this is to align flux pulse to readout? (diff in 2 pxdac cards)
        hw_delay = self.cfg['flux_pulse_info']['pxdac_hw_delay']

        if self.cfg['flux_pulse_info']['on_during_drive'] and self.cfg['flux_pulse_info']['on_during_readout']:

            flux_width = max(pulse_span_length + self.cfg['readout']['delay'] + self.cfg['readout']['width'] \
                             - self.cfg['flux_pulse_info']['flux_drive_delay'], 0)
            flux_comp_width = flux_width  # self.cfg['flux_pulse_info']['dc_comp_pulse_length']
            flux_delay = self.cfg['flux_pulse_info']['flux_drive_delay'] + hw_delay
            flux_idle = 100.0

        elif (self.cfg['flux_pulse_info']['on_during_drive']) and (
                not self.cfg['flux_pulse_info']['on_during_readout']):

            flux_width = max(pulse_span_length - self.cfg['flux_pulse_info']['flux_drive_delay'], 0)
            flux_comp_width = flux_width  # self.cfg['flux_pulse_info']['dc_comp_pulse_length']
            flux_delay = self.cfg['flux_pulse_info']['flux_drive_delay'] + hw_delay
            flux_idle = self.cfg['readout']['delay'] + self.cfg['readout']['width'] + 100

        elif (not self.cfg['flux_pulse_info']['on_during_drive']) and (
        self.cfg['flux_pulse_info']['on_during_readout']):

            flux_width = self.cfg['readout']['delay'] + self.cfg['readout']['width']
            flux_comp_width = flux_width  # self.cfg['flux_pulse_info']['dc_comp_pulse_length']
            flux_delay = hw_delay + pulse_span_length
            flux_idle = 100.0

        else:
            flux_width = 0
            flux_comp_width = 0
            flux_delay = 0
            flux_idle = 0

        flux_a = self.cfg['flux_pulse_info']['flux_a']
        flux_freq = self.cfg['flux_pulse_info']['flux_freq']

        for ii in range(4):

            flux_comp_a = - flux_a[ii]  # flux_area/float(flux_comp_width)

            if flux_width > 0:
                self.psb.append('flux_'+str(ii+1), 'general', 'square', amp=flux_a[ii],
                            length = flux_width, freq = flux_freq[ii],
                            delay = flux_delay)
            if flux_comp_width > 0:
                self.psb.append('flux_' + str(ii + 1), 'general', 'square', amp=flux_comp_a,
                            length = flux_comp_width, freq = flux_freq[ii],
                            delay = flux_delay + flux_idle)


    def add_heterodyne_pulses(self):

        if self.cfg['readout']['is_multitone_heterodyne']:
            het_carrier_freq = self.cfg['readout']['heterodyne_carrier_freq']
            het_read_freq_list = array(self.cfg['readout']['heterodyne_freq_list'])
            het_a_list = array(self.cfg['readout']['heterodyne_a_list'])
            het_IFreqList = het_read_freq_list - het_carrier_freq
        else:
            het_carrier_freq = self.cfg['readout']['frequency'] - self.cfg['readout']['heterodyne_freq']
            het_read_freq_list = array([self.cfg['readout']['frequency']])
            het_a_list = array([self.cfg['readout']['heterodyne_a']])
            het_IFreqList = het_read_freq_list - het_carrier_freq

        if sum(het_a_list) > 1:
            print 'Warning! Sum of heterodyne amplitudes > 1 in QubitPulseSequence.'

        if not self.cfg['readout']['is_fast_awg']:

            for ii in range(len(het_IFreqList)):
                # q2 pulses are hacked to be fixed in time, so can append multiple pulses for heterodyne readout
                self.psb.append('hetero', 'general', 'square', amp= het_a_list[ii],
                                length=self.psb.measurement_width,
                                freq= het_IFreqList[ii],
                                delay=self.psb.measurement_width/2.0 + 100) # the 100ns is a mysterious delay
        else:

            # heterodyne carrier - LO
            self.psb.append('hetero_carrier', 'general', 'square', amp=self.cfg['readout']['hetero_carrier_a'],
                            length=self.psb.measurement_width,
                            freq=het_carrier_freq,
                            delay=self.psb.measurement_width / 2.0 + 100)  # the 100ns is a mysterious delay
            if self.cfg['readout']['is_hetero_phase_ref']:
                # hetero phase reference to solve alazar jitter
                self.psb.append('hetero_carrier', 'general', 'square', amp=self.cfg['readout']['hetero_phase_ref_a'],
                                length=self.psb.measurement_width,
                                freq=self.cfg['readout']['hetero_phase_ref_freq'],
                                delay=self.psb.measurement_width / 2.0 + 100)  # the 100ns is a mysterious delay

            for ii in range(len(het_IFreqList)):
                # q2 pulses are hacked to be fixed in time, so can append multiple pulses for heterodyne readout
                self.psb.append('hetero', 'general', 'square', amp= het_a_list[ii],
                                length=self.psb.measurement_width,
                                freq=het_read_freq_list[ii],
                                delay=self.psb.measurement_width / 2.0 + 100)  # the 100ns is a mysterious delay

    def build_sequence(self):
        PulseSequence.build_sequence(self)
        # wtpts = self.get_waveform_times('qubit drive I')
        # # mtpts = self.get_marker_times('qubit buffer')
        # # ftpts = self.get_waveform_times('qubit 1 flux')
        # # markers_readout = self.markers['readout pulse']
        # # markers_card = self.markers['card trigger']
        # waveforms_qubit_I = self.waveforms['qubit drive I']
        # waveforms_qubit_Q = self.waveforms['qubit drive Q']
        # # waveforms_qubit_flux = self.waveforms['qubit 1 flux']
        # # markers_qubit_buffer = self.markers['qubit buffer']
        # # markers_ch3m1 = self.markers['ch3m1']
        # # for second PXDAC4800
        # waveforms_pxdac4800_2_ch1 = self.waveforms['pxdac4800_2_ch1']
        # waveforms_pxdac4800_2_ch2 = self.waveforms['pxdac4800_2_ch2']

        self.waveforms_dict = {}
        self.waveforms_tpts_dict = {}

        for awg in self.awg_info:
            for waveform in awg['waveforms']:
                self.waveforms_dict[waveform['name']] = self.waveforms[waveform['name']]
                self.waveforms_tpts_dict[waveform['name']] = self.get_waveform_times(waveform['name'])

        start_time = time.time()
        print '\nStart building sequences...'

        self.psb.prepare_build(self.waveforms_tpts_dict,self.waveforms_dict)

        # self.psb.prepare_build(wtpts, mtpts, ftpts, markers_readout, markers_card, waveforms_qubit_I, waveforms_qubit_Q, waveforms_qubit_flux,
        #                       markers_qubit_buffer, markers_ch3m1,waveforms_pxdac4800_2_ch1,waveforms_pxdac4800_2_ch2)
        generated_sequences = self.psb.build(self.pulse_sequence_matrix,self.total_flux_pulse_span_length_list)

        self.waveforms_dict = generated_sequences

        for waveform_key in self.waveforms_dict:
            self.waveforms[waveform_key] = self.waveforms_dict[waveform_key]

        end_time = time.time()
        print 'Finished building sequences in', end_time - start_time, 'seconds.\n'

        # np.save('S:\\_Data\\160711 - Nb Tunable Coupler\\data\\waveform.npy',self.waveforms['qubit drive Q'])
        ### in ipython notebook: call np.load('file_path/file_name.npy')


        # change amplitude
        # self.waveforms['pxdac4800_2_ch2'] = 0.5 * self.waveforms['pxdac4800_2_ch2']

    def reshape_data(self, data):
        return np.reshape(data, (self.sequence_length, self.waveform_length))