__author__ = 'Nelson'

from slab.instruments.awg.PulseSequence import *
from numpy import arange, linspace, array
from slab.experiments.ExpLib.PulseSequenceBuilder import *

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
        total_pulse_span_length_list = []
        self.total_flux_pulse_span_length_list = []

        for ii, pt in enumerate(self.expt_pts):
            # obtain pulse sequence for each experiment point
            define_pulses(pt)

            ## add heterodyne pulse
            self.add_heterodyne_pulses()

            self.pulse_sequence_matrix.append(self.psb.get_pulse_sequence())
            total_pulse_span_length_list.append(self.psb.get_total_pulse_span_length())
            self.total_flux_pulse_span_length_list.append(self.psb.get_total_flux_pulse_span_length())

        if len(calibration_pts) > 0:

            for jj, pt in enumerate(calibration_pts):
                if jj ==0:
                    self.psb.idle(10)
                if jj ==1:
                    self.psb.append('q','cal_pi', self.pulse_type)
                if jj ==2:
                    self.psb.append('q','cal_pi', self.pulse_type)
                    self.psb.append('q', 'pi_q_ef', self.pulse_type)

                ## add heterodyne pulse
                self.add_heterodyne_pulses()

                self.pulse_sequence_matrix.append(self.psb.get_pulse_sequence())
                total_pulse_span_length_list.append(self.psb.get_total_pulse_span_length())
                self.total_flux_pulse_span_length_list.append(self.psb.get_total_flux_pulse_span_length())

        max_length = self.psb.get_max_length(total_pulse_span_length_list)
        max_flux_length = self.psb.get_max_flux_length(self.total_flux_pulse_span_length_list)
        self.set_all_lengths(max_length)
        self.set_waveform_length("qubit 1 flux", max_flux_length)

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

        for ii in range(len(het_IFreqList)):

            # q2 pulses are hacked to be fixed in time, so can append multiple pulses for heterodyne readout
            self.psb.append('q2', 'general', 'square', amp= het_a_list[ii],
                            length=self.cfg['readout']['width'] + 1000,
                            freq= het_IFreqList[ii],
                            delay=(self.cfg['readout']['width'] + 1000) / 2)

    def build_sequence(self):
        PulseSequence.build_sequence(self)
        wtpts = self.get_waveform_times('qubit drive I')
        mtpts = self.get_marker_times('qubit buffer')
        ftpts = self.get_waveform_times('qubit 1 flux')
        markers_readout = self.markers['readout pulse']
        markers_card = self.markers['card trigger']
        waveforms_qubit_I = self.waveforms['qubit drive I']
        waveforms_qubit_Q = self.waveforms['qubit drive Q']
        waveforms_qubit_flux = self.waveforms['qubit 1 flux']
        markers_qubit_buffer = self.markers['qubit buffer']
        markers_ch3m1 = self.markers['ch3m1']


        # for second PXDAC4800
        waveforms_pxdac4800_2_ch1 = self.waveforms['pxdac4800_2_ch1']
        waveforms_pxdac4800_2_ch2 = self.waveforms['pxdac4800_2_ch2']


        self.psb.prepare_build(wtpts, mtpts, ftpts, markers_readout, markers_card, waveforms_qubit_I, waveforms_qubit_Q, waveforms_qubit_flux,
                              markers_qubit_buffer, markers_ch3m1,waveforms_pxdac4800_2_ch1,waveforms_pxdac4800_2_ch2)
        generated_sequences = self.psb.build(self.pulse_sequence_matrix,self.total_flux_pulse_span_length_list)

        self.markers['readout pulse'], self.markers['card trigger'], self.waveforms['qubit drive I'], self.waveforms[
            'qubit drive Q'], self.waveforms['qubit 1 flux'], self.markers['qubit buffer'], self.markers['ch3m1'],self.waveforms['pxdac4800_2_ch1'], self.waveforms['pxdac4800_2_ch2']= generated_sequences

        # np.save('S:\\_Data\\160711 - Nb Tunable Coupler\\data\\waveform.npy',self.waveforms['qubit drive Q'])
        ### in ipython notebook: call np.load('file_path/file_name.npy')

    def reshape_data(self, data):
        return np.reshape(data, (self.sequence_length, self.waveform_length))