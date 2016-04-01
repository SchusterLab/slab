__author__ = 'Nelson'

from slab.instruments.awg.PulseSequence import *
from numpy import arange, linspace
from slab.experiments.ExpLib.PulseSequenceBuilder import *

from liveplot import LivePlotClient

class QubitPulseSequence(PulseSequence):
    '''
    Parent class for all the single qubit pulse sequences.
    '''
    def __init__(self, name, cfg, expt_cfg, define_points, define_parameters, define_pulses, **kwargs):

        self.expt_cfg = expt_cfg
        define_points()
        define_parameters()
        sequence_length = len(self.expt_pts)

        if "multimode" not in name.lower():
            cfg['awgs'][1]['upload'] = False
        else:
            cfg['awgs'][1]['upload'] = True

        if (expt_cfg['use_pi_calibration']):
            sequence_length+=2

        PulseSequence.__init__(self, name, cfg['awgs'], sequence_length)

        self.psb = PulseSequenceBuilder(cfg)
        self.pulse_sequence_matrix = []
        total_pulse_span_length_list = []
        self.total_flux_pulse_span_length_list = []

        for ii, pt in enumerate(self.expt_pts):
            # obtain pulse sequence for each experiment point
            define_pulses(pt)
            self.pulse_sequence_matrix.append(self.psb.get_pulse_sequence())
            total_pulse_span_length_list.append(self.psb.get_total_pulse_span_length())
            self.total_flux_pulse_span_length_list.append(self.psb.get_total_flux_pulse_span_length())

        if (expt_cfg['use_pi_calibration']):
            calibration_pts = [0,1]

            for jj, pt in enumerate(calibration_pts):
                if jj ==0:
                    self.psb.idle(10)
                if jj ==1:
                    self.psb.append('q','cal_pi', self.pulse_type)
                self.pulse_sequence_matrix.append(self.psb.get_pulse_sequence())
                total_pulse_span_length_list.append(self.psb.get_total_pulse_span_length())
                self.total_flux_pulse_span_length_list.append(self.psb.get_total_flux_pulse_span_length())

        max_length = self.psb.get_max_length(total_pulse_span_length_list)
        max_flux_length = self.psb.get_max_flux_length(self.total_flux_pulse_span_length_list)
        self.set_all_lengths(max_length)
        self.set_waveform_length("qubit 1 flux", max_flux_length)

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
        self.psb.prepare_build(wtpts, mtpts, ftpts, markers_readout, markers_card, waveforms_qubit_I, waveforms_qubit_Q, waveforms_qubit_flux,
                              markers_qubit_buffer, markers_ch3m1)
        generated_sequences = self.psb.build(self.pulse_sequence_matrix,self.total_flux_pulse_span_length_list)

        self.markers['readout pulse'], self.markers['card trigger'], self.waveforms['qubit drive I'], self.waveforms[
            'qubit drive Q'], self.waveforms['qubit 1 flux'], self.markers['qubit buffer'], self.markers['ch3m1'] = generated_sequences

    def reshape_data(self, data):
        return np.reshape(data, (self.sequence_length, self.waveform_length))