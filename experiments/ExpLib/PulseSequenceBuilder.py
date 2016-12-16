__author__ = 'Nelson'

from slab.instruments.awg.PulseSequence import *
from slab.experiments.ExpLib import awgpulses as ap
from numpy import arange, linspace
from slab.experiments.ExpLib.PulseWaveformBuildingLibrary import *
from slab.instruments.pulseblaster.pulseblaster import start_pulseblaster
import math
from liveplot import LivePlotClient


class Pulse():
    def __init__(self, target, name, type, amp, length, freq, phase, span_length):
        self.target = target
        self.name = name
        self.type = type
        self.amp = amp
        self.length = length
        self.freq = freq
        self.phase = phase
        self.span_length = span_length


class PulseSequenceBuilder():
    def __init__(self, cfg):
        buffer_cfg = cfg['buffer']
        self.cfg = cfg
        self.exp_period_ns = cfg['expt_trigger']['period_ns']
        self.start_end_buffer = buffer_cfg['tek1_start_end']
        self.marker_start_buffer = buffer_cfg['marker_start']
        self.marker_end_buffer = buffer_cfg['marker_end']
        self.tek2_trigger_delay = buffer_cfg['tek2_trigger_delay']
        self.pulse_cfg = cfg['pulse_info']
        self.readout_cfg = cfg['readout']
        self.pulse_sequence_list = []
        self.total_pulse_span_length = 0
        self.total_flux_pulse_span_length = 0
        self.flux_pulse_started = False
        self.pulse_span_length_list_temp = []
        self.qubit_cfg = cfg['qubit']

    def append(self, target, name, type='gauss', amp=0, length=0, freq=0, phase=None, **kwargs):
        '''
        Append a pulse in the pulse sequence.
        '''
        if target == "q":
            if name == "0":
                amp = 0
                length = self.pulse_cfg[type]['pi_length']
                freq = self.pulse_cfg[type]['iq_freq']
                if phase == None:
                    phase = self.pulse_cfg[type]['phase']
            if name == "pi" or name == "cal_pi":
                amp = self.pulse_cfg[type]['pi_a']
                length = self.pulse_cfg[type]['pi_length']
                # print amp
                # print length
                freq = self.pulse_cfg[type]['iq_freq']
                if phase == None:
                    phase = self.pulse_cfg[type]['phase']
            if name == "half_pi":
                amp = self.pulse_cfg[type]['half_pi_a']
                length = self.pulse_cfg[type]['half_pi_length']
                freq = self.pulse_cfg[type]['iq_freq']
                if phase == None:
                    phase = self.pulse_cfg[type]['phase']
            if name == "neg_half_pi":
                amp = self.pulse_cfg[type]['half_pi_a']
                length = self.pulse_cfg[type]['half_pi_length']
                freq = self.pulse_cfg[type]['iq_freq']
                if phase == None:
                    phase = self.pulse_cfg[type]['phase']+180
            if name == "pi_y":
                amp = self.pulse_cfg[type]['pi_a']
                length = self.pulse_cfg[type]['pi_length']
                freq = self.pulse_cfg[type]['iq_freq']
                if phase == None:
                    phase = self.pulse_cfg[type]['y_phase']
            if name == "half_pi_y":
                amp = self.pulse_cfg[type]['half_pi_a']
                length = self.pulse_cfg[type]['half_pi_length']
                freq = self.pulse_cfg[type]['iq_freq']
                if phase == None:
                    phase = self.pulse_cfg[type]['y_phase']
            if name == "neg_half_pi_y":
                amp = self.pulse_cfg[type]['half_pi_a']
                length = self.pulse_cfg[type]['half_pi_length']
                freq = self.pulse_cfg[type]['iq_freq']
                if phase == None:
                    phase = self.pulse_cfg[type]['y_phase']+180
            if name == "pi_q_ef":
                amp = self.pulse_cfg[type]['pi_ef_a']
                length = self.pulse_cfg[type]['pi_ef_length']
                freq = self.pulse_cfg[type]['iq_freq']+self.qubit_cfg['alpha']
                if phase == None:
                    phase = 0

            if name == "half_pi_q_ef":
                amp = self.pulse_cfg[type]['pi_ef_a']
                length = self.pulse_cfg[type]['half_pi_ef_length']
                freq = self.pulse_cfg[type]['iq_freq']+self.qubit_cfg['alpha']
                if phase == None:
                    phase = 0

            pulse_span_length = ap.get_pulse_span_length(self.pulse_cfg, type, length)
            if self.flux_pulse_started:
                self.pulse_span_length_list_temp.append(pulse_span_length)
        elif target[:4] == "q,mm":
            self.flux_pulse_started = True
            mm_target = int(target[4:])
            mm_target_info = self.cfg['multimodes'][mm_target]
            freq = mm_target_info['flux_pulse_freq']
            if name == "pi_ge":
                amp = mm_target_info['a']
                length = mm_target_info['flux_pi_length']
                type=mm_target_info['flux_pulse_type']
            if name == "2pi_ge":
                amp = mm_target_info['a']
                length = mm_target_info['flux_2pi_length']
                type=mm_target_info['flux_pulse_type']
            if name == "pi_ef":
                amp = mm_target_info['a_ef']
                length = mm_target_info['flux_pi_length_ef']
                freq = mm_target_info['flux_pulse_freq_ef']
                type=mm_target_info['flux_pulse_type_ef']
            if name == "2pi_ef":
                amp = mm_target_info['a_ef']
                length = mm_target_info['flux_2pi_length_ef']
                freq = mm_target_info['flux_pulse_freq_ef']
                type=mm_target_info['flux_pulse_type_ef']

            pulse_span_length = ap.get_pulse_span_length(self.cfg['flux_pulse_info'], type, length)
            flux_pulse_span_length = pulse_span_length
            for span_length_temp in self.pulse_span_length_list_temp:
                flux_pulse_span_length += span_length_temp
            self.pulse_span_length_list_temp = []
            self.total_flux_pulse_span_length += flux_pulse_span_length
        elif target[:4] == "q:mm":
            self.flux_pulse_started = True
            pulse_span_length = ap.get_pulse_span_length(self.cfg['flux_pulse_info'], type, length)
            flux_pulse_span_length = pulse_span_length
            for span_length_temp in self.pulse_span_length_list_temp:
                flux_pulse_span_length += span_length_temp
            self.pulse_span_length_list_temp = []
            self.total_flux_pulse_span_length += flux_pulse_span_length

        else:
            raise ValueError('Wrong target has been defined')

        if phase == None:
            phase = 0

        pulse = Pulse(target, name, type, amp, length, freq, phase, pulse_span_length)

        self.pulse_sequence_list.append(pulse)
        self.total_pulse_span_length += pulse_span_length

    def idle(self, length):
        '''
        Append an idle in the pulse sequence.
        '''
        pulse_info = Pulse('idle', 'idle', 'idle', 0, length, 0, 0, length)

        self.pulse_sequence_list.append(pulse_info)
        self.total_pulse_span_length += length
        self.total_flux_pulse_span_length += length

    def get_pulse_sequence(self):
        '''
        Being called externally to obtain the pulse sequence.
        '''
        pulse_sequence_list = self.pulse_sequence_list
        self.pulse_sequence_list = []
        return pulse_sequence_list

    def get_total_pulse_span_length(self):
        '''
        Being called externally to obtain the total pulse span length.
        '''
        total_pulse_span_length = self.total_pulse_span_length
        self.total_pulse_span_length = 0
        return total_pulse_span_length

    def get_total_flux_pulse_span_length(self):
        '''
        Being called externally to obtain the total pulse span length.
        '''
        total_flux_pulse_span_length = self.total_flux_pulse_span_length
        self.total_flux_pulse_span_length = 0
        self.flux_pulse_started = False
        self.pulse_span_length_list_temp = []
        return total_flux_pulse_span_length

    def acquire_readout_cfg(self):
        '''
        Being called internally to obtain the readout parameters.
        '''
        self.measurement_delay = self.readout_cfg['delay']

        self.measurement_width = self.readout_cfg['width']
        self.card_delay = self.readout_cfg['card_delay']
        self.card_trig_width = self.readout_cfg['card_trig_width']

        if self.card_delay + self.card_trig_width >= self.measurement_delay:
            raise ValueError('measurement delay should be larger than card delay by more than card_trig_width!')

    def get_max_length(self, total_pulse_span_length_list):
        '''
        Calculate the maximum of total pulse + marker length, of all the sequences.
        '''
        self.acquire_readout_cfg()
        max_total_pulse_span_length = max(total_pulse_span_length_list)
        max_total_pulse_span_length += self.tek2_trigger_delay  #
        self.max_length = round_samples(
            (max_total_pulse_span_length + self.measurement_delay + self.measurement_width + 2 * self.start_end_buffer))
        return self.max_length

    def get_max_flux_length(self, total_flux_pulse_span_length_list):
        '''
        Calculate the maximum of total pulse + marker length, of all the sequences.
        '''
        self.max_total_flux_pulse_span_length = max(total_flux_pulse_span_length_list)

        return self.max_total_flux_pulse_span_length

    def prepare_build(self, wtpts, mtpts, ftpts, markers_readout, markers_card, waveforms_qubit_I, waveforms_qubit_Q,
                      waveforms_qubit_flux,
                      markers_qubit_buffer, markers_ch3m1):
        '''
        Being called internally to set the variables.
        '''
        self.wtpts = wtpts
        self.mtpts = mtpts
        self.ftpts = ftpts
        self.markers_readout = markers_readout
        self.markers_card = markers_card
        self.waveforms_qubit_I = waveforms_qubit_I
        self.waveforms_qubit_Q = waveforms_qubit_Q
        self.waveforms_qubit_flux = waveforms_qubit_flux
        self.markers_qubit_buffer = markers_qubit_buffer
        self.markers_ch3m1 = markers_ch3m1


    def build(self, pulse_sequence_matrix, total_pulse_span_length_list):
        '''
        Parse the pulse sequence matrix generated previously.
        For each pulse sequence, location of readout and card is fixed.
        Pulses are appended backward, from the last pulse to the first pulse.
        '''
        def roundup100(x):
            return int(math.ceil(x / 100.0)) * 100
        self.origin = roundup100(self.max_length - (self.measurement_delay + self.measurement_width + self.start_end_buffer))
        self.uses_tek2 = False
        awg_trig_len=self.cfg['triggers']['awg']
        start_pulseblaster(self.exp_period_ns, awg_trig_len,self.origin+self.card_delay, self.origin + self.measurement_delay, self.card_trig_width, self.measurement_width)
        for ii in range(len(pulse_sequence_matrix)):
            self.markers_readout[ii] = ap.square(self.mtpts, 1, self.origin + self.measurement_delay,
                                                 self.measurement_width)
            self.markers_card[ii] = ap.square(self.mtpts, 1,
                                              self.origin - self.card_delay + self.measurement_delay,
                                              self.card_trig_width)
            self.waveforms_qubit_I[ii], self.waveforms_qubit_Q[ii] = \
                ap.sideband(self.wtpts,
                            np.zeros(len(self.wtpts)), np.zeros(len(self.wtpts)),
                            0, 0)
            self.waveforms_qubit_flux[ii] = ap.sideband(self.ftpts,
                                                        np.zeros(len(self.ftpts)), np.zeros(len(self.ftpts)),
                                                        0, 0)[0]
            self.markers_qubit_buffer[ii] = ap.square(self.mtpts, 0, 0, 0)
            pulse_location = 0
            flux_pulse_location = total_pulse_span_length_list[ii]
            flux_pulse_started = False
            flux_end_location = 0
            # The range defined in this way means having the for loop with index backward.
            for jj in range(len(pulse_sequence_matrix[ii]) - 1, -1, -1):
                pulse_defined = True
                pulse = pulse_sequence_matrix[ii][jj]
                if pulse.target == "q":
                    if pulse.type == "square":
                        qubit_waveforms, qubit_marker = square(self.wtpts, self.mtpts, self.origin,
                                                               self.marker_start_buffer, self.marker_end_buffer,pulse_location, pulse,
                                                               self.pulse_cfg)
                    elif pulse.type == "gauss":
                        qubit_waveforms, qubit_marker = gauss(self.wtpts, self.mtpts, self.origin,
                                                              self.marker_start_buffer, self.marker_end_buffer,
                                                              pulse_location, pulse)
                    else:
                        raise ValueError('Wrong pulse type has been defined')
                    if pulse_defined:
                        self.waveforms_qubit_I[ii] += qubit_waveforms[0]
                        self.waveforms_qubit_Q[ii] += qubit_waveforms[1]
                        self.markers_qubit_buffer[ii] += qubit_marker

                elif pulse.target[:4] == "q,mm":
                    self.uses_tek2 = True
                    if flux_pulse_started == False:
                        flux_end_location = pulse_location
                        flux_pulse_started = True
                    mm_target = int(pulse.target[4])
                    mm_target_info = self.cfg['multimodes'][mm_target]
                    flux_pulse_info = self.cfg['flux_pulse_info']
                    if pulse.type == "square":
                        # waveforms_qubit_flux = flux_square(self.ftpts, flux_pulse_location, pulse,
                        #                                    self.cfg['flux_pulse_info'])

                        waveforms_qubit_flux = flux_square_phase_fix(self.ftpts, flux_pulse_location, pulse,
                                                           self.cfg['flux_pulse_info'],mm_target_info, flux_pulse_info)
                    elif pulse.type == "gauss":
                        waveforms_qubit_flux = flux_gauss(self.ftpts, flux_pulse_location, pulse)
                    else:
                        raise ValueError('Wrong pulse type has been defined')
                    if pulse_defined:
                        self.waveforms_qubit_flux[ii] += waveforms_qubit_flux
                elif pulse.target[:4] == "q:mm":
                    self.uses_tek2 = True
                    if flux_pulse_started == False:
                        flux_end_location = pulse_location
                        flux_pulse_started = True
                    if pulse.type == "square":
                        waveforms_qubit_flux = flux_square(self.ftpts, flux_pulse_location, pulse,
                                                           self.cfg['flux_pulse_info'])
                    elif pulse.type == "gauss":
                        waveforms_qubit_flux = flux_gauss(self.ftpts, flux_pulse_location, pulse)
                    else:
                        raise ValueError('Wrong pulse type has been defined')
                    if pulse_defined:
                        self.waveforms_qubit_flux[ii] += waveforms_qubit_flux
                elif pulse.target == "idle":
                    pass

                high_values_indices = self.markers_qubit_buffer[ii] > 1
                self.markers_qubit_buffer[ii][high_values_indices] = 1

                pulse_location += pulse.span_length
                if flux_pulse_started:
                    flux_pulse_location -= pulse.span_length

            if self.uses_tek2:
                self.markers_ch3m1[ii] = ap.square(self.mtpts, 1,
                                                self.origin - flux_end_location - total_pulse_span_length_list[
                                                    ii] - self.tek2_trigger_delay,
                                                self.card_trig_width)
        return (self.markers_readout,
                self.markers_card,
                self.waveforms_qubit_I,
                self.waveforms_qubit_Q,
                self.waveforms_qubit_flux,
                self.markers_qubit_buffer,
                self.markers_ch3m1)
