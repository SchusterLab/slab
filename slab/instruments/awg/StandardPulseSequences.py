__author__ = 'dave'

from slab.instruments.awg.PulseSequence import *
from slab.instruments.awg import awgpulses2 as ap2
from numpy import arange, linspace

class RabiSequence(PulseSequence):
    def __init__(self, awg_info, rabi_cfg, readout_cfg):
        """
        rabi_cfg = {rabi_pts, sweep_type="time", pulse_type="square", a=0, w=0, sigma=0, freq=0, phase=0}
        readout_cfg = {"Q": 10000, "delay": , "width": , "card_delay": , "card_trig_width":  }
        """

        self.rabi_cfg=rabi_cfg
        if self.rabi_cfg['step'] is not None:
            self.rabi_pts = arange(self.rabi_cfg['start'], self.rabi_cfg['stop'], self.rabi_cfg['step'])
        else:
            self.rabi_pts = linspace(self.rabi_cfg['start'], self.rabi_cfg['stop'], self.rabi_cfg['num_pts'])
        sequence_length = len(self.rabi_pts)

        PulseSequence.__init__(self, "Rabi", awg_info, sequence_length)

        self.sweep_type = rabi_cfg['sweep_type']
        self.pulse_type = rabi_cfg['pulse_type']
        self.a = rabi_cfg['a']
        self.w = rabi_cfg['w']

        self.freq = rabi_cfg['freq']
        self.phase = rabi_cfg['phase']
        self.measurement_delay = readout_cfg['delay']
        self.measurement_width = readout_cfg['width']
        self.card_delay = readout_cfg['card_delay']
        self.monitor_pulses = readout_cfg['monitor_pulses']
        self.card_trig_width = readout_cfg['card_trig_width']

        if self.pulse_type == 'square':
            self.ramp_sigma = rabi_cfg['ramp_sigma']
            if self.rabi_cfg['sweep_type'] == 'time':
                max_pulse_width = max(self.rabi_pts) + 4 * self.ramp_sigma
            else:
                max_pulse_width = self.w + 4 * self.ramp_sigma

        if self.pulse_type == 'gauss':
            if self.rabi_cfg['sweep_type'] == 'time':
                max_pulse_width = 6*max(self.rabi_pts)
            else:
                max_pulse_width = 6*self.w

        self.max_length = round_samples((max_pulse_width + self.measurement_delay + self.measurement_width+1000))
        self.origin = self.max_length - (self.measurement_delay + self.measurement_width + 500)

        self.set_all_lengths(self.max_length)

    def build_sequence(self):
        PulseSequence.build_sequence(self)

        wtpts = self.get_waveform_times('qubit drive I')
        mtpts = self.get_marker_times('qubit buffer')

        for ii, d in enumerate(self.rabi_pts):
            self.markers['readout pulse'][ii] = ap2.square(mtpts, 1, self.origin+self.measurement_delay,
                                                           self.measurement_width)
            self.markers['card trigger'][ii] = ap2.square(mtpts, 1,
                                                          self.origin - self.card_delay + self.measurement_delay,
                                                          self.card_trig_width)

            if self.sweep_type.upper() == 'time'.upper():
                a = self.a
                w = d
            else:
                a = d
                w = self.w

            if self.pulse_type=='square':
                self.waveforms['qubit drive I'][ii], self.waveforms['qubit drive Q'][ii] = \
                    ap2.sideband(wtpts,
                                 ap2.square(wtpts, a,self.origin - w - 2 * self.ramp_sigma, w, self.ramp_sigma), np.zeros(len(wtpts)),
                                 self.freq, self.phase)

                self.markers['qubit buffer'][ii] = ap2.square(mtpts, 1, self.origin - w - 4 * self.ramp_sigma - 100,
                                                          w + 4 * self.ramp_sigma + 200)

            if self.pulse_type=='gauss':
                gauss_sigma=w #covention for the width of gauss pulse
                self.waveforms['qubit drive I'][ii], self.waveforms['qubit drive Q'][ii] = \
                    ap2.sideband(wtpts,
                                 ap2.gauss(wtpts, a,self.origin - 3* gauss_sigma , gauss_sigma), np.zeros(len(wtpts)),
                                 self.freq, self.phase)

                self.markers['qubit buffer'][ii] = ap2.square(mtpts, 1, self.origin - 6*gauss_sigma - 100,
                                                          6*gauss_sigma + 200)

    def reshape_data(self, data):
        return np.reshape(data, (self.sequence_length, self.waveform_length))


class T1Sequence(PulseSequence):
    def __init__(self, awg_info, t1_cfg, readout_cfg, pulse_cfg):
        """
        rabi_cfg = {rabi_pts, sweep_type="time", pulse_type="square", a=0, w=0, sigma=0, freq=0, phase=0}
        readout_cfg = {"Q": 10000, "delay": , "width": , "card_delay": , "card_trig_width":  }
        """

        self.t1_cfg=t1_cfg
        if self.t1_cfg['step'] is not None:
            self.t1_pts = arange(self.t1_cfg['start'], self.t1_cfg['stop'], self.t1_cfg['step'])
        else:
            self.t1_pts = linspace(self.t1_cfg['start'], self.t1_cfg['stop'], self.t1_cfg['num_pts'])

        PulseSequence.__init__(self, "T1", awg_info,sequence_length=len(self.t1_pts))


        self.pulse_type = t1_cfg['pulse_type']
        self.start = t1_cfg['start']
        self.stop = t1_cfg['stop']
        self.step = t1_cfg['step']
        self.a = pulse_cfg['a']
        self.pi_length = pulse_cfg['pi_length']
        self.freq = pulse_cfg['freq']
        self.phase = pulse_cfg['phase']
        self.measurement_delay = readout_cfg['delay']
        self.measurement_width = readout_cfg['width']
        self.card_delay = readout_cfg['card_delay']
        self.monitor_pulses = readout_cfg['monitor_pulses']
        self.card_trig_width = readout_cfg['card_trig_width']

        if self.pulse_type == 'square':
            self.ramp_sigma = pulse_cfg['ramp_sigma']
            max_pulse_width = self.pi_length + 4 * self.ramp_sigma

        if self.pulse_type == 'gauss':
            max_pulse_width = 6*self.pi_length

        self.max_length = round_samples((max_pulse_width + self.measurement_delay + self.measurement_width+self.stop+1000))
        self.origin = self.max_length - (self.measurement_delay + self.measurement_width + 500)

        self.set_all_lengths(self.max_length)
        self.set_waveform_length("qubit 1 flux", 1)

    def build_sequence(self):
        PulseSequence.build_sequence(self)

        wtpts = self.get_waveform_times('qubit drive I')
        mtpts = self.get_marker_times('qubit buffer')

        for ii, d in enumerate(self.t1_pts):
            self.markers['readout pulse'][ii] = ap2.square(mtpts, 1, self.origin+self.measurement_delay,
                                                           self.measurement_width)
            self.markers['card trigger'][ii] = ap2.square(mtpts, 1,
                                                          self.origin - self.card_delay + self.measurement_delay,
                                                          self.card_trig_width)

            delay = d
            w=self.pi_length
            a=self.a

            if self.pulse_type == 'square':
                self.waveforms['qubit drive I'][ii], self.waveforms['qubit drive Q'][ii] = \
                    ap2.sideband(wtpts,
                                 ap2.square(wtpts, a,self.origin - w - 2 * self.ramp_sigma - delay, w, self.ramp_sigma), np.zeros(len(wtpts)),
                                  self.freq, self.phase)

                self.markers['qubit buffer'][ii] = ap2.square(mtpts, 1, self.origin - w - 4 * self.ramp_sigma - delay - 100,
                                                          w + 4 * self.ramp_sigma + 200)

            if self.pulse_type == 'gauss':
                gauss_sigma=w #covention for the width of gauss pulse
                self.waveforms['qubit drive I'][ii], self.waveforms['qubit drive Q'][ii] = \
                    ap2.sideband(wtpts,
                                 ap2.gauss(wtpts, a,self.origin - 3* gauss_sigma - delay, gauss_sigma), np.zeros(len(wtpts)),
                                  self.freq, self.phase)

                self.markers['qubit buffer'][ii] = ap2.square(mtpts, 1, self.origin - 6*gauss_sigma - 100 - delay ,
                                                          6*gauss_sigma + 200)

    def reshape_data(self, data):
        return np.reshape(data, (self.sequence_length, self.waveform_length))


class RamseySequence(PulseSequence):
    def __init__(self, awg_info, ramsey_cfg, readout_cfg, pulse_cfg):
        """
        rabi_cfg = {rabi_pts, sweep_type="time", pulse_type="square", a=0, w=0, sigma=0, freq=0, phase=0}
        readout_cfg = {"Q": 10000, "delay": , "width": , "card_delay": , "card_trig_width":  }
        """

        self.ramsey_cfg=ramsey_cfg
        if self.ramsey_cfg['step'] is not None:
            self.ramsey_pts = arange(self.ramsey_cfg['start'], self.ramsey_cfg['stop'], self.ramsey_cfg['step'])
        else:
            self.ramsey_pts = linspace(self.ramsey_cfg['start'], self.ramsey_cfg['stop'], self.ramsey_cfg['num_pts'])

        sequence_length = len(self.ramsey_pts)
        PulseSequence.__init__(self, "Ramsey", awg_info, sequence_length)

        self.pulse_type = ramsey_cfg['pulse_type']
        self.start = ramsey_cfg['start']
        self.stop = ramsey_cfg['stop']
        self.step = ramsey_cfg['step']

        self.a = pulse_cfg['a']
        self.half_pi_length = pulse_cfg['half_pi_length']
        self.freq = pulse_cfg['freq']
        self.phase = pulse_cfg['phase']
        self.measurement_delay = readout_cfg['delay']
        self.measurement_width = readout_cfg['width']
        self.card_delay = readout_cfg['card_delay']
        self.monitor_pulses = readout_cfg['monitor_pulses']
        self.card_trig_width = readout_cfg['card_trig_width']

        if self.pulse_type == 'square':
            self.ramp_sigma = pulse_cfg['ramp_sigma']
            max_pulse_width = self.half_pi_length + 4 * self.ramp_sigma

        if self.pulse_type == 'gauss':
            max_pulse_width = 6*self.half_pi_length

        self.max_length = round_samples((2*max_pulse_width + self.measurement_delay + self.measurement_width+self.stop+1000))
        self.origin = self.max_length - (self.measurement_delay + self.measurement_width + 500)

        self.set_all_lengths(self.max_length)
        self.set_waveform_length("qubit 1 flux", 1)

    def build_sequence(self):
        PulseSequence.build_sequence(self)
        wtpts = self.get_waveform_times('qubit drive I')
        mtpts = self.get_marker_times('qubit buffer')

        for ii, d in enumerate(self.ramsey_pts):
            self.markers['readout pulse'][ii] = ap2.square(mtpts, 1, self.origin+self.measurement_delay,
                                                           self.measurement_width)
            self.markers['card trigger'][ii] = ap2.square(mtpts, 1,
                                                          self.origin - self.card_delay + self.measurement_delay,
                                                          self.card_trig_width)

            delay = d
            w=self.half_pi_length
            a=self.a

            if self.pulse_type == 'square':
                self.waveforms['qubit drive I'][ii], self.waveforms['qubit drive Q'][ii] = \
                    np.add(ap2.sideband(wtpts,
                                 ap2.square(wtpts, a,self.origin - w - 2 * self.ramp_sigma, w, self.ramp_sigma), np.zeros(len(wtpts)),
                                  self.freq, self.phase),ap2.sideband(wtpts,
                                 ap2.square(wtpts, a,self.origin - w - 2 * self.ramp_sigma - delay, w, self.ramp_sigma), np.zeros(len(wtpts)),
                                  self.freq, self.phase))

                self.markers['qubit buffer'][ii] = np.add(ap2.square(mtpts, 1, self.origin - w - 4 * self.ramp_sigma - 100,
                                                          w + 4 * self.ramp_sigma + 200),(ap2.square(mtpts, 1, self.origin - w - 4 * self.ramp_sigma - delay - 100,
                                                          w + 4 * self.ramp_sigma + 200)))
                high_values_indices = self.markers['qubit buffer'][ii] > 1
                self.markers['qubit buffer'][ii][high_values_indices] = 1

            if self.pulse_type == 'gauss':
                gauss_sigma = w
                self.waveforms['qubit drive I'][ii], self.waveforms['qubit drive Q'][ii] = \
                    np.add(ap2.sideband(wtpts,
                                 ap2.gauss(wtpts, a,self.origin - 3* gauss_sigma, w), np.zeros(len(wtpts)),
                                  self.freq, self.phase),ap2.sideband(wtpts,
                                 ap2.gauss(wtpts, a,self.origin - 3* gauss_sigma - delay, w), np.zeros(len(wtpts)),
                                  self.freq, self.phase))

                self.markers['qubit buffer'][ii] = np.add(ap2.square(mtpts, 1, self.origin - 6*gauss_sigma-100,
                                                         6*gauss_sigma + 200),(ap2.square(mtpts, 1, self.origin - 6*gauss_sigma - delay - 100,
                                                          6*gauss_sigma + 200)))
                high_values_indices = self.markers['qubit buffer'][ii] > 1
                self.markers['qubit buffer'][ii][high_values_indices] = 1

    def reshape_data(self, data):
        return np.reshape(data, (self.sequence_length, self.waveform_length))