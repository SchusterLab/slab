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
        if self.rabi_cfg.step is not None:
            self.rabi_pts = arange(self.rabi_cfg.start, self.rabi_cfg.stop, self.rabi_cfg.step)
        else:
            self.rabi_pts = linspace(self.rabi_cfg.start, self.rabi_cfg.stop, self.rabi_cfg.steps)
        self.sweep_type = self.sweep_type
        self.pulse_type = self.pulse_type
        self.a = rabi_cfg.a
        self.w = rabi_cfg.w
        self.sigma = rabi_cfg.sigma
        self.freq = rabi_cfg.freq
        self.phase = rabi_cfg.phase
        self.measurement_delay = readout_cfg.delay
        self.measurement_width = readout_cfg.width
        self.card_delay = readout_cfg.card_delay
        self.monitor_pulses = readout_cfg.monitor_pulses
        self.card_trig_width = readout_cfg.card_trig_width

        sequence_length = len(self.rabi_pts)

        if self.rabi_cfg.sweep_type == 'time':
            max_pulse_width = max(self.rabi_pts) + 4 * self.sigma
        else:
            max_pulse_width = self.w + 4 * self.sigma

        self.max_length = max_pulse_width + self.measurement_delay + self.measurement_width
        self.origin = self.max_length - self.measurement_delay + self.measurement_width + 500

        PulseSequence.__init__(self, awg_info, self.max_length, sequence_length)

    def build_sequence(self):
        wtpts = self.get_waveform_times('qubit drive I')
        mtpts = self.get_marker_times['qubit buffer']
        for ii, d in enumerate(self.rabi_pts):
            self.markers['readout pulse'][ii] = ap2.square(mtpts, 1, self.origin,
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

            self.waveforms['qubit drive I'][ii], self.waveforms['qubit drive Q'][ii] = \
                ap2.sideband(wtpts,
                             ap2.square(wtpts, self.origin - w - 2 * self.sigma, w, self.sigma), np.zeros(len(wtpts)),
                             self.freq, self.phase)

            self.markers['qubit buffer'] = ap2.square(mtpts, 1, self.origin - w - 2 * self.sigma - 100,
                                                      w + 4 * self.sigma + 100)

    def reshape_data(self, data):
        return np.reshape(data, (self.sequence_length, self.waveform_length))