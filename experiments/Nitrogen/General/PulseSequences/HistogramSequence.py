__author__ = 'Nelson'

from slab.instruments.awg.PulseSequence import *
from slab.experiments.ExpLib import awgpulses as ap
from numpy import arange, linspace
from slab.experiments.ExpLib.TEK1PulseOrganizer import *

from liveplot import LivePlotClient


class HistogramSequence(PulseSequence):
    def __init__(self, awg_info, histo_cfg, readout_cfg, pulse_cfg, buffer_cfg):

        self.histo_cfg = histo_cfg
        self.histo_pts = [0, 1]

        self.start_end_buffer = buffer_cfg['tek1_start_end']
        self.marker_start_buffer = buffer_cfg['marker_start']

        PulseSequence.__init__(self, "Histogram", awg_info, sequence_length=2)
        self.pulse_cfg=pulse_cfg
        self.pulse_type = histo_cfg['pulse_type']
        self.a = pulse_cfg['a']
        self.pi_length = pulse_cfg['pi_length']
        print self.pi_length
        self.measurement_delay = readout_cfg['delay']
        self.measurement_width = readout_cfg['width']
        self.card_delay = readout_cfg['card_delay']
        self.card_trig_width = readout_cfg['card_trig_width']

        if self.pulse_type == 'square':
            self.ramp_sigma = pulse_cfg['ramp_sigma']
            max_pulse_width = self.pi_length + 4 * self.ramp_sigma

        if self.pulse_type == 'gauss':
            max_pulse_width = 6 * self.pi_length

        self.max_length = round_samples((max_pulse_width + self.measurement_delay + self.measurement_width + 2*self.start_end_buffer))
        self.origin = self.max_length - (self.measurement_delay + self.measurement_width + self.start_end_buffer)

        self.set_all_lengths(self.max_length)
        self.set_waveform_length("qubit 1 flux", 1)

    def build_sequence(self):
        PulseSequence.build_sequence(self)

        wtpts = self.get_waveform_times('qubit drive I')
        mtpts = self.get_marker_times('qubit buffer')

        for ii, d in enumerate(self.histo_pts):
            self.markers['readout pulse'][ii] = ap.square(mtpts, 1, self.origin + self.measurement_delay,
                                                           self.measurement_width)
            self.markers['card trigger'][ii] = ap.square(mtpts, 1,
                                                          self.origin - self.card_delay + self.measurement_delay,
                                                          self.card_trig_width)

            w = self.pi_length
            a = d * self.a

            if self.pulse_type == 'square':
                self.waveforms['qubit drive I'][ii], self.waveforms['qubit drive Q'][ii] = \
                    ap.sideband(wtpts,
                                 ap.square(wtpts, a, self.origin - w - 2 * self.ramp_sigma , w,
                                            self.ramp_sigma), np.zeros(len(wtpts)),
                                 self.pulse_cfg['iq_freq'], 0)

                self.markers['qubit buffer'][ii] = ap.square(mtpts, 1,
                                                              self.origin - w - 4 * self.ramp_sigma - self.marker_start_buffer,
                                                              w + 4 * self.ramp_sigma + self.marker_start_buffer)

            if self.pulse_type == 'gauss':
                gauss_sigma = w  #covention for the width of gauss pulse
                self.waveforms['qubit drive I'][ii], self.waveforms['qubit drive Q'][ii] = \
                    ap.sideband(wtpts,
                                 ap.gauss(wtpts, a, self.origin - 3 * gauss_sigma , gauss_sigma),
                                 np.zeros(len(wtpts)),
                                 self.pulse_cfg['iq_freq'], 0)

                self.markers['qubit buffer'][ii] = ap.square(mtpts, 1, self.origin - 6 * gauss_sigma - self.marker_start_buffer ,
                                                              6 * gauss_sigma + self.marker_start_buffer)

    def reshape_data(self, data):
        return np.reshape(data, (self.sequence_length, self.waveform_length))
