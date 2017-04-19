__author__ = 'Nelson'

from slab.instruments.awg.PulseSequence import *
from slab.experiments.ExpLib import awgpulses as ap
from numpy import arange, linspace
from slab.instruments.pulseblaster.pulseblaster import start_pulseblaster

from liveplot import LivePlotClient


class HistogramSequence(PulseSequence):
    def __init__(self, awg_info, histo_cfg, readout_cfg, pulse_cfg, buffer_cfg, cfg):

        self.cfg = cfg
        self.readout_cfg = readout_cfg
        self.histo_cfg = histo_cfg

        if self.histo_cfg['include_f']:
            seq_len = 3
        else:
            seq_len = 2

        self.histo_pts = range(seq_len)

        self.start_end_buffer = buffer_cfg['tek1_start_end']
        self.marker_start_buffer = buffer_cfg['marker_start']

        PulseSequence.__init__(self, "Histogram", awg_info, sequence_length=seq_len)
        self.exp_period_ns = self.cfg['expt_trigger']['period_ns']
        self.pulse_cfg=pulse_cfg
        self.pulse_type = histo_cfg['pulse_type']
        self.pi_a = pulse_cfg['pi_a']
        self.pi_length = pulse_cfg['pi_length']
        self.pi_ef_a = pulse_cfg['pi_ef_a']
        self.pi_ef_length = pulse_cfg['pi_ef_length']
        self.measurement_delay = readout_cfg['delay']
        self.measurement_width = readout_cfg['width']
        self.card_delay = readout_cfg['card_delay']
        self.card_trig_width = readout_cfg['card_trig_width']

        if self.pulse_type == 'square':
            self.ramp_sigma = pulse_cfg['ramp_sigma']
            self.max_pulse_width = (self.pi_length + 4 * self.ramp_sigma) + (self.pi_ef_length + 4 * self.ramp_sigma)

        if self.pulse_type == 'gauss':
            self.max_pulse_width = 6 * self.pi_length + 6 * self.pi_ef_length

        self.max_length = round_samples((self.max_pulse_width + self.measurement_delay + self.measurement_width + 2*self.start_end_buffer))
        self.origin = self.max_length - (self.measurement_delay + self.measurement_width + self.start_end_buffer)

        self.set_all_lengths(self.max_length)
        self.set_waveform_length("qubit 1 flux", 1)

    def build_sequence(self):
        PulseSequence.build_sequence(self)

        wtpts = self.get_waveform_times('qubit drive I')
        mtpts = self.get_marker_times('qubit buffer')

        # todo: why no pulse blaster code here? c.f. vacuum rabi

        awg_trig_len = 100
        start_pulseblaster(self.exp_period_ns, awg_trig_len, self.origin + self.card_delay,
                           self.origin + self.measurement_delay,
                           self.card_trig_width, self.measurement_width)


        for ii, d in enumerate(self.histo_pts):

            self.markers['readout pulse'][ii] = ap.square(mtpts, 1, self.origin + self.measurement_delay,
                                                           self.measurement_width)
            self.markers['card trigger'][ii] = ap.square(mtpts, 1,
                                                          self.origin - self.card_delay + self.measurement_delay,
                                                          self.card_trig_width)

            w = self.pi_length
            if ii>0:
                a = self.pi_a
            else:
                a = 0

            w_ef = self.pi_ef_length
            if ii>1:
                a_ef = self.pi_ef_a
            else:
                a_ef = 0

            self.waveforms['qubit drive I'][ii] = np.zeros(len(wtpts))
            self.waveforms['qubit drive Q'][ii] = np.zeros(len(wtpts))

            if self.pulse_type == 'square':
                pulsedata = ap.square(wtpts, a,
                                      self.origin - (w + 2 * self.ramp_sigma) - (w_ef + 4 * self.ramp_sigma), w,
                                      self.ramp_sigma)
                pulsedata_ef = ap.square(wtpts, a_ef, self.origin - (w + 2 * self.ramp_sigma), w_ef, self.ramp_sigma)

            if self.pulse_type == 'gauss':
                pulsedata = ap.gauss(wtpts, a, self.origin - 3 * w - 6 * w_ef, w)
                pulsedata_ef = ap.gauss(wtpts, a_ef, self.origin - 3 * w_ef, w_ef)

            temp_I, temp_Q = ap.sideband(wtpts, pulsedata, np.zeros(len(wtpts)), self.pulse_cfg['iq_freq'], 0)
            self.waveforms['qubit drive I'][ii] += temp_I
            self.waveforms['qubit drive Q'][ii] += temp_Q

            # ef pulse shifted in freq by alpha
            # consistent with implementation in ef_rabi
            temp_I_ef, temp_Q_ef = ap.sideband(wtpts, pulsedata_ef, np.zeros(len(wtpts)),
                                         self.pulse_cfg['iq_freq'] + self.cfg['qubit']['alpha'], 0)
            self.waveforms['qubit drive I'][ii] += temp_I_ef
            self.waveforms['qubit drive Q'][ii] += temp_Q_ef

            self.markers['qubit buffer'][ii] = ap.square(mtpts, 1,
                                                         self.origin - self.max_pulse_width - self.marker_start_buffer,
                                                         self.max_pulse_width + self.marker_start_buffer)

            ## heterodyne pulse
            self.marker_start_buffer = 0
            self.marker_end_buffer = 0

            heterodyne_pulsedata = ap.square(wtpts, 0.5, self.origin, self.readout_cfg['width'] + 1000, 10)

            self.waveforms['pxdac4800_2_ch1'][ii], self.waveforms['pxdac4800_2_ch2'][ii] = \
                ap.sideband(wtpts, heterodyne_pulsedata, np.zeros(len(wtpts)), self.readout_cfg['heterodyne_freq'],
                            0)
            ##

    def reshape_data(self, data):
        return np.reshape(data, (self.sequence_length, self.waveform_length))
