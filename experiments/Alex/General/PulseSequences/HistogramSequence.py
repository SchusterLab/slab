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

        self.histo_pts = list(range(seq_len))

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
        # self.set_waveform_length("qubit 1 flux", 1)

    def build_sequence(self):
        PulseSequence.build_sequence(self)

        # waveform dict
        self.waveforms_dict = {}
        self.waveforms_tpts_dict = {}

        for awg in self.awg_info:
            for waveform in awg['waveforms']:
                self.waveforms_dict[waveform['name']] = self.waveforms[waveform['name']]
                self.waveforms_tpts_dict[waveform['name']] = self.get_waveform_times(waveform['name'])

        wtpts = self.get_waveform_times('qubit drive I')
        # mtpts = self.get_marker_times('qubit buffer')

        # todo: why no pulse blaster code here? c.f. vacuum rabi

        awg_trig_len = 100
        start_pulseblaster(self.exp_period_ns, awg_trig_len, self.origin + self.card_delay,
                           self.origin + self.measurement_delay,
                           self.card_trig_width, self.measurement_width)


        for ii, d in enumerate(self.histo_pts):

            # self.markers['readout pulse'][ii] = ap.square(mtpts, 1, self.origin + self.measurement_delay,
            #                                                self.measurement_width)
            # self.markers['card trigger'][ii] = ap.square(mtpts, 1,
            #                                               self.origin - self.card_delay + self.measurement_delay,
            #                                               self.card_trig_width)

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

            if 'qubit drive I' in self.waveforms_dict:
                self.waveforms['qubit drive I'][ii] = np.zeros(len(wtpts))
            if 'qubit drive Q' in self.waveforms_dict:
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
            if 'qubit drive I' in self.waveforms_dict:
                self.waveforms['qubit drive I'][ii] += temp_I
            if 'qubit drive Q' in self.waveforms_dict:
                self.waveforms['qubit drive Q'][ii] += temp_Q

            # ef pulse shifted in freq by alpha
            # consistent with implementation in ef_rabi
            temp_I_ef, temp_Q_ef = ap.sideband(wtpts, pulsedata_ef, np.zeros(len(wtpts)),
                                         self.pulse_cfg['iq_freq'] + self.cfg['qubit']['alpha'], 0)
            if 'qubit drive I' in self.waveforms_dict:
                self.waveforms['qubit drive I'][ii] += temp_I_ef
            if 'qubit drive Q' in self.waveforms_dict:
                self.waveforms['qubit drive Q'][ii] += temp_Q_ef

            # self.markers['qubit buffer'][ii] = ap.square(mtpts, 1,
            #                                              self.origin - self.max_pulse_width - self.marker_start_buffer,
            #                                              self.max_pulse_width + self.marker_start_buffer)

            ## heterodyne pulse
            self.marker_start_buffer = 0
            self.marker_end_buffer = 0


            heterodyne_pulsedata = ap.square(wtpts, self.readout_cfg['heterodyne_a'], self.origin, self.readout_cfg['width'] + 1000, 10)

            temp_h_I, temp_h_Q = \
                ap.sideband(wtpts, heterodyne_pulsedata, np.zeros(len(wtpts)), self.cfg['readout']['heterodyne_freq'],
                            0)

            if 'hetero_ch1' in self.waveforms_dict:
                self.waveforms['hetero_ch1'][ii] = temp_h_I
            if 'hetero_ch2' in self.waveforms_dict:
                self.waveforms['hetero_ch2'][ii] = temp_h_Q
            ##

            ## flux pulses

            # this is to align flux pulse to readout? (diff in 2 pxdac cards)
            hw_delay = self.cfg['flux_pulse_info']['pxdac_hw_delay']

            if self.cfg['flux_pulse_info']['on_during_drive'] and self.cfg['flux_pulse_info']['on_during_readout']:

                flux_width = max(self.max_pulse_width + self.readout_cfg['delay'] + self.readout_cfg['width'] \
                                 - self.cfg['flux_pulse_info']['flux_drive_delay'], 0)
                flux_comp_width = flux_width  # self.cfg['flux_pulse_info']['dc_comp_pulse_length']
                flux_delay = self.cfg['flux_pulse_info']['flux_drive_delay'] + hw_delay
                flux_idle = 100.0
                flux_start = self.origin - self.max_pulse_width + flux_delay
                flux_comp_start = flux_start + flux_width + flux_idle

            elif (self.cfg['flux_pulse_info']['on_during_drive']) and (
            not self.cfg['flux_pulse_info']['on_during_readout']):

                flux_width = max(self.max_pulse_width - self.cfg['flux_pulse_info']['flux_drive_delay'], 0)
                flux_comp_width = flux_width  # self.cfg['flux_pulse_info']['dc_comp_pulse_length']
                flux_delay = self.cfg['flux_pulse_info']['flux_drive_delay'] + hw_delay
                flux_idle = self.readout_cfg['delay'] + self.readout_cfg['width'] + 100
                flux_start = self.origin - self.max_pulse_width + flux_delay
                flux_comp_start = flux_start + flux_width + flux_idle

            elif (not self.cfg['flux_pulse_info']['on_during_drive']) and (self.cfg['flux_pulse_info']['on_during_readout']):

                flux_width = self.readout_cfg['delay'] + self.readout_cfg['width']
                flux_comp_width = flux_width  # self.cfg['flux_pulse_info']['dc_comp_pulse_length']
                flux_delay = hw_delay
                flux_idle = 100.0
                flux_start = self.origin + flux_delay
                flux_comp_start = flux_start + flux_width + flux_idle

            else:
                flux_width = 0
                flux_comp_width = 0
                flux_delay = 0
                flux_idle = 0
                flux_start = self.origin
                flux_comp_start = flux_start

            flux_a = self.cfg['flux_pulse_info']['flux_a']
            flux_freq = self.cfg['flux_pulse_info']['flux_freq']
            # pulse_exponent = self.cfg['flux_pulse_info']['pulse_exponent']
            sigma = self.cfg["pulse_info"]["square_exp"]["ramp_sigma"]

            for jj in range(4):

                # flux_area = flux_width * flux_a[ii]
                flux_comp_a = - flux_a[jj]  # flux_area/float(flux_comp_width)

                f_I = np.zeros(len(self.waveforms_tpts_dict['flux_%d' % (jj + 1)]))
                f_Q = np.zeros(len(self.waveforms_tpts_dict['flux_%d' % (jj + 1)]))

                if flux_width > 0:
                    # self.psb.append('flux_' + str(ii + 1), 'general', 'square', amp=flux_a[ii],
                    #                 length=flux_width, freq=flux_freq[ii],
                    #                 delay=flux_delay)

                    flux_pulsedata = ap.square(self.waveforms_tpts_dict['flux_%d' % (jj + 1)], flux_a[jj], flux_start,
                                               flux_width, sigma)
                    temp_f_I, temp_f_Q = \
                        ap.sideband(self.waveforms_tpts_dict['flux_%d' % (jj + 1)], flux_pulsedata,
                                    np.zeros(len(self.waveforms_tpts_dict['flux_%d' % (jj + 1)])),
                                    flux_freq[jj], 0)
                    f_I += temp_f_I
                    f_Q += temp_f_Q

                if flux_comp_width > 0:
                    # self.psb.append('flux_' + str(ii + 1), 'general', 'square', amp=flux_comp_a,
                    #                 length=flux_comp_width, freq=flux_freq[ii],
                    #                 delay=flux_delay + flux_idle)

                    flux_pulsedata = ap.square(self.waveforms_tpts_dict['flux_%d' % (jj + 1)], flux_comp_a, flux_comp_start,
                                               flux_comp_width, sigma)
                    temp_f_I, temp_f_Q = \
                        ap.sideband(self.waveforms_tpts_dict['flux_%d' % (jj + 1)], flux_pulsedata,
                                    np.zeros(len(self.waveforms_tpts_dict['flux_%d' % (jj + 1)])),
                                    flux_freq[jj], 0)
                    f_I += temp_f_I
                    f_Q += temp_f_Q

                if 'flux_%d' %(jj+1) in self.waveforms_dict:
                    self.waveforms['flux_%d' %(jj+1)][ii] = f_I

    def reshape_data(self, data):
        return np.reshape(data, (self.sequence_length, self.waveform_length))
