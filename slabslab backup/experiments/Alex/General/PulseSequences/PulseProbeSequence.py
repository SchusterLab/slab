__author__ = 'Nelson'


from slab.instruments.awg.PulseSequence import *
from slab.experiments.ExpLib import awgpulses as ap
from numpy import arange, linspace
from slab.instruments.pulseblaster.pulseblaster import start_pulseblaster, run_pulseblaster
#from slab.experiments.ExpLib.TEK1PulseOrganizer import *

class PulseProbeSequence(PulseSequence):
    def __init__(self, awg_info, pulse_probe_cfg, readout_cfg, pulse_cfg, buffer_cfg,cfg):

        self.cfg = cfg
        self.pulse_probe_cfg = pulse_probe_cfg

        self.start_end_buffer = buffer_cfg['tek1_start_end']
        self.marker_start_buffer = buffer_cfg['marker_start']

        PulseSequence.__init__(self, "Pulse Probe", awg_info, sequence_length=1)

        self.exp_period_ns = self.cfg['expt_trigger']['period_ns']
        self.pulse_type = pulse_probe_cfg['pulse_type']
        self.pulse_probe_len = pulse_probe_cfg['pulse_probe_len']
        self.a = pulse_probe_cfg['a']
        self.measurement_delay = readout_cfg['delay']
        self.measurement_width = readout_cfg['width']
        self.card_delay = readout_cfg['card_delay']
        self.card_trig_width = readout_cfg['card_trig_width']

        if self.pulse_type == 'square':
            self.ramp_sigma = pulse_cfg['ramp_sigma']
            max_pulse_width = 6 * self.ramp_sigma + self.pulse_probe_len

        if self.pulse_type == 'gauss':
            max_pulse_width = 6 * self.pulse_probe_len

        self.max_length = round_samples((max_pulse_width + self.measurement_delay + self.measurement_width + 2*self.start_end_buffer))
        self.origin = self.max_length - (self.measurement_delay + self.measurement_width + self.start_end_buffer)

        self.set_all_lengths(self.max_length)
#        self.set_waveform_length("qubit 1 flux", 1)

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
#        mtpts = self.get_marker_times('qubit buffer')

        ii = 0
        # TODO: pulseblaster out of sync bug#

        awg_trig_len = 100
        start_pulseblaster(self.exp_period_ns, awg_trig_len, self.origin+self.card_delay, self.origin + self.measurement_delay,
                           self.card_trig_width, self.measurement_width)
        run_pulseblaster()

        # self.markers['readout pulse'][ii] = ap.square(mtpts, 1, self.origin + self.measurement_delay,
        #                                                self.measurement_width)
        # self.markers['card trigger'][ii] = ap.square(mtpts, 1,
        #                                               self.origin - self.card_delay + self.measurement_delay,
        #                                               self.card_trig_width)

        pulse_probe_len = self.pulse_probe_len
        a = self.a

        if self.pulse_type == 'square':
            if 'qubit drive I' in self.waveforms_dict:
                self.waveforms['qubit drive I'][ii] = ap.sideband(wtpts,
                                                                   ap.square(wtpts, a,
                                                                              self.origin - pulse_probe_len - 3 * self.ramp_sigma,
                                                                              pulse_probe_len, self.ramp_sigma),
                                                                   np.zeros(len(wtpts)),
                                                                  self.pulse_probe_cfg['iq_freq'], 0)[0]
            if 'qubit drive Q' in self.waveforms_dict:
                self.waveforms['qubit drive Q'][ii] = ap.sideband(wtpts,
                                                                   ap.square(wtpts, a,
                                                                              self.origin - pulse_probe_len - 3 * self.ramp_sigma,
                                                                          pulse_probe_len, self.ramp_sigma),
                                                               np.zeros(len(wtpts)),
                                                              self.pulse_probe_cfg['iq_freq'], 0)[1]
            # self.markers['qubit buffer'][ii] = ap.square(mtpts, 1,
            #                                               self.origin - pulse_probe_len - 3 * self.ramp_sigma - self.marker_start_buffer,
            #                                               pulse_probe_len + 4 * self.ramp_sigma + self.marker_start_buffer)
            #
            # high_values_indices = self.markers['qubit buffer'][ii] > 1
            # self.markers['qubit buffer'][ii][high_values_indices] = 1

        if self.pulse_type == 'gauss':
            if 'qubit drive I' in self.waveforms_dict:
                self.waveforms['qubit drive I'][ii] = ap.sideband(wtpts,
                                                               ap.gauss(wtpts, a, self.origin - 3 * pulse_probe_len,
                                                                         pulse_probe_len), np.zeros(len(wtpts)),
                                                              self.pulse_probe_cfg['iq_freq'], 0)[0]
            if 'qubit drive Q' in self.waveforms_dict:
                self.waveforms['qubit drive Q'][ii] = ap.sideband(wtpts,
                                                               ap.gauss(wtpts, a, self.origin - 3 * pulse_probe_len,
                                                                         pulse_probe_len), np.zeros(len(wtpts)),
                                                              self.pulse_probe_cfg['iq_freq'], 0)[1]
            # self.markers['qubit buffer'][ii] = ap.square(mtpts, 1, self.origin - 6 * pulse_probe_len - self.marker_start_buffer,
            #                                                6 * pulse_probe_len + self.marker_start_buffer)
            #
            # high_values_indices = self.markers['qubit buffer'][ii] > 1
            # self.markers['qubit buffer'][ii][high_values_indices] = 1

        ## heterodyne pulse
        self.marker_start_buffer = 0
        self.marker_end_buffer = 0

        heterodyne_pulsedata = ap.square(wtpts, self.cfg['readout']['heterodyne_a'], self.origin, self.cfg['readout']['width']+1000, 10)

        temp_h_I, temp_h_Q = \
            ap.sideband(wtpts, heterodyne_pulsedata, np.zeros(len(wtpts)), self.cfg['readout']['heterodyne_freq'], 0)

        if 'hetero_ch1' in self.waveforms_dict:
            self.waveforms['hetero_ch1'][ii] = temp_h_I
        if 'hetero_ch2' in self.waveforms_dict:
            self.waveforms['hetero_ch2'][ii] = temp_h_Q
        ##

        # np.save('S:\\_Data\\170102 - Tunable Coupler Higher Qubit with TWPA and Isolater\\data\\waveform_I.npy', self.waveforms['qubit drive I'])
        # np.save('S:\\_Data\\170102 - Tunable Coupler Higher Qubit with TWPA and Isolater\\data\\time.npy',wtpts)
        # np.save('S:\\_Data\\170102 - Tunable Coupler Higher Qubit with TWPA and Isolater\\data\\readout.npy', self.markers['readout pulse'])
        # np.save('S:\\_Data\\170102 - Tunable Coupler Higher Qubit with TWPA and Isolater\\data\\card.npy',self.markers['card trigger'])


    def reshape_data(self, data):
        return np.reshape(data, (self.sequence_length, self.waveform_length))