__author__ = 'Nelson'


from slab.instruments.awg.PulseSequence import *
from slab.experiments.ExpLib import awgpulses as ap
from numpy import arange, linspace
#from slab.experiments.ExpLib.TEK1PulseOrganizer import *

class VacuumRabiSequence(PulseSequence):
    def __init__(self, awg_info, vacuum_rabi_cfg, readout_cfg,buffer_cfg,pulse_cfg):

        self.vacuum_rabi_cfg = vacuum_rabi_cfg

        self.start_end_buffer = buffer_cfg['tek1_start_end']
        self.marker_start_buffer = buffer_cfg['marker_start']

        PulseSequence.__init__(self, "Vacuum Rabi", awg_info, sequence_length=1)

        self.pulse_type = vacuum_rabi_cfg['pulse_type']
        self.pi_pulse = vacuum_rabi_cfg['pi_pulse']
        self.a = vacuum_rabi_cfg['a']
        self.measurement_delay = readout_cfg['delay']
        self.measurement_width = readout_cfg['width']
        self.card_delay = readout_cfg['card_delay']
        self.card_trig_width = readout_cfg['card_trig_width']
        self.pulse_cfg = pulse_cfg[self.pulse_type]

        if self.pi_pulse:
            self.pi_length = pulse_cfg[self.pulse_type]['pi_length']
        else:
            self.pi_length = 0

        if self.pulse_type == 'square':
            self.ramp_sigma = pulse_cfg[self.pulse_type]['ramp_sigma']
            max_pulse_width = self.pi_length + 4 * self.ramp_sigma

        if self.pulse_type == 'gauss':
            max_pulse_width = 6 * self.pi_length


        self.max_length = round_samples((max_pulse_width + self.measurement_delay + self.measurement_width + 2*self.start_end_buffer+self.card_delay+self.card_trig_width))
        self.origin = self.max_length - (self.measurement_delay + self.measurement_width + self.start_end_buffer)

        self.set_all_lengths(self.max_length)
        self.set_waveform_length("qubit 1 flux", 1)

    def build_sequence(self):
        PulseSequence.build_sequence(self)

        wtpts = self.get_waveform_times('qubit drive I')
        mtpts = self.get_marker_times('qubit buffer')

        ii = 0
        w =self.pi_length
        a=self.pulse_cfg['a']
        if self.pi_pulse:
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

        self.markers['readout pulse'][ii] = ap.square(mtpts, 1, self.origin + self.measurement_delay,
                                                       self.measurement_width)
        self.markers['card trigger'][ii] = ap.square(mtpts, 1,
                                                      self.origin - self.card_delay + self.measurement_delay,
                                                      self.card_trig_width)



    def reshape_data(self, data):
        return np.reshape(data, (self.sequence_length, self.waveform_length))