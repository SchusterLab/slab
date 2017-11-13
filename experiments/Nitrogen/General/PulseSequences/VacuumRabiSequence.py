__author__ = 'Nelson'


from slab.instruments.awg.PulseSequence import *
from slab.experiments.ExpLib import awgpulses as ap
from numpy import arange, linspace
from slab.experiments.ExpLib.TEK1PulseOrganizer import *
from slab.experiments.ExpLib.PulseSequenceBuilder import *

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
            max_pulse_width = 4 * self.pi_length


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
        print(a)
        print(w)

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
                                 ap.gauss(wtpts, a, self.origin - 2 * gauss_sigma , gauss_sigma),
                                 np.zeros(len(wtpts)),
                                 self.pulse_cfg['iq_freq'], 0)
                self.markers['qubit buffer'][ii] = ap.square(mtpts, 1, self.origin - 4 * gauss_sigma - self.marker_start_buffer ,
                                                              4 * gauss_sigma + self.marker_start_buffer)

        self.markers['readout pulse'][ii] = ap.square(mtpts, 1, self.origin + self.measurement_delay,
                                                       self.measurement_width)
        self.markers['card trigger'][ii] = ap.square(mtpts, 1,
                                                      self.origin - self.card_delay + self.measurement_delay,
                                                      self.card_trig_width)

        # print "Saving vacuum Rabi pulse..."
        np.save('pi_pulse_vacuum_rabi',self.waveforms['qubit drive I'])
        np.save('pi_pulse_vacuum_rabi_time',wtpts)
        np.save('readout_pulse',self.markers['readout pulse'])
        np.save('readout_pulse_time',mtpts)

        # print np.shape(self.waveforms['qubit drive I'])

    def reshape_data(self, data):
        return np.reshape(data, (self.sequence_length, self.waveform_length))


class MultimodeVacuumRabiSequence(PulseSequence):
    def __init__(self, awg_info, vacuum_rabi_cfg, readout_cfg,buffer_cfg,pulse_cfg,cfg):

        self.vacuum_rabi_cfg = vacuum_rabi_cfg
        self.multimode_cfg = cfg['multimodes']
        self.flux_pulse_cfg = cfg['flux_pulse_info']
        self.spacing = cfg['flux_pulse_info']['spacing']

        # print self.multimode_cfg[0]['flux_pulse_freq']
        self.start_end_buffer = buffer_cfg['tek1_start_end']
        self.marker_start_buffer = buffer_cfg['marker_start']
        self.tek2_trigger_delay = buffer_cfg['tek2_trigger_delay']

        PulseSequence.__init__(self, "Multimode Vacuum Rabi", awg_info, sequence_length=1)

        self.pulse_type = vacuum_rabi_cfg['pulse_type']
        self.flux_pulse_type = vacuum_rabi_cfg['flux_pulse_type']
        self.pi_pulse = vacuum_rabi_cfg['pi_pulse']
        self.load_photon = vacuum_rabi_cfg['load_photon']
        self.mode = vacuum_rabi_cfg['mode']
        self.measurement_delay = readout_cfg['delay']
        self.measurement_width = readout_cfg['width']
        self.card_delay = readout_cfg['card_delay']
        self.card_trig_width = readout_cfg['card_trig_width']
        self.pulse_cfg = pulse_cfg[self.pulse_type]

        self.psb = PulseSequenceBuilder(cfg)
        self.pulse_sequence_matrix = []
        total_pulse_span_length_list = []
        self.total_flux_pulse_span_length_list = []


        if self.pi_pulse:
            self.pi_length = pulse_cfg[self.pulse_type]['pi_length']
        else:
            self.pi_length = 0

        if self.pulse_type == 'square':
            self.ramp_sigma = pulse_cfg[self.pulse_type]['ramp_sigma']
            max_pulse_width = self.pi_length + 4 * self.ramp_sigma

        if self.pulse_type == 'gauss':
            max_pulse_width = 4 * self.pi_length


        self.max_length = round_samples((max_pulse_width + self.measurement_delay + self.measurement_width + 2*self.start_end_buffer+self.card_delay+self.card_trig_width))
        print(self.max_length)
        self.origin = self.max_length - (self.measurement_delay + self.measurement_width + self.start_end_buffer)

        if self.load_photon:
            self.flux_length = self.multimode_cfg[self.mode]['flux_pi_length'] + 4*self.multimode_cfg[self.mode]['ramp_sigma'] + self.spacing
            self.ramp_sigma_flux = self.multimode_cfg[self.mode]['ramp_sigma']
            self.a_flux = self.multimode_cfg[self.mode]['a']
            self.flux_pi_length = self.multimode_cfg[self.mode]['flux_pi_length']
            self.max_length = self.max_length + self.flux_length
            self.flux_pulse_freq = self.multimode_cfg[self.mode]['flux_pulse_freq']

        print(self.max_length)

        self.set_all_lengths(self.max_length)
        self.set_waveform_length("qubit 1 flux", self.flux_length)

    def build_sequence(self):
        PulseSequence.build_sequence(self)
        wtpts = self.get_waveform_times('qubit drive I')
        mtpts = self.get_marker_times('qubit buffer')
        ftpts = self.get_waveform_times('qubit 1 flux')
        print(len(ftpts))

        print("max = " +str(np.max(ftpts)))
        print("min = " +str(np.min(ftpts)))

        ii = 0
        w =self.pi_length
        a=self.pulse_cfg['a']
        if self.load_photon:
            start_pi = -self.flux_length
        else:
            start_pi = 0

        print(a)
        print(w)

        if self.pi_pulse:
            if self.pulse_type == 'square':
                    self.waveforms['qubit drive I'][ii], self.waveforms['qubit drive Q'][ii] = \
                        ap.sideband(wtpts,
                                     ap.square(wtpts, a, self.origin - start_pi - 2 * self.ramp_sigma , w,
                                                self.ramp_sigma), np.zeros(len(wtpts)),
                                     self.pulse_cfg['iq_freq'], 0)

                    self.markers['qubit buffer'][ii] = ap.square(mtpts, 1,
                                                                  self.origin - w - 4 * self.ramp_sigma - self.marker_start_buffer,
                                                                  w + 4 * self.ramp_sigma + self.marker_start_buffer)

            if self.pulse_type == 'gauss':
                gauss_sigma = w  #covention for the width of gauss pulse
                self.waveforms['qubit drive I'][ii], self.waveforms['qubit drive Q'][ii] = \
                    ap.sideband(wtpts,
                                 ap.gauss(wtpts, a, self.origin + start_pi - 2 * gauss_sigma , gauss_sigma),
                                 np.zeros(len(wtpts)),
                                 self.pulse_cfg['iq_freq'], 0)
                self.markers['qubit buffer'][ii] = ap.square(mtpts, 1, self.origin + start_pi - 4 * gauss_sigma - self.marker_start_buffer ,
                                                              4 * gauss_sigma + self.marker_start_buffer)


        if self.load_photon:

            print("Amp = " + str(self.a_flux))
            # self.waveforms['qubit 1 flux'] = ap.sideband(ftpts,0.5, np.zeros(len(ftpts)),
            #                           self.flux_pulse_freq, 0,offset=False)[0]
            self.waveforms['qubit 1 flux'] = ap.sideband(ftpts,ap.square(ftpts, self.a_flux, 0, self.ramp_sigma_flux), np.zeros(len(ftpts)),
                                      self.flux_pulse_freq, 0,offset=False)[0]
            np.save("fpulse",self.waveforms['qubit 1 flux'])
            self.waveforms['qubit 1 flux'] = np.reshape(self.waveforms['qubit 1 flux'],(1,np.shape(self.waveforms['qubit 1 flux'])[0]))
            self.markers_ch3m1 = ap.square(mtpts, 1,
                                                self.origin - self.flux_length - self.tek2_trigger_delay,
                                                self.card_trig_width)



        self.markers['readout pulse'][ii] = ap.square(mtpts, 1, self.origin + self.measurement_delay,
                                                       self.measurement_width)
        self.markers['card trigger'][ii] = ap.square(mtpts, 1,
                                                      self.origin - self.card_delay + self.measurement_delay,
                                                      self.card_trig_width)
        print(self.a_flux)
        np.save("qtime",wtpts)
        np.save("qpulse",self.waveforms['qubit drive I'])
        np.save("ftime",ftpts)
        np.save("fpulse2",self.waveforms['qubit 1 flux'])
        np.save("rtime",mtpts)
        np.save("rpulse",self.markers['readout pulse'])
        # print np.shape(self.waveforms['qubit drive I'])

    def reshape_data(self, data):
        return np.reshape(data, (self.sequence_length, self.waveform_length))


