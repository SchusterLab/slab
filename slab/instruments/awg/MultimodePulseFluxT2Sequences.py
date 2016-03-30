__author__ = 'Nelson'

from slab.instruments.awg.PulseSequence import *
from slab.instruments.awg import awgpulses2 as ap2
from numpy import arange, linspace

from liveplot import LivePlotClient

#lp=LivePlotClient()


def round_samples(x, base=64):
    return int(base * round(float(x)/base))

class MultimodeFluxSideBandT2Sequence(PulseSequence):
    def __init__(self, awg_info, mm_flux_sideband_cfg, readout_cfg, pulse_cfg):
        """
        rabi_cfg = {rabi_pts, sweep_type="time", pulse_type="square", a=0, w=0, sigma=0, freq=0, phase=0}
        readout_cfg = {"Q": 10000, "delay": , "width": , "card_delay": , "card_trig_width":  }
        """

        self.mm_flux_sideband_cfg=mm_flux_sideband_cfg
        # if self.mm_flux_sideband_cfg['step'] is not None:
        #     self.mm_flux_sideband_freq_pts = arange(self.mm_flux_sideband_cfg['start_freq'], self.mm_flux_sideband_cfg['stop_freq'], self.mm_flux_sideband_cfg['freq_step'])
        # else:
        #     self.mm_flux_sideband_freq_pts = linspace(self.mm_flux_sideband_cfg['start_freq'], self.mm_flux_sideband_cfg['stop_freq'], self.mm_flux_sideband_cfg['freq_num_pts'])


        self.mm_flux_sideband_pts = arange(self.mm_flux_sideband_cfg['start_length'], self.mm_flux_sideband_cfg['stop_length'], self.mm_flux_sideband_cfg['length_step'])

        sequence_length = len(self.mm_flux_sideband_pts)

        PulseSequence.__init__(self, "Multimode Sideband Rabi", awg_info, sequence_length)

        self.pulse_type = mm_flux_sideband_cfg['pulse_type']
        self.flux_freq = mm_flux_sideband_cfg['flux_freq']
        self.dc_offset_freq = mm_flux_sideband_cfg['dc_offset_freq']
        self.flux_a = mm_flux_sideband_cfg['flux_a']
        self.flux_pi_length = mm_flux_sideband_cfg['flux_pi_length']
        self.a = pulse_cfg['a']
        self.half_pi_length = pulse_cfg['half_pi_length']
        self.freq = pulse_cfg['freq']
        self.phase = pulse_cfg['phase']
        self.measurement_delay = readout_cfg['delay']
        self.measurement_width = readout_cfg['width']
        self.card_delay = readout_cfg['card_delay']
        self.monitor_pulses = readout_cfg['monitor_pulses']
        self.card_trig_width = readout_cfg['card_trig_width']



        max_flux_pulse_width = max(self.mm_flux_sideband_pts) +2*self.flux_pi_length+ 60



        if self.pulse_type == 'square':
            self.ramp_sigma = pulse_cfg['ramp_sigma']
            max_pulse_width = self.pi_length + 4 * self.ramp_sigma

        if self.pulse_type == 'gauss':
            max_pulse_width = 6*self.half_pi_length

        ## hard coding & not really sure where this 1.3us came from
        self.tek2_delay_time = 1300

        self.max_length = round_samples((max_pulse_width + max_flux_pulse_width + self.tek2_delay_time + self.measurement_delay + self.measurement_width +1000))
        self.origin = self.max_length - (self.measurement_delay + self.measurement_width + 500)

        self.set_all_lengths(self.max_length)
        self.set_waveform_length("qubit 1 flux", max_flux_pulse_width)

    def build_sequence(self):
        PulseSequence.build_sequence(self)
        wtpts = self.get_waveform_times('qubit drive I')
        mtpts = self.get_marker_times('qubit buffer')
        ftpts = self.get_waveform_times('qubit 1 flux')

        print len(wtpts)
        print len(ftpts)

        for ii, d in enumerate(self.mm_flux_sideband_pts):
            self.delay = d
            self.markers['readout pulse'][ii] = ap2.square(mtpts, 1, self.origin+self.measurement_delay,
                                                           self.measurement_width)
            self.markers['card trigger'][ii] = ap2.square(mtpts, 1,
                                                          self.origin - self.card_delay + self.measurement_delay,
                                                          self.card_trig_width)

            pulse_delay = self.delay + 2* self.flux_pi_length +6*self.half_pi_length + 100

            self.markers['ch3m1'][ii] = ap2.square(mtpts, 1,self.origin - pulse_delay - self.tek2_delay_time,
                                                          100)
            w=self.half_pi_length
            a=self.a

            if self.pulse_type == 'square':
                self.waveforms['qubit drive I'][ii], self.waveforms['qubit drive Q'][ii] = \
                    ap2.sideband(wtpts,
                                 ap2.square(wtpts, a,self.origin - w - 2 * self.ramp_sigma - pulse_delay, w, self.ramp_sigma), np.zeros(len(wtpts)),
                                  self.freq, self.phase)

                self.markers['qubit buffer'][ii] = ap2.square(mtpts, 1, self.origin - w - 4 * self.ramp_sigma - pulse_delay - 100,
                                                          w + 4 * self.ramp_sigma + 200)

            if self.pulse_type == 'gauss':
                gauss_sigma=w #covention for the width of gauss pulse
                self.waveforms['qubit drive I'][ii], self.waveforms['qubit drive Q'][ii] = \
                    ap2.sideband(wtpts,
                                 ap2.gauss(wtpts, a,self.origin - 3* gauss_sigma - pulse_delay, gauss_sigma), np.zeros(len(wtpts)),
                                  self.freq, self.phase)

                self.waveforms['qubit drive I'][ii] +=ap2.sideband(wtpts,
                                 ap2.gauss(wtpts, a,self.origin - 3* gauss_sigma, gauss_sigma), np.zeros(len(wtpts)),
                                  self.freq, self.phase)[0]
                self.waveforms['qubit drive Q'][ii] += ap2.sideband(wtpts,
                                 ap2.gauss(wtpts, a,self.origin - 3* gauss_sigma, gauss_sigma), np.zeros(len(wtpts)),
                                  self.freq, self.phase)[1]

                self.markers['qubit buffer'][ii] = ap2.square(mtpts, 1, self.origin - 6*gauss_sigma - 100 - pulse_delay ,
                                                          6*gauss_sigma + 200)
                self.markers['qubit buffer'][ii] += ap2.square(mtpts, 1, self.origin - 6*gauss_sigma - 100,
                                                          6*gauss_sigma + 200)


            flux_a = self.flux_a
            flux_freq = self.flux_freq
            flux_pi_length = self.flux_pi_length
            dc_offset_freq = self.dc_offset_freq

            if True:
                self.waveforms['qubit 1 flux'][ii] = ap2.sideband(ftpts,
                                     ap2.square(ftpts, flux_a, 15 , flux_pi_length,5), np.zeros(len(ftpts)),
                                      flux_freq, self.phase)[0]
                self.waveforms['qubit 1 flux'][ii] += ap2.sideband(ftpts,
                                     ap2.square(ftpts, flux_a, 3*15+flux_pi_length +self.delay , flux_pi_length,5), np.zeros(len(ftpts)),
                                      flux_freq, self.phase + 360*(dc_offset_freq)*(self.delay+flux_pi_length+2*15)/1e9)[1]
            if False:
                flux_gauss = self.flux_w/4
                self.waveforms['qubit 1 flux'][ii] = ap2.sideband(ftpts,
                                 ap2.gauss(ftpts, flux_a,2* flux_gauss , flux_gauss), np.zeros(len(ftpts)),
                                  flux_freq, self.phase)[0]


    def reshape_data(self, data):
        return np.reshape(data, (self.sequence_length, self.waveform_length))