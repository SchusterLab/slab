__author__ = 'Nelson'

from slab import *
from slab.instruments.Alazar import Alazar
from numpy import mean, arange, sqrt
from slab.experiments.General.PulseSequences.VacuumRabiSequence import *
from slab.experiments.General.PulseSequences.VacuumRabiSequencePSB import *

class VacuumRabiExperiment(Experiment):
    def __init__(self, path='', prefix='Vacuum_Rabi', config_file='..\\config.json', use_cal=False, **kwargs):
        Experiment.__init__(self, path=path, prefix=prefix, config_file=config_file, **kwargs)

        self.expt_cfg_name = prefix.lower()

        self.pulse_sequence = VacuumRabiSequence(self.cfg['awgs'], self.cfg[self.expt_cfg_name], self.cfg['readout'], self.cfg['buffer'], self.cfg['pulse_info'])
        self.pulse_sequence.build_sequence()
        self.pulse_sequence.write_sequence(os.path.join(self.path, '../sequences/'), prefix, upload=True)

        self.cfg['alazar']['samplesPerRecord'] = 2 ** (self.cfg['readout']['width'] - 1).bit_length()
        self.cfg['alazar']['recordsPerBuffer'] = 100
        self.cfg['alazar']['recordsPerAcquisition'] = 10000

        self.expt_pts = arange(self.cfg[self.expt_cfg_name]['start'], self.cfg[self.expt_cfg_name]['stop'], self.cfg[self.expt_cfg_name]['step'])

        return

    def go(self):
        # self.plotter.clear()

        print "Prep Instruments"
        self.readout.set_output(True)
        self.readout.set_power(self.cfg['readout']['power'])
        if (self.cfg[self.expt_cfg_name]['pulsed']):
            self.readout.set_ext_pulse(mod=True)
        else:
            self.readout.set_ext_pulse(mod=False)

        if (self.cfg[self.expt_cfg_name]['pi_pulse']):
            self.drive.set_output(True)
            self.drive.set_ext_pulse(mod=True)
        else:
            self.drive.set_output(False)
            self.drive.set_ext_pulse(mod=False)


        # self.drive.set_ext_pulse(mod=False)
        self.readout_atten.set_attenuator(self.cfg['readout']['dig_atten'])

        self.awg.set_amps_offsets(self.cfg['cal']['iq_amps'], self.cfg['cal']['iq_offsets'])

        self.awg.run()

        print "Prep Card"
        adc = Alazar(self.cfg['alazar'])


        for freq in self.expt_pts:
            self.readout.set_frequency(freq)
            self.readout_shifter.set_phase((self.cfg['readout']['start_phase'] + self.cfg['readout']['phase_slope'] * (freq - self.cfg['readout']['frequency']))%360, freq)
            # print self.readout_shifter.get_phase()
            tpts, ch1_pts, ch2_pts = adc.acquire_avg_data()

            mag = sqrt(ch1_pts**2+ch2_pts**2)


            # self.plotter.append_xy('readout_avg_freq_scan1', freq, mean(ch1_pts[0:]))
            # self.plotter.append_xy('readout_avg_freq_scan2', freq, mean(ch2_pts[0:]))
            # self.plotter.append_xy('readout_avg_freq_scan_mag', freq, mean(mag[0:]))
            # self.plotter.append_z('scope1',ch1_pts)
            # self.plotter.append_z('scope2',ch2_pts)
            # self.plotter.append_z('scope_mag',mag)

            with self.datafile() as f:
                f.append_pt('freq', freq)
                f.append_pt('ch1_mean', mean(ch1_pts[0:]))
                f.append_pt('ch2_mean', mean(ch2_pts[0:]))
                f.append_pt('mag_mean', mean(mag[0:]))


class MultimodeVacuumRabiExperiment(Experiment):
    def __init__(self, path='', prefix='Multimode_Vacuum_Rabi', config_file='..\\config.json', use_cal=False, **kwargs):
        Experiment.__init__(self, path=path, prefix=prefix, config_file=config_file, **kwargs)

        self.expt_cfg_name = prefix.lower()



        self.pulse_sequence = MultimodeVacuumRabiSequencePSB(self.cfg['awgs'], self.cfg[self.expt_cfg_name], self.cfg['readout'], self.cfg['buffer'], self.cfg['pulse_info'],self.cfg)
        self.pulse_sequence.build_sequence()
        self.pulse_sequence.write_sequence(os.path.join(self.path, '../sequences/'), prefix, upload=True)

        self.cfg['alazar']['samplesPerRecord'] = 2 ** (self.cfg['readout']['width'] - 1).bit_length()
        self.cfg['alazar']['recordsPerBuffer'] = 100
        self.cfg['alazar']['recordsPerAcquisition'] = 10000

        self.expt_pts = arange(self.cfg[self.expt_cfg_name]['start'], self.cfg[self.expt_cfg_name]['stop'], self.cfg[self.expt_cfg_name]['step'])

        return

    def go(self):
        self.plotter.clear()

        print "Prep Instruments"
        self.readout.set_output(True)
        self.readout.set_power(self.cfg['readout']['power'])
        if (self.cfg[self.expt_cfg_name]['pulsed']):
            self.readout.set_ext_pulse(mod=True)
        else:
            self.readout.set_ext_pulse(mod=False)

        if (self.cfg[self.expt_cfg_name]['pi_pulse']):
            self.drive.set_output(True)
            self.drive.set_ext_pulse(mod=True)
        else:
            self.drive.set_output(False)
            self.drive.set_ext_pulse(mod=False)


        # self.drive.set_ext_pulse(mod=False)
        self.readout_atten.set_attenuator(self.cfg['readout']['dig_atten'])

        self.awg.set_amps_offsets(self.cfg['cal']['iq_amps'], self.cfg['cal']['iq_offsets'])

        self.awg.run()

        print "Prep Card"
        adc = Alazar(self.cfg['alazar'])


        for freq in self.expt_pts:
            self.readout.set_frequency(freq)
            self.readout_shifter.set_phase((self.cfg['readout']['start_phase'] + self.cfg['readout']['phase_slope'] * (freq - self.cfg['readout']['frequency']))%360, freq)
            # print self.readout_shifter.get_phase()
            tpts, ch1_pts, ch2_pts = adc.acquire_avg_data()

            mag = sqrt(ch1_pts**2+ch2_pts**2)

            self.plotter.append_xy('readout_avg_freq_scan1', freq, mean(ch1_pts[0:]))
            self.plotter.append_xy('readout_avg_freq_scan2', freq, mean(ch2_pts[0:]))
            self.plotter.append_xy('readout_avg_freq_scan_mag', freq, mean(mag[0:]))
            self.plotter.append_z('scope1',ch1_pts)
            self.plotter.append_z('scope2',ch2_pts)
            self.plotter.append_z('scope_mag',mag)

            with self.datafile() as f:
                f.append_pt('freq', freq)
                f.append_pt('ch1_mean', mean(ch1_pts[0:]))
                f.append_pt('ch2_mean', mean(ch2_pts[0:]))
                f.append_pt('mag_mean', mean(mag[0:]))
