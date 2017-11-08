__author__ = 'Nelson'

from slab import *
from slab.instruments.Alazar import Alazar
from numpy import mean, arange


class CWDriveExperiment(Experiment):
    def __init__(self, path='', prefix='CW_Drive', config_file='..\\config.json', use_cal=False, **kwargs):
        Experiment.__init__(self, path=path, prefix=prefix, config_file=config_file, **kwargs)

        self.expt_cfg_name = prefix.lower()

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
        self.readout.set_ext_pulse(mod=False)
        self.readout_atten.set_attenuator(self.cfg['readout']['dig_atten'])
        self.readout.set_frequency(self.cfg['readout']['frequency'])
        self.readout_shifter.set_phase((self.cfg['readout']['start_phase'])%360, self.cfg['readout']['frequency'])

        self.drive.set_output(True)
        self.drive.set_power(self.cfg['drive']['power'])
        self.drive.set_ext_pulse(mod=False)

        self.awg.set_amps_offsets(self.cfg['cal']['iq_amps'],self.cfg[self.expt_cfg_name]['iq_offsets'])

        print "Prep Card"
        adc = Alazar(self.cfg['alazar'])

        for freq in self.expt_pts:
            self.drive.set_frequency(freq)

            tpts, ch1_pts, ch2_pts = adc.acquire_avg_data()

            mag = sqrt(ch1_pts**2+ch2_pts**2)

            self.plotter.append_xy('avg_cw_drive_freq_scan1', freq, mean(ch1_pts[0:]))
            self.plotter.append_xy('avg_cw_drive_freq_scan2', freq, mean(ch2_pts[0:]))
            self.plotter.append_xy('avg_cw_drive_freq_scan_mag', freq, mean(mag[0:]))
            self.plotter.append_z('scope1',ch1_pts)
            self.plotter.append_z('scope2',ch2_pts)
            self.plotter.append_z('scope_mag',mag)

            with self.datafile() as f:
                f.append_pt('freq', freq)
                f.append_pt('ch1_mean', mean(ch1_pts[0:]))
                f.append_pt('ch2_mean', mean(ch2_pts[0:]))
                f.append_pt('mag_mean', mean(mag[0:]))


