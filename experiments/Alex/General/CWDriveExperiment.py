__author__ = 'Nelson'

from slab import *
from slab.instruments.Alazar import Alazar
from numpy import mean, arange
from tqdm import tqdm


class CWDriveExperiment(Experiment):
    def __init__(self, path='', liveplot_enabled = False, prefix='CW_Drive', config_file='..\\config.json', use_cal=False, **kwargs):
        Experiment.__init__(self, path=path, prefix=prefix, config_file=config_file, liveplot_enabled = liveplot_enabled, **kwargs)

        self.expt_cfg_name = prefix.lower()

        self.liveplot_enabled = liveplot_enabled

        self.cfg['alazar']['samplesPerRecord'] = 2 ** (self.cfg['readout']['width'] - 1).bit_length()
        self.cfg['alazar']['recordsPerBuffer'] = 100
        self.cfg['alazar']['recordsPerAcquisition'] = 10000

        self.expt_pts = arange(self.cfg[self.expt_cfg_name]['start'], self.cfg[self.expt_cfg_name]['stop'], self.cfg[self.expt_cfg_name]['step'])

        return

    def go(self):

        if self.liveplot_enabled:
            self.plotter.clear()

        print("Prep Instruments")
        self.readout.set_output(True)
        self.readout.set_power(self.cfg['readout']['power'])
        self.readout.set_ext_pulse(mod=False)
        try:
            self.readout_atten.set_attenuator(self.cfg['readout']['dig_atten'])
        except:
            print("error in setting digital attenuator")
        self.readout.set_frequency(self.cfg['readout']['frequency'])
        self.readout_shifter.set_phase((self.cfg['readout']['start_phase'])%360, self.cfg['readout']['frequency'])

        self.drive.set_output(True)
        self.drive.set_power(self.cfg['drive']['power'])
        self.drive.set_ext_pulse(mod=False)

        try:
            self.awg.set_amps_offsets(self.cfg['cal']['iq_amps'],self.cfg[self.expt_cfg_name]['iq_offsets'])
        except:
            print("error in setting awg")

        print("Prep Card")
        adc = Alazar(self.cfg['alazar'])

        for freq in self.expt_pts:
            self.drive.set_frequency(freq)

            expt_data_ch1 = None
            expt_data_ch2 = None
            expt_data_mag = None
            for ii in tqdm(arange(max(1, self.cfg[self.expt_cfg_name]['averages'] / 100))):
                tpts, ch1_pts, ch2_pts = adc.acquire_avg_data()

                mag = sqrt(ch1_pts ** 2 + ch2_pts ** 2)

                if expt_data_ch1 is None:
                    expt_data_ch1 = ch1_pts
                    expt_data_ch2 = ch2_pts
                else:
                    expt_data_ch1 = (expt_data_ch1 * ii + ch1_pts) / (ii + 1.0)
                    expt_data_ch2 = (expt_data_ch2 * ii + ch2_pts) / (ii + 1.0)
            expt_mag = sqrt(expt_data_ch1 ** 2 + expt_data_ch2 ** 2)

            if self.liveplot_enabled:
                self.plotter.append_xy('readout_avg_freq_scan1', freq, mean(expt_data_ch1[0:]))
                self.plotter.append_xy('readout_avg_freq_scan2', freq, mean(expt_data_ch2[0:]))
                self.plotter.append_xy('readout_avg_freq_scan_mag', freq, mean(expt_mag[0:]))
                self.plotter.append_z('scope1', expt_data_ch1)
                self.plotter.append_z('scope2', expt_data_ch2)
                self.plotter.append_z('scope_mag', expt_mag)

            with self.datafile() as f:
                f.append_pt('freq', freq)
                f.append_pt('ch1_mean', mean(expt_data_ch1[0:]))
                f.append_pt('ch2_mean', mean(expt_data_ch2[0:]))
                f.append_pt('mag_mean', mean(expt_mag[0:]))


            # tpts, ch1_pts, ch2_pts = adc.acquire_avg_data()
            #
            # mag = sqrt(ch1_pts**2+ch2_pts**2)
            # if self.liveplot_enabled:
            #     self.plotter.append_xy('avg_cw_drive_freq_scan1', freq, mean(ch1_pts[0:]))
            #     self.plotter.append_xy('avg_cw_drive_freq_scan2', freq, mean(ch2_pts[0:]))
            #     self.plotter.append_xy('avg_cw_drive_freq_scan_mag', freq, mean(mag[0:]))
            #     self.plotter.append_z('scope1',ch1_pts)
            #     self.plotter.append_z('scope2',ch2_pts)
            #     self.plotter.append_z('scope_mag',mag)
            #
            # with self.datafile() as f:
            #     f.append_pt('freq', freq)
            #     f.append_pt('ch1_mean', mean(ch1_pts[0:]))
            #     f.append_pt('ch2_mean', mean(ch2_pts[0:]))
            #     f.append_pt('mag_mean', mean(mag[0:]))


