__author__ = 'Nelson'

from slab import *
from slab.instruments.Alazar import Alazar
from numpy import mean, arange, sqrt
from slab.experiments.General.PulseSequences.VacuumRabiSequence import *
from tqdm import tqdm

class VacuumRabiExperiment(Experiment):
    def __init__(self, path='', liveplot_enabled = False, prefix='Vacuum_Rabi', config_file='..\\config.json', use_cal=False, **kwargs):
        Experiment.__init__(self, path=path, prefix=prefix, config_file=config_file, liveplot_enabled = liveplot_enabled, **kwargs)

        self.expt_cfg_name = prefix.lower()
        self.pulse_type = self.cfg[self.expt_cfg_name]['pulse_type']

        self.liveplot_enabled = liveplot_enabled

        self.pulse_sequence = VacuumRabiSequence(self.cfg['awgs'], self.cfg[self.expt_cfg_name], self.cfg['readout'], self.cfg['buffer'], self.cfg['pulse_info'],self.cfg)
        self.pulse_sequence.build_sequence()
        self.pulse_sequence.write_sequence(os.path.join(self.path, '../sequences/'), prefix, upload=True)

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
        if (self.cfg[self.expt_cfg_name]['pulsed']):
            self.readout.set_ext_pulse(mod=True)
        else:
            self.readout.set_ext_pulse(mod=False)

        try:
            self.drive.set_frequency(
                self.cfg['qubit']['frequency'] - self.cfg['pulse_info'][self.pulse_type]['iq_freq'])
            self.drive.set_power(self.cfg['drive']['power'])
            self.drive.set_ext_pulse(mod=self.cfg['drive']['mod'])

            if (self.cfg[self.expt_cfg_name]['pi_pulse']):
                self.drive.set_output(True)
            else:
                self.drive.set_output(False)
            print("Drive set successfully..")

        except:
            print("No drive found")

        self.drive.set_ext_pulse(mod=False)

        try:
            self.readout_atten.set_attenuator(self.cfg['readout']['dig_atten'])
        except:
            print("Error in setting digital attenuator.")

        try:
            self.awg.set_amps_offsets(self.cfg['cal']['iq_amps'], self.cfg['cal']['iq_offsets'])
            self.awg.run()
        except:
            print("error in setting self.awg")

        print("Prep Card")
        adc = Alazar(self.cfg['alazar'])


        for freq in self.expt_pts:
            self.readout.set_frequency(freq-self.cfg['readout']['heterodyne_freq'])
            self.readout_shifter.set_phase((self.cfg['readout']['start_phase'] + self.cfg['readout']['phase_slope'] * (freq - self.cfg['readout']['frequency']))%360, freq)
            # print self.readout_shifter.get_phase()

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


            # todo: fix heterodyne - need to take cos/sin of one channel
            # todo: & excise window

            if self.cfg['readout']['heterodyne_freq'] == 0:
                # homodyne
                mean_ch1 = mean(expt_data_ch1)
                mean_ch2 = mean(expt_data_ch2)
                mean_mag = mean(expt_mag)
            else:
                # heterodyne
                heterodyne_freq = self.cfg['readout']['heterodyne_freq']
                # ifft by numpy default has the correct 1/N normalization
                fft_ch1 = np.abs(np.fft.ifft(expt_data_ch1))
                fft_ch2 = np.abs(np.fft.ifft(expt_data_ch2))
                fft_mag = np.abs(np.fft.ifft(expt_mag))

                hetero_f_ind = int(round(heterodyne_freq * tpts.size * 1e-9))  # position in ifft

                # todo: single freq v.s. finite freq window (latter: more robust but noisier and with distortion)
                # expt_avg_data = np.average(expt_data_fft_amp[:, (hetero_f_ind - 1):(hetero_f_ind + 1)], axis=1)
                mean_ch1 = fft_ch1[hetero_f_ind]
                mean_ch2 = fft_ch2[hetero_f_ind]
                mean_mag = fft_mag[hetero_f_ind]

            if self.liveplot_enabled:
                self.plotter.append_xy('readout_avg_freq_scan1', freq, mean_ch1)
                self.plotter.append_xy('readout_avg_freq_scan2', freq, mean_ch2)
                self.plotter.append_xy('readout_avg_freq_scan_mag', freq, mean_mag)
                self.plotter.append_z('scope1',expt_data_ch1)
                self.plotter.append_z('scope2',expt_data_ch2)
                self.plotter.append_z('scope_mag',expt_mag)

            with self.datafile() as f:
                f.append_pt('freq', freq)
                f.append_pt('ch1_mean', mean_ch1)
                f.append_pt('ch2_mean', mean_ch2)
                f.append_pt('mag_mean', mean_mag)
                f.append_line('expt_2d_ch1', expt_data_ch1)
                f.append_line('expt_2d_ch2', expt_data_ch2)



