__author__ = 'Nelson'

from slab import *
from slab.instruments.Alazar import Alazar
from slab.experiments.General.PulseSequences.PulseProbeSequence import *
from numpy import mean, arange
from tqdm import tqdm
from slab.instruments.RedPitaya.RedPitayaADC import *
from slab.instruments.pulseblaster.pulseblaster import *

class PulseProbeExperiment(Experiment):
    def __init__(self, path='', liveplot_enabled = False, prefix='Pulse_Probe', config_file='..\\config.json', use_cal=False, **kwargs):
        Experiment.__init__(self, path=path, prefix=prefix, config_file=config_file, liveplot_enabled = liveplot_enabled, **kwargs)

        self.expt_cfg_name = prefix.lower()

        self.liveplot_enabled = liveplot_enabled

        self.pulse_type = self.cfg[self.expt_cfg_name]['pulse_type']

        self.expt_pts = arange(self.cfg[self.expt_cfg_name]['start'], self.cfg[self.expt_cfg_name]['stop'], self.cfg[self.expt_cfg_name]['step'])

        self.pulse_sequence = PulseProbeSequence(self.cfg['awgs'], self.cfg[self.expt_cfg_name], self.cfg['readout'],self.cfg['pulse_info'][self.pulse_type], self.cfg['buffer'],self.cfg)
        self.pulse_sequence.build_sequence()
        self.pulse_sequence.write_sequence(os.path.join(self.path, '../sequences/'), prefix, upload=True)

        self.cfg['alazar']['samplesPerRecord'] = 2 ** (self.cfg['readout']['width'] - 1).bit_length()
        self.cfg['alazar']['recordsPerBuffer'] = 100
        self.cfg['alazar']['recordsPerAcquisition'] = 10000

        return

    def go(self):
        if self.liveplot_enabled:
            self.plotter.clear()

        # self.save_config()

        print "Prep Instruments"
        self.readout.set_frequency(self.cfg['readout']['frequency'])
        self.readout.set_power(self.cfg['readout']['power'])
        self.readout.set_ext_pulse(mod=self.cfg['readout']['mod'])
        self.readout.set_output(True)
        # self.readout_shifter.set_phase(self.cfg['readout']['start_phase']%360, self.cfg['readout']['frequency'])


        self.drive.set_power(self.cfg['drive']['power'])
        self.drive.set_ext_pulse(mod=self.cfg['drive']['mod'])
        self.drive.set_output(True)
        try:
            self.readout_atten.set_attenuator(self.cfg['readout']['dig_atten'])
        except:
            print("error in setting digital attenuator")

        try:
            self.awg.set_amps_offsets(self.cfg['cal']['iq_amps'], self.cfg['cal']['iq_offsets'])
            self.awg.run()
        except:
            print("error in setting awg")




        for freq in self.expt_pts:
            self.drive.set_frequency(freq-self.cfg[self.expt_cfg_name]['iq_freq'])

            if self.cfg['readout']['adc'] == 'redpitaya':
                num_experiments = 1
                ch1_pts, ch2_pts = setup_redpitaya_adc(num_experiments=num_experiments,
                                                       window=self.cfg['readout']['window'],
                                                       shots=self.cfg[self.expt_cfg_name][
                                                           'averages'],
                                                       plot_data=False,
                                                       start_function=self.awg_run,
                                                       stop_function=self.awg_prep)

                with self.datafile() as f:
                    f.append_pt('freq', freq)
                    f.append_pt('ch1_mean', ch1_pts)
                    f.append_pt('ch2_mean', ch2_pts)
            else:
                print "Prep Card"
                adc = Alazar(self.cfg['alazar'])
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
            #
            #
            # mag = sqrt(ch1_pts**2+ch2_pts**2)
            # if self.liveplot_enabled:
            #     self.plotter.append_xy('avg_pulse_probe_freq_scan1', freq, mean(ch1_pts[0:]))
            #     self.plotter.append_xy('avg_pulse_probe_freq_scan2', freq, mean(ch2_pts[0:]))
            #     self.plotter.append_xy('avg_pulse_probe_freq_mag', freq, mean(mag[0:]))
            #     self.plotter.append_z('scope1',ch1_pts)
            #     self.plotter.append_z('scope2',ch2_pts)
            #     self.plotter.append_z('scope_mag',mag)
            #
            # with self.datafile() as f:
            #     f.append_pt('freq', freq)
            #     f.append_pt('ch1_mean', mean(ch1_pts[0:]))
            #     f.append_pt('ch2_mean', mean(ch2_pts[0:]))
            #     f.append_pt('mag_mean', mean(mag[0:]))

    def awg_prep(self):
        stop_pulseblaster()
        LocalInstruments().inst_dict['pxdac4800_1'].stop()
        LocalInstruments().inst_dict['pxdac4800_2'].stop()

    def awg_run(self):
        LocalInstruments().inst_dict['pxdac4800_1'].run_experiment()
        LocalInstruments().inst_dict['pxdac4800_2'].run_experiment()
        run_pulseblaster()