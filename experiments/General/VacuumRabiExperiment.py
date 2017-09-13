__author__ = 'Nelson'

from slab import *
from slab.instruments.Alazar import Alazar
from numpy import mean, arange, sqrt
from slab.experiments.General.PulseSequences.VacuumRabiSequence import *
from tqdm import tqdm
from slab.instruments.RedPitaya.RedPitayaADC import *
from slab.instruments.pulseblaster.pulseblaster import *

class VacuumRabiExperiment(Experiment):
    def __init__(self, path='', liveplot_enabled = False, calibrate_start_phase = False,prefix='Vacuum_Rabi', config_file='..\\config.json', use_cal=False, **kwargs):
        Experiment.__init__(self, path=path, prefix=prefix, config_file=config_file, liveplot_enabled = liveplot_enabled, **kwargs)

        self.expt_cfg_name = prefix.lower()
        self.liveplot_enabled = liveplot_enabled
        self.calibrate_start_phase = calibrate_start_phase
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

        print "Prep Instruments"
        self.readout.set_output(True)
        self.readout.set_power(self.cfg['readout']['power'])
        if (self.cfg[self.expt_cfg_name]['pulsed']):
            self.readout.set_ext_pulse(mod=True)
        else:
            self.readout.set_ext_pulse(mod=False)

        if (self.cfg[self.expt_cfg_name]['pi_pulse']):
            self.drive.set_output(True)
            self.drive.set_ext_pulse(mod=False)
        else:
            self.drive.set_output(False)
            self.drive.set_ext_pulse(mod=False)


        # self.drive.set_output(True)
        # self.drive.set_ext_pulse(mod=False)
        # # self.drive.set_ext_pulse(mod=False)

        try:
            self.readout_atten.set_attenuator(self.cfg['readout']['dig_atten'])
        except:
            print "Error in setting digital attenuator."

        try:
            self.awg.set_amps_offsets(self.cfg['cal']['iq_amps'], self.cfg['cal']['iq_offsets'])
            self.awg.run()
        except:
            print "error in setting self.awg"




        for freq in self.expt_pts:
            self.readout.set_frequency(freq)
            self.readout_shifter.set_phase((self.cfg['readout']['start_phase'] + self.cfg['readout']['phase_slope'] * (freq - self.cfg['readout']['frequency']))%360, freq)
            print self.readout_shifter.get_phase()

            expt_data_ch1 = None
            expt_data_ch2 = None
            expt_data_mag = None

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
                    self.plotter.append_z('scope1',expt_data_ch1)
                    self.plotter.append_z('scope2',expt_data_ch2)
                    self.plotter.append_z('scope_mag',expt_mag)

                with self.datafile() as f:
                    f.append_pt('freq', freq)
                    f.append_pt('ch1_mean', mean(expt_data_ch1[0:]))
                    f.append_pt('ch2_mean', mean(expt_data_ch2[0:]))
                    f.append_pt('mag_mean', mean(expt_mag[0:]))


                print cfg['readout']['start_phase']

    def awg_prep(self):
        stop_pulseblaster()
        LocalInstruments().inst_dict['pxdac4800_1'].stop()
        # LocalInstruments().inst_dict['pxdac4800_2'].stop()

    def awg_run(self):
        LocalInstruments().inst_dict['pxdac4800_1'].run_experiment()
        # LocalInstruments().inst_dict['pxdac4800_2'].run_experiment()
        run_pulseblaster()