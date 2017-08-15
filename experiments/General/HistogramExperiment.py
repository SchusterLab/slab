__author__ = 'Nelson'

from slab import *
from slab.instruments.Alazar import Alazar
from slab.experiments.General.PulseSequences.HistogramSequence import HistogramSequence
from numpy import *


class HistogramExperiment(Experiment):
    def __init__(self, path='', prefix='Histogram', config_file='..\\config.json', use_cal=False, **kwargs):
        Experiment.__init__(self, path=path, prefix=prefix, config_file=config_file, **kwargs)

        self.expt_cfg_name = prefix.lower()

        self.pulse_type = self.cfg[self.expt_cfg_name]['pulse_type']

        self.pulse_sequence = HistogramSequence(self.cfg['awgs'], self.cfg[self.expt_cfg_name], self.cfg['readout'],self.cfg['pulse_info'][self.pulse_type],self.cfg['buffer'])
        self.pulse_sequence.build_sequence()
        self.pulse_sequence.write_sequence(os.path.join(self.path, '../sequences/'), prefix, upload=True)

        self.expt_pts = self.pulse_sequence.histo_pts

        self.cfg['alazar']['samplesPerRecord'] = 2 ** (self.cfg['readout']['width'] - 1).bit_length()
        self.cfg['alazar']['recordsPerBuffer'] = self.pulse_sequence.sequence_length*self.cfg[self.expt_cfg_name]['repeats']
        self.cfg['alazar']['recordsPerAcquisition'] = int(
            self.pulse_sequence.sequence_length * self.cfg[self.expt_cfg_name]['repeats']* max(self.cfg[self.expt_cfg_name]['averages'], 10))

        self.ready_to_go = True
        return


    def go(self):
        #self.plotter.clear()

        print "Prep Instruments"
        self.readout.set_frequency(self.cfg['readout']['frequency'])
        self.readout.set_power(self.cfg['readout']['power'])
        self.readout.set_ext_pulse(mod=True)


        self.drive.set_frequency(self.cfg['qubit']['frequency'] - self.cfg['pulse_info'][self.pulse_type]['iq_freq'])
        self.drive.set_power(self.cfg[self.expt_cfg_name]['power'])
        self.drive.set_ext_pulse(mod=True)
        self.drive.set_output(True)

        self.awg.set_amps_offsets(self.cfg['cal']['iq_amps'], self.cfg['cal']['iq_offsets'])

        print "Prep Card"
        adc = Alazar(self.cfg['alazar'])

        attenpts = arange(self.cfg[self.expt_cfg_name]['atten_start'], self.cfg[self.expt_cfg_name]['atten_stop'], self.cfg[self.expt_cfg_name]['atten_step'])
        freqpts = arange(self.cfg[self.expt_cfg_name]['freq_start'], self.cfg[self.expt_cfg_name]['freq_stop'], self.cfg[self.expt_cfg_name]['freq_step'])
        num_bins = self.cfg[self.expt_cfg_name]['num_bins']

        for xx, atten in enumerate(attenpts):
            self.im.atten.set_attenuator(atten)
            max_contrast_data_ch1 = zeros(len(freqpts))
            max_contrast_data_ch2 = zeros(len(freqpts))
            #self.plotter.clear('max contrast')

            print "atten at: %s" %atten

            for yy, freq in enumerate(freqpts):
                self.readout.set_frequency(freq)
                #self.readout_shifter.set_phase(self.cfg['readout']['start_phase'] , freq)
                self.readout_shifter.set_phase((self.cfg['readout']['start_phase'] + self.cfg['readout']['phase_slope'] * (freq - self.cfg['readout']['frequency']))%360, freq)
                tpts, ch1_pts, ch2_pts = adc.acquire_avg_data_by_record(prep_function=self.awg.stop_and_prep, start_function=self.awg.run,excise=self.cfg['readout']['window'])
                # self.plotter.plot_z("current",ch1_pts)
                # with self.datafile() as f:
                #     f.append('time_trace', ch1_pts)
                ss_data = zeros((len(self.expt_pts), num_bins))
                sss_data = zeros((len(self.expt_pts), num_bins))

                ss1, ss2 = adc.acquire_singleshot_data(prep_function=self.awg.stop_and_prep, start_function=self.awg.run,
                                                       excise=self.cfg['readout']['window'])

                #with self.datafile() as f:
                #    f.append_line('ss1', ss1)
                #    f.append_line('ss2', ss2)


                ss1 = reshape(ss1, (self.cfg['alazar']['recordsPerAcquisition'] / len(self.expt_pts), len(self.expt_pts))).T
                histo_range = (ss1.min() / 1.05, ss1.max() * 1.05)
                for jj, ss in enumerate(ss1):
                    sshisto, ssbins = np.histogram(ss, bins=num_bins, range=histo_range)
                    ss_data[jj] += sshisto
                    sss_data[jj] = cumsum(ss_data[[jj]])
                #     self.plotter.plot_xy('histogram %d' % jj, ssbins[:-1], ss_data[jj])
                #     self.plotter.plot_xy('cum histo %d' % jj, ssbins[:-1], sss_data[jj])

                max_contrast_data_ch1[yy] = abs(((sss_data[0] - sss_data[1]) / ss_data[0].sum())).max()

                ss2 = reshape(ss2, (self.cfg['alazar']['recordsPerAcquisition'] / len(self.expt_pts), len(self.expt_pts))).T
                histo_range = (ss2.min() / 1.05, ss2.max() * 1.05)
                for jj, ss in enumerate(ss2):
                    sshisto, ssbins = np.histogram(ss, bins=num_bins, range=histo_range)
                    ss_data[jj] += sshisto
                    sss_data[jj] = cumsum(ss_data[[jj]])
                    #self.plotter.plot_xy('histogram %d' % jj, ssbins[:-1], ss_data[jj])
                    #self.plotter.plot_xy('cum histo %d' % jj, ssbins[:-1], sss_data[jj])

                #self.plotter.plot_xy('contrast', ssbins[:-1], abs(sss_data[0] - sss_data[1]) / ss_data[0].sum())
                max_contrast_data_ch2[yy] = abs(((sss_data[0] - sss_data[1]) / ss_data[0].sum())).max()
                #self.plotter.append_xy('max contrast', freq, max_contrast_data[yy])
            if len(attenpts)>1:
                print "plotting max contrast 2"
                pass
                # self.plotter.append_z('max contrast 2', max_contrast_data, start_step=(
                #  (attenpts[0], attenpts[1] - attenpts[0]),(freqpts[0] / 1.0e9, (freqpts[1] - freqpts[0]) / 1.0e9)))
            with self.datafile() as f:
                f.append_pt('atten', atten)
                f.append_line('freq', freqpts)
                f.append_line('max_contrast_data_ch1', max_contrast_data_ch1)
                f.append_line('max_contrast_data_ch2', max_contrast_data_ch2)