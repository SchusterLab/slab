__author__ = 'Nelson'

from slab import *
from slab.instruments.Alazar import Alazar
from slab.experiments.General.PulseSequences.HistogramSequence import HistogramSequence
from numpy import *

from tqdm import tqdm
from slab.instruments.awg.PXDAC4800 import PXDAC4800
from slab.instruments.pulseblaster.pulseblaster import *

class HistogramHeteroExperiment(Experiment):
    def __init__(self, path='', prefix='Histogram_Hetero', config_file='..\\config.json', use_cal=False, **kwargs):

        Experiment.__init__(self, path=path, prefix=prefix, config_file=config_file, **kwargs)

        self.extra_args={}

        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)

        if 'liveplot_enabled' in self.extra_args:
            self.liveplot_enabled = self.extra_args['liveplot_enabled']
        else:
            self.liveplot_enabled = False

        if 'data_prefix' in self.extra_args:
            data_prefix = self.extra_args['data_prefix']
        else:
            data_prefix = prefix

        if 'prep_tek2' in self.extra_args:
            self.prep_tek2 = self.extra_args['prep_tek2']
        else:
            self.prep_tek2 = False

        if 'adc' in self.extra_args:
            self.adc = self.extra_args['adc']
        else:
            self.adc = None

        if 'data_file' in self.extra_args:
            self.data_file = self.extra_args['data_file']
        else:
            self.data_file = None

        if 'flux_freq' in self.extra_args:
            self.flux_freq = self.extra_args['flux_freq']
        else:
            self.flux_freq = None

        self.prefix = prefix
        self.expt_cfg_name = prefix.lower()

        #self.pre_run = pre_run
        #self.post_run = post_run

        self.pulse_type = self.cfg[self.expt_cfg_name]['pulse_type']

        self.pulse_sequence = HistogramSequence(self.cfg['awgs'], self.cfg[self.expt_cfg_name], self.cfg['readout'],self.cfg['pulse_info'][self.pulse_type],self.cfg['buffer'], self.cfg)
        self.pulse_sequence.build_sequence()
        self.pulse_sequence.write_sequence(os.path.join(self.path, '../sequences/'), prefix, upload=True)

        self.expt_pts = self.pulse_sequence.histo_pts

        self.cfg['alazar']['samplesPerRecord'] = 2 ** (self.cfg['readout']['width'] - 1).bit_length()
        self.cfg['alazar']['recordsPerBuffer'] = self.pulse_sequence.sequence_length*self.cfg[self.expt_cfg_name]['hw_repeats']
        # self.cfg['alazar']['recordsPerAcquisition'] = int(
        #     self.pulse_sequence.sequence_length * self.cfg[self.expt_cfg_name]['repeats']* max(self.cfg[self.expt_cfg_name]['averages'], 10))

        self.cfg['alazar']['recordsPerAcquisition'] = int(self.pulse_sequence.sequence_length * self.cfg[self.expt_cfg_name]['hw_repeats']* max(self.cfg[self.expt_cfg_name]['averages'], 10))

        self.ready_to_go = True
        return


    def go(self):
        if self.liveplot_enabled:
            self.plotter.clear()

        print "Prep Instruments"
        self.readout.set_frequency(self.cfg['readout']['frequency'])
        self.readout.set_power(self.cfg['readout']['power'])
        self.readout.set_ext_pulse(mod=True)

        self.drive.set_frequency(self.cfg['qubit']['frequency'] - self.cfg['pulse_info'][self.pulse_type]['iq_freq'])
        self.drive.set_power(self.cfg[self.expt_cfg_name]['power'])
        self.drive.set_ext_pulse(mod=False)
        self.drive.set_output(True)

        try:
            self.awg.set_amps_offsets(self.cfg['cal']['iq_amps'], self.cfg['cal']['iq_offsets'])
        except:
            print "self.awg not loaded."

        print "Prep Card"
        adc = Alazar(self.cfg['alazar'])

        heterodyne_carrier_freq = self.cfg['readout']['heterodyne_carrier_freq']
        heterodyne_read_freqs = self.cfg['readout']['heterodyne_read_freq']

        attenpts = arange(self.cfg[self.expt_cfg_name]['atten_start'], self.cfg[self.expt_cfg_name]['atten_stop'], self.cfg[self.expt_cfg_name]['atten_step'])
        freqpts = arange(self.cfg[self.expt_cfg_name]['freq_start'], self.cfg[self.expt_cfg_name]['freq_stop'], self.cfg[self.expt_cfg_name]['freq_step'])
        num_bins = self.cfg[self.expt_cfg_name]['num_bins']

        ss_cos_data_all = zeros((2, len(attenpts), len(freqpts), len(self.expt_pts),
                             self.cfg['alazar']['recordsPerAcquisition'] / len(self.expt_pts))
                            )  # (channel, atten, freq, g/e/(f), average)
        ss_sin_data_all = zeros((2, len(attenpts), len(freqpts), len(self.expt_pts),
                                 self.cfg['alazar']['recordsPerAcquisition'] / len(self.expt_pts))
                                )  # (channel, atten, freq, g/e/(f), average)

        for xx, atten in enumerate(attenpts):
            #self.im.atten.set_attenuator(atten)
            try:
                self.readout_atten.set_attenuator(atten)
            except:
                print "Digital attenuator not loaded."

            max_contrast_data = zeros((2,len(freqpts)))

            if self.liveplot_enabled:
                self.plotter.clear('max contrast')

            for yy, freq in enumerate(freqpts):
                self.readout.set_frequency(freq-self.cfg['readout']['heterodyne_freq'])
                #self.readout_shifter.set_phase(self.cfg['readout']['start_phase'] , freq)
                self.readout_shifter.set_phase((self.cfg['readout']['start_phase'] + self.cfg['readout']['phase_slope'] * (freq - self.cfg['readout']['frequency']))%360, freq)

                ss_data = zeros((len(self.expt_pts), num_bins))
                sss_data = zeros((len(self.expt_pts), num_bins))

                print "runnning atten no.", xx, ", freq no.", yy

                single_data1, single_data2, single_record1 = \
                    adc.acquire_singleshot_heterodyne_data(self.cfg['readout']['heterodyne_freq'],\
                                                       prep_function=self.awg_prep, start_function=self.awg_run, excise=self.cfg['readout']['window'])

                ss_cos_data_all[0, xx, yy, :, :] = reshape(single_data1[0], (self.cfg['alazar']['recordsPerAcquisition'] / len(self.expt_pts), len(self.expt_pts))).T
                ss_cos_data_all[1, xx, yy, :, :] = reshape(single_data2[0], (self.cfg['alazar']['recordsPerAcquisition'] / len(self.expt_pts), len(self.expt_pts))).T
                ss_sin_data_all[0, xx, yy, :, :] = reshape(single_data1[1], (self.cfg['alazar']['recordsPerAcquisition'] / len(self.expt_pts), len(self.expt_pts))).T
                ss_sin_data_all[1, xx, yy, :, :] = reshape(single_data2[1], (self.cfg['alazar']['recordsPerAcquisition'] / len(self.expt_pts), len(self.expt_pts))).T

                with self.datafile() as f:
                    f.add('ss_cos_data_ch1', ss_cos_data_all[0])
                    f.add('ss_cos_data_ch2', ss_cos_data_all[1])
                    f.add('ss_sin_data_ch1', ss_sin_data_all[0])
                    f.add('ss_sin_data_ch2', ss_sin_data_all[1])
                    f.close()

                #here max_contrast only calculated for ch1 & 2 of (g/e)
                ss1 = single_data1[0]
                ss2 = single_data1[1]

                for kk, ssthis in enumerate([ss1, ss2]):

                    ssthis = reshape(ssthis, (self.cfg['alazar']['recordsPerAcquisition'] / len(self.expt_pts), len(self.expt_pts))).T

                    print 'ss i/q ch', str(kk+1), 'max/min =', ssthis.max(), ssthis.min()
                    dist = ssthis.max() - ssthis.min()
                    histo_range = (ssthis.min() - 0.01*dist, ssthis.max() + 0.01*dist)

                    for jj, ss in enumerate(ssthis):
                        sshisto, ssbins = np.histogram(ss, bins=num_bins, range=histo_range)
                        ss_data[jj] += sshisto
                        sss_data[jj] = cumsum(ss_data[[jj]])
                        if self.liveplot_enabled:
                            self.plotter.plot_xy('histogram %d' % jj, ssbins[:-1], ss_data[jj])
                            self.plotter.plot_xy('cum histo %d' % jj, ssbins[:-1], sss_data[jj])

                    max_contrast_data[kk, yy] = abs(((sss_data[0] - sss_data[1]) / ss_data[0].sum())).max()

                    if self.liveplot_enabled:
                        self.plotter.plot_xy('contrast_ch' + str(kk+1), ssbins[:-1], abs(sss_data[0] - sss_data[1]) / ss_data[0].sum())
                        self.plotter.append_xy('max contrast_ch' + str(kk+1), freq, max_contrast_data[kk,yy])

            with self.datafile() as f:
                f.append_pt('atten', atten)
                f.add('attenpts', attenpts)
                f.append_line('freq', freqpts)
                f.append_line('max_contrast_data_ch1', max_contrast_data[0, :])
                f.append_line('max_contrast_data_ch2', max_contrast_data[1, :])
                f.add('single_record1', single_record1)
                f.close()


    def awg_prep(self):
        stop_pulseblaster()
        LocalInstruments().inst_dict['pxdac4800_1'].stop()
        LocalInstruments().inst_dict['pxdac4800_2'].stop()

    def awg_run(self):
        LocalInstruments().inst_dict['pxdac4800_1'].run_experiment()
        LocalInstruments().inst_dict['pxdac4800_2'].run_experiment()
        time.sleep(0.5)
        run_pulseblaster()