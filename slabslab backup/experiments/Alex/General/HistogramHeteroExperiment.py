__author__ = 'AlexMa'

from slab import *
from slab.instruments.Alazar import Alazar
from slab.experiments.Alex.ExpLib.QubitPulseSequenceExperiment import *
from numpy import mean, arange
import numpy as np
from tqdm import tqdm

class HistogramHeteroExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='HistogramHetero', config_file='..\\config.json', **kwargs):

        if prefix == "HistogramRabiThermalizer":
            QubitPulseSequenceExperiment.__init__(self, path=path, prefix='Histogram_Rabi_Thermalizer', config_file=config_file,
                                                  PulseSequence=HistogramRabiThermalizerSequence, pre_run=self.pre_run,
                                                  post_run=self.post_run, **kwargs)
        else:
            QubitPulseSequenceExperiment.__init__(self, path=path, prefix='Histogram_Hetero', config_file=config_file,
                                                  PulseSequence=HistogramHeteroSequence, pre_run=self.pre_run,
                                                  post_run=self.post_run, **kwargs)

    def pre_run(self):
        pass

    def post_run(self, expt_pts, expt_avg_data):
        pass

    # this overrides the method in QubitPulseSequenceExperiment
    def take_data(self):

        print('take_data() in HistogramHetero')

        if self.pre_run is not None:
            self.pre_run()

        if self.adc == None:
            print("Prep Card")
            adc = Alazar(self.cfg['alazar'])
        else:
            adc = self.adc

        ###

        het_IFreqList = array([self.cfg['readout']['heterodyne_freq']])

        avgPerAcquisition = int(min(self.cfg[self.expt_cfg_name]['averages'], 100))
        numAcquisition = int(np.ceil(self.cfg[self.expt_cfg_name]['averages'] / 100))

        attenpts = arange(self.cfg[self.expt_cfg_name]['atten_start'], self.cfg[self.expt_cfg_name]['atten_stop'], self.cfg[self.expt_cfg_name]['atten_step'])
        freqpts = arange(self.cfg[self.expt_cfg_name]['freq_start'], self.cfg[self.expt_cfg_name]['freq_stop'], self.cfg[self.expt_cfg_name]['freq_step'])

        # (channel, atten, freq, g/e/f, average)
        ss_cos_data_all = zeros((2, len(attenpts), len(freqpts), 3, avgPerAcquisition * numAcquisition) )
        ss_sin_data_all = zeros((2, len(attenpts), len(freqpts), 3, avgPerAcquisition * numAcquisition) )

        for xx, atten in enumerate(attenpts):

            try:
                # im = InstrumentManager()
                # atten2 = im['atten2']
                # atten2.set_attenuator(atten)
                self.readout_atten.set_attenuator(atten)
                print(atten, "Digital atten:") #, self.readout_atten.get_attenuator())
                time.sleep(0.5)
                atten2 = None
            except:
                print("Digital attenuator not loaded.")

            # pump_freqs = arange(6.90e9, 7.04e9, 4e6)
            # pump_powers = arange(-6.0, -5, 0.2)
            #
            # self.im['RF6'].set_power( pump_powers[(atten % 5)] )
            # print("TWPA Pump power:", pump_powers[(atten % 5)])
            # self.im['RF6'].set_frequency( pump_freqs[int(atten/5)] )
            # print("TWPA Pump freq:", pump_freqs[int(atten/5)])
            #
            # try:
            #     self.readout_atten.set_attenuator(16.0)
            #     # print("Digital atten:", atten)
            # except:
            #     print("Digital attenuator not loaded.")


            # (ch1/2, exp_pts, heterodyne_freq, cos/sin, all averages)
            ss_data = zeros((2, len(self.expt_pts), len(het_IFreqList), 2, avgPerAcquisition * numAcquisition))
            for ii in tqdm(arange(numAcquisition)):

                if not self.cfg['readout']['is_hetero_phase_ref']:

                    # single_data1/2: index: (hetero_freqs, cos/sin, all_seqs)
                    single_data1, single_data2, single_record1, single_record2 = \
                        adc.acquire_singleshot_heterodyne_multitone_data(het_IFreqList, prep_function=self.awg_prep,
                                                                         start_function=self.awg_run,
                                                                         excise=self.cfg['readout']['window'])
                    # saving the raw time traces
                    # single_data1, single_data2, single_record1, single_record2 = \
                    #     adc.acquire_singleshot_heterodyne_multitone_data(het_IFreqList, prep_function=self.awg_prep,
                    #                                                      start_function=self.awg_run,
                    #                                                      excise=None, save_raw_data=True)

                else:
                    # single_data1/2: index: (hetero_freqs, cos/sin, all_seqs)
                    single_data1, single_data2, single_record1, single_record2 = \
                        adc.acquire_singleshot_heterodyne_multitone_data_phase_ref(het_IFreqList,
                                                                         self.cfg['readout']['hetero_phase_ref_freq'],
                                                                         prep_function=self.awg_prep,
                                                                         start_function=self.awg_run,
                                                                         excise=self.cfg['readout']['window'],
                                                                         isCompensatePhase=True,
                                                                         save_raw_data=False)
                single_data = array([single_data1, single_data2])

                # reshape to fit into histogram data
                # index: (ch1/2, hetero_freqs(0), cos / sin, avgs, freqpts(exp seq), g/e/f)
                single_data = np.reshape(single_data,
                                         (single_data.shape[0], single_data.shape[1], single_data.shape[2],
                                            int(self.cfg['alazar'][
                                              'recordsPerAcquisition'] / self.pulse_sequence.sequence_length),
                                          int(self.pulse_sequence.sequence_length/3), 3))

                # (channel, hetero_freqs(0), cos/sin, freqpts(exp seq), g/e/f, average)
                single_data = np.transpose(single_data, (0, 1, 2, 4, 5, 3))

                # (channel, atten, freqpts, g/e/f, average)
                ss_cos_data_all[:, xx, :, :, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition)] = single_data[:, 0, 0, :, :, :]
                ss_sin_data_all[:, xx, :, :, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition)] = single_data[:, 0, 1, :, :, :]

            # this needs to stay here
            self.slab_file = self.datafile(data_file=self.data_file)
            with self.slab_file as f:

                f.add('attenpts', attenpts)
                f.add('freqpts', freqpts)
                f.add('het_IFreqList', het_IFreqList)
                f.add('ss_cos_data_ch1', ss_cos_data_all[0])
                f.add('ss_cos_data_ch2', ss_cos_data_all[1])
                f.add('ss_sin_data_ch1', ss_sin_data_all[0])
                f.add('ss_sin_data_ch2', ss_sin_data_all[1])
                f.append_line('single_record1', single_record1)
                f.append_line('single_record2', single_record2)
                f.close()

        ###

        # if self.post_run is not None:
        #     self.post_run(self.expt_pts, expt_avg_data)

        if self.cfg['stop_awgs'] == True:
            self.awg_prep()

        # close Alazar and release buffer
        adc.close()