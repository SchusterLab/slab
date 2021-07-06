from slab import InstrumentManager
from slab.instruments.awg import write_Tek5014_file
from slab.instruments.awg.M8195A import upload_M8195A_sequence
from slab.instruments.awg import M8195A
from slab.instruments.Alazar import Alazar
import numpy as np
import os
import time
from tqdm import tqdm
import visdom
from slab.datamanagement import SlabFile
from slab.dataanalysis import get_next_filename
import json
from slab.experiments.PulseExperiments.get_data import get_iq_data, get_singleshot_data


class Experiment:
    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg):
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg

        im = InstrumentManager()

        self.tek = im['TEK']
        self.m8195a = im['M8195A_2']

        self.rf1 = im['RF1']
        self.rf2 = im['RF2']

        self.flux1 = im['YOKO3']
        self.flux2 = im['YOKO2']

        #self.trig = im['trig']


    def initiate_tek(self, name, path, sequences):
        print(self.tek.get_id())
        tek_waveform_channels_num = 4
        tek_waveform_channels = self.hardware_cfg['awg_info']['tek5014a']['waveform_channels']
        tek_marker_channels = self.hardware_cfg['awg_info']['tek5014a']['marker_channels']
        tek_waveforms = []
        for channel in tek_waveform_channels:
            if not channel == None:
                tek_waveforms.append(sequences[channel])
            else:
                tek_waveforms.append(np.zeros_like(sequences[tek_waveform_channels[0]]))
        tek_markers = []
        for channel in tek_marker_channels:
            if not channel == None:
                tek_markers.append(sequences[channel])
            else:
                tek_markers.append(np.zeros_like(sequences[tek_marker_channels[0]]))
        write_Tek5014_file(tek_waveforms, tek_markers, os.path.join(path, 'sequences/tek.awg'), name)
        self.tek.pre_load()
        self.tek.load_sequence_file(os.path.join(path, 'sequences/tek.awg'), force_reload=True)
        self.tek.set_amps_offsets(channel_offsets=self.hardware_cfg['awg_info']['tek5014a']['offsets'])


    def initiate_m8195a(self, path, sequences):
        print(self.m8195a.get_id())
        waveform_channels = self.hardware_cfg['awg_info']['m8195a']['waveform_channels']
        waveform_matrix = [sequences[channel] for channel in waveform_channels]

        awg_info = self.hardware_cfg['awg_info']['m8195a']

        upload_M8195A_sequence(self.m8195a, waveform_matrix, awg_info, path)

    def awg_prep(self):
        self.tek.stop()
        self.tek.prep_experiment()
        self.m8195a.stop_output()
        time.sleep(1)

    def awg_run(self):
        self.m8195a.start_output()
        time.sleep(1)
        self.tek.run()


    def initiate_alazar(self, sequence_length, averages):
        self.hardware_cfg['alazar']['samplesPerRecord'] = 2 ** (
            self.quantum_device_cfg['alazar_readout']['width'] - 1).bit_length()
        self.hardware_cfg['alazar']['recordsPerBuffer'] = sequence_length
        self.hardware_cfg['alazar']['recordsPerAcquisition'] = int(
            sequence_length * min(averages, 100))
        print("Prep Alazar Card")
        self.adc = Alazar(self.hardware_cfg['alazar'])
        print("Alazar prepared")

    def initiate_readout_rf(self):
        self.rf1.set_frequency(self.quantum_device_cfg['heterodyne']['1']['lo_freq'] * 1e9)
        self.rf2.set_frequency(self.quantum_device_cfg['heterodyne']['2']['lo_freq'] * 1e9)
        self.rf1.set_power(self.quantum_device_cfg['heterodyne']['1']['lo_power'])
        self.rf2.set_power(self.quantum_device_cfg['heterodyne']['2']['lo_power'])
        self.rf1.set_ext_pulse(mod=True)
        self.rf2.set_ext_pulse(mod=True)

    def initiate_flux(self):
        if self.quantum_device_cfg['freq_flux']['calibrated']:
            # self.flux1.ramp_current((self.quantum_device_cfg['freq_flux']['1']['current_mA'] + 0.342424 * self.quantum_device_cfg['freq_flux']['2']['current_mA'] - 0.117121)* 1e-3)
            # self.flux2.ramp_current((self.quantum_device_cfg['freq_flux']['2']['current_mA'] + 0.0857875 * self.quantum_device_cfg['freq_flux']['1']['current_mA'] + 0.0585787) * 1e-3)
            print('Flux1: ', (self.quantum_device_cfg['freq_flux']['1']['current_mA'] + 0.0670103 * self.quantum_device_cfg['freq_flux']['2']['current_mA'] + 4.90))
            print('Flux2: ', (self.quantum_device_cfg['freq_flux']['2']['current_mA'] - 0.353293 * self.quantum_device_cfg['freq_flux']['1']['current_mA'] + 6.50))
            self.flux1.ramp_current((self.quantum_device_cfg['freq_flux']['1']['current_mA'] + 0.0670103 * self.quantum_device_cfg['freq_flux']['2']['current_mA'] + 4.90) * 1e-3)
            self.flux2.ramp_current((self.quantum_device_cfg['freq_flux']['2']['current_mA'] - 0.353293 * self.quantum_device_cfg['freq_flux']['1']['current_mA'] + 6.50) * 1e-3)
        else:
            print('Flux1: ', self.quantum_device_cfg['freq_flux']['1']['current_mA'])
            print('Flux2: ', self.quantum_device_cfg['freq_flux']['2']['current_mA'])
            self.flux1.ramp_current(self.quantum_device_cfg['freq_flux']['1']['current_mA'] * 1e-3)
            self.flux2.ramp_current(self.quantum_device_cfg['freq_flux']['2']['current_mA'] * 1e-3)

    def set_trigger(self):
       try:
           period = self.hardware_cfg['trigger']['period_us']
           self.trig.set_period(period*1e-6)
           print ("Trigger period set to ", period,"us")
       except:
           print("Error in trigger configuration")

    def set_trigger_tek(self): # Added by Tanay
       try:
           period = self.hardware_cfg['trigger']['period_us']
           self.tek.set_trigger_source('INT')
           self.tek.set_trigger_interval(period*1e-6)
           print ("Trigger period set to ", period,"us")
       except:
           print("Error in trigger configuration")

    def save_cfg_info(self, f):
        f.attrs['quantum_device_cfg'] = json.dumps(self.quantum_device_cfg)
        f.attrs['experiment_cfg'] = json.dumps(self.experiment_cfg)
        f.attrs['hardware_cfg'] = json.dumps(self.hardware_cfg)
        f.close()

    def get_singleshot_data(self, sequence_length, acquisition_num, data_file, seq_data_file):
        avgPerAcquisition = int(min(acquisition_num, 100))
        numAcquisition = int(np.ceil(acquisition_num / 100))
        het_IFreqList = []

        for qubit_id in ["1","2"]:
            het_IFreqList += [self.quantum_device_cfg['heterodyne'][qubit_id]['freq']]

        single_data1_list = []
        single_data2_list = []
        for ii in tqdm(np.arange(numAcquisition)):
            # single_data1/2: index: (hetero_freqs, cos/sin, all_seqs)
            single_data1, single_data2, single_record1, single_record2 = \
                self.adc.acquire_singleshot_heterodyne_multitone_data_2(het_IFreqList, prep_function=self.awg_prep,
                                                                        start_function=self.awg_run,
                                                                        excise=
                                                                        self.quantum_device_cfg['alazar_readout'][
                                                                            'window'])
            single_data1_list.append(single_data1)
            single_data2_list.append(single_data2)

            single_data1 = np.array(single_data1_list)
            single_data2 = np.array(single_data2_list)

            single_data1 = np.transpose(single_data1, (1, 2, 0, 3))
            single_data1 = single_data1.reshape(*single_data1.shape[:2], -1)

            single_data2 = np.transpose(single_data2, (1, 2, 0, 3))
            single_data2 = single_data2.reshape(*single_data2.shape[:2], -1)

            single_data1 = single_data1.reshape(*single_data1.shape[:2], -1, sequence_length)
            single_data2 = single_data2.reshape(*single_data2.shape[:2], -1, sequence_length)


            single_data1 = np.transpose(single_data1, (0, 1, 3, 2))
            single_data2 = np.transpose(single_data2, (0, 1, 3, 2))
            # now becomes: single_data1/2: index: (hetero_freqs, cos/sin , seqs, acquisitions)
            data_1_cos_list, data_1_sin_list, data_1_list = get_singleshot_data(single_data1, 0,
                                                                                self.expt_cfg.get('pi_calibration',
                                                                                                  False))
            data_2_cos_list, data_2_sin_list, data_2_list = get_singleshot_data(single_data2, 1,
                                                                                self.expt_cfg.get('pi_calibration',
                                                                                                  False))
            data_1_avg_list = np.mean(data_1_list, axis=1)
            data_2_avg_list = np.mean(data_2_list, axis=1)

            if seq_data_file == None:
                self.slab_file = SlabFile(data_file)
                with self.slab_file as f:
                    f.add('single_data1', single_data1)
                    f.add('expt_avg_data_ch1', data_1_avg_list)
                    f.add('single_data2', single_data2)
                    f.add('expt_avg_data_ch2', data_2_avg_list)
                    f.close()

        if not seq_data_file == None:
            self.slab_file = SlabFile(data_file)
            with self.slab_file as f:
                f.append('single_data1', single_data1)
                f.append('single_data2', single_data2)
                f.append_line('expt_avg_data_ch1', data_1_avg_list)
                f.append_line('expt_avg_data_ch2', data_2_avg_list)
                f.close()
        self.adc.close()
        self.awg_prep()

    def get_avg_data(self, acquisition_num, data_file, seq_data_file):
        expt_data_ch1 = None
        expt_data_ch2 = None
        for ii in tqdm(np.arange(max(1, int(acquisition_num / 100)))):
            tpts, ch1_pts, ch2_pts = self.adc.acquire_avg_data_by_record(prep_function=self.awg_prep,
                                                                         start_function=self.awg_run,
                                                                         excise=
                                                                         self.quantum_device_cfg['alazar_readout'][
                                                                             'window'])

            if expt_data_ch1 is None:
                expt_data_ch1 = ch1_pts
                expt_data_ch2 = ch2_pts
            else:
                expt_data_ch1 = (expt_data_ch1 * ii + ch1_pts) / (ii + 1.0)
                expt_data_ch2 = (expt_data_ch2 * ii + ch2_pts) / (ii + 1.0)

            data_1_cos_list, data_1_sin_list, data_1_list = get_iq_data(expt_data_ch1,
                                                                        het_freq=
                                                                        self.quantum_device_cfg['heterodyne']['1'][
                                                                            'freq'],
                                                                        td=0,
                                                                        pi_cal=self.expt_cfg.get('pi_calibration',
                                                                                                 False))
            data_2_cos_list, data_2_sin_list, data_2_list = get_iq_data(expt_data_ch2,
                                                                        het_freq=
                                                                        self.quantum_device_cfg['heterodyne']['2'][
                                                                            'freq'],
                                                                        td=0,
                                                                        pi_cal=self.expt_cfg.get('pi_calibration',
                                                                                                 False))

            if seq_data_file == None:
                self.slab_file = SlabFile(data_file)
                with self.slab_file as f:
                    f.add('expt_avg_data_ch1', data_1_list)
                    f.add('expt_avg_data_ch2', data_2_list)
                    if self.expt_cfg.get('time_bin_data', False): # Added by Tanay - Jul 16 2019
                        f.add('expt_data_ch1', expt_data_ch1)
                        f.add('expt_data_ch2', expt_data_ch2)
                    else:
                        pass
                    f.close()

        self.adc.close()
        self.awg_prep()
        if not seq_data_file == None:
            self.slab_file = SlabFile(data_file)
            with self.slab_file as f:
                f.append_line('expt_avg_data_ch1', data_1_list)
                f.append_line('expt_avg_data_ch2', data_2_list)
                f.close()

    def run_experiment(self, sequences, path, name, seq_data_file=None, update_awg=True):

        self.initiate_readout_rf()
        self.initiate_flux()

        if update_awg:
            self.initiate_tek(name, path, sequences)
            self.initiate_m8195a(path, sequences)

        self.m8195a.start_output()
        self.tek.prep_experiment()
        self.tek.run()
        # self.set_trigger() # Disabled by Tanay as it was not working properly
        self.set_trigger_tek()

        sequence_length = len(sequences['charge1'])

        self.expt_cfg = self.experiment_cfg[name]

        acquisition_num = self.expt_cfg['acquisition_num']

        self.initiate_alazar(sequence_length, acquisition_num)

        if seq_data_file == None:
            data_path = os.path.join(path, 'data/')
            data_file = os.path.join(data_path, get_next_filename(data_path, name, suffix='.h5'))
        else:
            data_file = seq_data_file
        self.slab_file = SlabFile(data_file)
        with self.slab_file as f:
            self.save_cfg_info(f)

        print('\n')
        print(data_file)

        if self.expt_cfg.get('singleshot', False): # Changed default to False - Tanay Jul 15 2019
            self.get_singleshot_data(sequence_length, acquisition_num, data_file, seq_data_file)
        else:
            self.get_avg_data(acquisition_num, data_file, seq_data_file)

        return data_file

