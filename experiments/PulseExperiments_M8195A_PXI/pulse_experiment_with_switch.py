from slab import InstrumentManager
from slab.instruments.awg import write_Tek5014_file
from slab.instruments.awg.M8195A import upload_M8195A_sequence
# import keysight_pxi_load as ks_pxi
from slab.experiments.PulseExperiments_M8195A_PXI import keysight_pxi_load_m8195a_with_switch as ks_pxi
from slab.instruments.keysight import KeysightLib as key
from slab.instruments.keysight import keysightSD1 as SD1
from slab.instruments.awg.Tek70001 import write_Tek70001_sequence
# from slab.instruments.awg.Tek70001 import write_Tek70001_file
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
# from slab.experiments.PulseExperiments.get_data import get_iq_data, get_singleshot_data
from slab.experiments.PulseExperiments_M8195A_PXI.PostExperimentAnalysis import PostExperiment

im = InstrumentManager()

class Experiment:
    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg,sequences=None, name=None, hvi=True,
                 use_cavity=False):
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg
        self.use_cavity = use_cavity

        try: self.pxi = ks_pxi.KeysightSingleQubit(self.experiment_cfg, self.hardware_cfg,self.quantum_device_cfg,
                                                    sequences, name, hvi=hvi)
        except: print("Not connected to keysight PXI")

        try: self.drive_los = [im[lo] for lo in self.hardware_cfg['drive_los']]
        except: print ("No drive function generator specified in hardware config")
        
        try: self.jpa_pump_los = [im[lo] for lo in self.hardware_cfg['jpa_pump_los']]
        except: print ("No JPA pump LO specified in hardware config")

        try: self.readout_los = [im[lo] for lo in self.hardware_cfg['readout_los']]
        except: print ("No readout function generator specified in hardware config")
        
        try: self.cavity_drive_los = [im[lo] for lo in self.hardware_cfg['cavity_drive_los']]
        except: print ("No cavity drive function generator specified in hardware config")

        try: self.attens = [im[atten] for atten in self.hardware_cfg['attens']]
        except: print ("No digital attenuator specified in hardware config")

        try: self.trig = im['trig']
        except: print ("No trigger function generator specied in hardware cfg")

        try: self.m8195a = im['M8195A_1']
        except: print ("Unable to connect to M8195a")

        try:self.tek2 = im['TEK2']
        except:print("No tek2")

        try:
            self.rotate_iq = self.quantum_device_cfg['readout']['rotate_iq_dig']
            self.iq_angle = self.quantum_device_cfg['readout']['iq_angle']
        except:
            self.rotate_iq = False
            self.iq_angle = 0.0

        self.I = None
        self.Q = None
        self.prep_tek2 = False


    def initiate_pxi(self, name, sequences):
        try:self.m8195a.stop_output()
        except: print("Could not stop M8195A output!")
        try:
            self.pxi.AWG_module.stopAll()
            self.pxi.m_module.stopAll()
            self.pxi.trig_module.stopAll()
            self.pxi.DIG_module.stopAll()
            # self.pxi.switch_module.stopAll()
        except: print("Could not stop a module!")

        pxi_waveform_channels = self.hardware_cfg['awg_info']['keysight_pxi']['waveform_channels']
        pxi_sequences = {}
        for channel in pxi_waveform_channels:
            pxi_sequences[channel] = sequences[channel]

        self.pxi.configureChannels(self.hardware_cfg, self.experiment_cfg, name)
        self.pxi.loadAndQueueWaveforms(pxi_sequences)

    def initiate_tek2(self, name,path, sequences):
        if 'sideband' in name:
            try:
                print("Connected to", self.tek2.get_id())
                tek2_waveform_channels = self.hardware_cfg['awg_info']['tek70001a']['waveform_channels']
                tek2_waveforms = [sequences[channel] for channel in tek2_waveform_channels]
                for waveform in tek2_waveforms:
                    write_Tek70001_sequence(waveform,os.path.join(path, 'sequences/'), name,awg=self.tek2)
                self.tek2.prep_experiment()
            except:print("tek2 sequence not uploaded")

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

    def awg_run(self,run_pxi = True,name=None):
        self.m8195a.start_output()
        time.sleep(1)
        if run_pxi:
            self.pxi.run()
    
    def clear_and_restart_digitizer(self):
        self.pxi.clear_and_start_digitizer()

    def awg_stop(self,name):
        # try:
        self.pxi.AWG_module.stopAll()
        self.pxi.AWG_module.clearAll()
        self.pxi.m_module.stopAll()
        self.pxi.m_module.clearAll()
        self.pxi.trig_module.stopAll()
        self.pxi.trig_module.clearAll()
        self.pxi.DIG_module.stopAll()
        self.pxi.chassis.close()
        # except:print('Error in stopping and closing PXI')
    
        try:
            self.m8195a.stop_output()
            time.sleep(0.1)
        except:print('Error in stopping M8195a')

    def pxi_stop(self):
        try:
            self.pxi.AWG_module.stopAll()
            self.pxi.AWG_module.clearAll()
            self.pxi.m_module.stopAll()
            self.pxi.m_module.clearAll()
            self.pxi.trig_module.stopAll()
            self.pxi.trig_module.clearAll()
            self.pxi.DIG_module.stopAll()
            self.pxi.chassis.close()
        except:print('Error in stopping and closing PXI')

    def initiate_alazar(self, sequence_length, averages):
        self.hardware_cfg['alazar']['samplesPerRecord'] = 2 ** (
            self.quantum_device_cfg['alazar_readout']['width'] - 1).bit_length()
        self.hardware_cfg['alazar']['recordsPerBuffer'] = sequence_length
        self.hardware_cfg['alazar']['recordsPerAcquisition'] = int(
            sequence_length * min(averages, 100))
        print("Prep Alazar Card")
        self.adc = Alazar(self.hardware_cfg['alazar'])

    def initiate_readout_rf_m8195a(self):
        self.rf1.set_frequency(self.quantum_device_cfg['heterodyne']['1']['lo_freq'] * 1e9)
        self.rf2.set_frequency(self.quantum_device_cfg['heterodyne']['2']['lo_freq'] * 1e9)
        self.rf1.set_power(self.quantum_device_cfg['heterodyne']['1']['lo_power'])
        self.rf2.set_power(self.quantum_device_cfg['heterodyne']['2']['lo_power'])
        self.rf1.set_ext_pulse(mod=True)
        self.rf2.set_ext_pulse(mod=True)

    def initiate_drive_LOs(self):
        try:
            for ii,d in enumerate(self.drive_los):
                drive_freq = self.quantum_device_cfg['qubit'][str(ii+1)]['freq'] - self.quantum_device_cfg['pulse_info'][str(ii+1)]['iq_freq']
                d.set_frequency(drive_freq*1e9)
                d.set_power(self.quantum_device_cfg['qubit_drive_lo_powers'][str(ii+1)])
                d.set_ext_pulse(mod=True)
        except:print ("Error in qubit drive LO configuration")
        
        
    def initiate_jpa_pump_LOs(self):
        if self.quantum_device_cfg['readout']['jpa_pump']:

            try:
                for ii,d in enumerate(self.jpa_pump_los):
                    pump_freq = self.quantum_device_cfg['readout']['jpa_pump_freq']
                    d.set_frequency(pump_freq*1e9)
                    # d.set_power(self.quantum_device_cfg['jpa_pump_lo_powers'][str(ii+1)])
                    d.set_ext_pulse(mod=True)
            except:print ("Error in jpa pump LO configuration")
        else:print("JPA pump is off")


    def initiate_readout_LOs(self):
        try:
            for ii, d in enumerate(self.readout_los):
                d.set_frequency(self.quantum_device_cfg['readout']['freq']*1e9)
                print ("Readout frequency = ",self.quantum_device_cfg['readout']['freq'],"GHz")
                # if self.quantum_device_cfg['readout']['jpa_pump'] and  self.quantum_device_cfg['readout']['iq_mix_readout']:
                #     d.set_power(self.quantum_device_cfg['readout']['jpa_pump_power'])
                #     print ("MXG being used to pump JPA, JPA power set to",self.quantum_device_cfg['readout']['jpa_pump_power'])
                # else:d.set_power(self.quantum_device_cfg['readout_drive_lo_powers'][str(ii + 1)])
                d.set_power(self.quantum_device_cfg['readout_drive_lo_powers'][str(ii + 1)])
                d.set_ext_pulse(mod=True)
        except:print("Error in readout drive LO configuration")
        
    def initiate_cavity_drive_LOs(self,name):
        if 'cavity_drive' in name:
            try:
                for ii, d in enumerate(self.cavity_drive_los):
                    drive_freq = self.quantum_device_cfg['cavity'][str(ii + 1)]['freq'] - \
                                 self.quantum_device_cfg['cavity_pulse_info'][str(ii + 1)]['iq_freq']
                    d.set_frequency(drive_freq * 1e9)
                    d.set_power(self.quantum_device_cfg['cavity_drive_lo_powers'][str(ii + 1)])
                    d.set_ext_pulse(mod=True)
                    print ("Cavity LO configured")
            except:
                print("Error in cavity drive LO configuration")
        elif self.quantum_device_cfg['readout']['iq_mix_readout']:
            pass

    def initiate_attenuators(self):
        try:
            for ii, d in enumerate(self.attens):
                d.set_attenuator(self.quantum_device_cfg['readout']['dig_atten'])
            print("Digital attenuator set to", self.quantum_device_cfg['readout']['dig_atten'])
        except:
            print("Error in digital attenuator configuration")

    def set_trigger(self):
        try:
            period = self.hardware_cfg['trigger']['period_us']
            self.trig.set_period(period*1e-6)
            print ("Trigger period set to ", period,"us")
        except:
            print("Error in trigger configuration")

    def initiate_flux(self):
        self.flux1.ramp_current(self.quantum_device_cfg['freq_flux']['1']['current_mA'] * 1e-3)
        self.flux2.ramp_current(self.quantum_device_cfg['freq_flux']['2']['current_mA'] * 1e-3)

    def save_cfg_info(self, f):
        f.attrs['quantum_device_cfg'] = json.dumps(self.quantum_device_cfg)
        f.attrs['experiment_cfg'] = json.dumps(self.experiment_cfg)
        f.attrs['hardware_cfg'] = json.dumps(self.hardware_cfg)
        f.close()

    def generate_datafile(self,path,name,seq_data_file = None):
        # seq_data_file = None
        if seq_data_file == None:
            data_path = os.path.join(path, 'data/')
            self.data_file = os.path.join(data_path, get_next_filename(data_path, name, suffix='.h5'))
        else:
            self.data_file = seq_data_file
        self.slab_file = SlabFile(self.data_file)
        with self.slab_file as f:
            self.save_cfg_info(f)
        print('\n')
        print(self.data_file)

    def get_singleshot_data_alazar(self, sequence_length, acquisition_num, data_file, seq_data_file):
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

            # single_data1/2: index: (hetero_freqs, cos/sin , seqs, acquisitions)
            single_data1 = np.transpose(single_data1, (0, 1, 3, 2))
            single_data2 = np.transpose(single_data2, (0, 1, 3, 2))

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

    def get_avg_data_alazar(self, acquisition_num, data_file, seq_data_file):
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
                    f.add('expt_data_ch1', expt_data_ch1)
                    f.add('expt_avg_data_ch1', data_1_list)
                    f.add('expt_data_ch2', expt_data_ch2)
                    f.add('expt_avg_data_ch2', data_2_list)
                    f.close()
        self.adc.close()
        self.awg_prep()
        if not seq_data_file == None:
            self.slab_file = SlabFile(data_file)
            with self.slab_file as f:
                f.append_line('expt_avg_data_ch1', data_1_list)
                f.append_line('expt_avg_data_ch2', data_2_list)
                f.close()

    def get_avg_data_pxi(self,expt_cfg, seq_data_file,rotate_iq = False,phi=0):
        w = self.pxi.readout_window/self.pxi.dt_dig
        # expt_pts = np.arange(expt_cfg['start'],expt_cfg['stop'],expt_cfg['step'])

        try:pi_calibration = expt_cfg['pi_calibration']
        except:pi_calibration = False

        I,Q = self.pxi.acquire_avg_data(w,pi_calibration,rotate_iq,phi)
        if seq_data_file == None:
            self.slab_file = SlabFile(self.data_file)
            with self.slab_file as f:
                # f.add('expt_pts',expt_pts)
                f.add('I', I)
                f.add('Q', Q)
        else:
            self.slab_file = SlabFile(seq_data_file)
            with self.slab_file as f:
                f.append_line('I', I)
                f.append_line('Q', Q)

        return I,Q

    def get_ss_data_pxi(self,expt_cfg, seq_data_file):
        w = self.pxi.readout_window/self.pxi.dt_dig

        I,Q = self.pxi.SSdata_many(w)
        if seq_data_file == None:
            self.slab_file = SlabFile(self.data_file)
            with self.slab_file as f:
                f.add('I', I)
                f.add('Q', Q)
        else:
            self.slab_file = SlabFile(seq_data_file)
            with self.slab_file as f:
                f.append_line('I', I.flatten())
                f.append_line('Q', Q.flatten())

        return I,Q

    def run_experiment(self, sequences, path, name, seq_data_file=None, update_awg=True):

        self.initiate_readout_rf_m8195a()
        self.initiate_flux()

        if update_awg:
            self.initiate_tek(name, path, sequences)
            self.initiate_m8195a(path, sequences)

        self.m8195a.start_output()
        self.tek.prep_experiment()
        self.tek.run()

        sequence_length = len(sequences['charge1'])

        self.expt_cfg = self.experiment_cfg[name]
        acquisition_num = self.expt_cfg['acquisition_num']

        self.initiate_alazar(sequence_length, acquisition_num)
        self.generate_datafile(path, name, seq_data_file)

        if self.expt_cfg.get('singleshot', True):
            self.get_singleshot_data_alazar(sequence_length, acquisition_num, self.data_file, seq_data_file)
        else:
            self.get_avg_data_alazar(acquisition_num, self.data_file, seq_data_file)

        return self.data_file

    def run_experiment_pxi(self, sequences, path, name, seq_data_file=None,update_awg=False,expt_num = 0,check_sync = False,save_errs = False):
        self.expt_cfg = self.experiment_cfg[name]
        self.generate_datafile(path,name,seq_data_file=seq_data_file)
        self.set_trigger()
        self.initiate_readout_LOs()
        self.initiate_jpa_pump_LOs()
        self.initiate_attenuators()
        self.initiate_pxi(name, sequences)
        self.initiate_m8195a(path,sequences)
        # old way of triggering switch before experiment starts and leaving it on the whole time (not useful)
        # switch_trig = key.KeysightChannelTrig(self.pxi.switch_module, mode=0)  # 0 out, 1 in, defaults to 0
        # if 'cavity' in name or self.use_cavity:
        #     switch_trig.write(True)
        #     print("Switch for cavity drive triggered")
        # else:
        #     switch_trig.write(False)
        #     print("Cavity drive off")
        
        time.sleep(0.1)
        self.awg_run(run_pxi=True,name=name)
        try:
            if check_sync:self.pxi.acquireandplot(expt_num)
            else:
                if self.expt_cfg['singleshot']:
                    self.I,self.Q =  self.get_ss_data_pxi(self.expt_cfg,seq_data_file=seq_data_file)
                else:
                    self.I,self.Q = self.get_avg_data_pxi(self.expt_cfg,seq_data_file=seq_data_file,rotate_iq = self.rotate_iq,phi=self.iq_angle)
        except:print("Error in data acquisition from PXI")
        
        # old way of turning off switch after entire experiment
        # switch_trig.write(False)
        self.awg_stop(name)
        return self.I,self.Q
    
    def run_experiment_pxi_repeated(self, sequences, path, name, seq_data_file=None,update_awg=False,expt_num = 0,check_sync = False,save_errs = False, no_load = False, clear_pxi = False):
        self.expt_cfg = self.experiment_cfg[name]
        self.generate_datafile(path,name,seq_data_file=seq_data_file)
        self.initiate_pxi(name, sequences)
        if not no_load:
            self.initiate_m8195a(path,sequences)
            self.set_trigger()
            self.initiate_readout_LOs()
            self.initiate_jpa_pump_LOs()
            self.initiate_attenuators()
        time.sleep(0.1)
        self.awg_run(run_pxi=True,name=name)
        try:
            if check_sync:self.pxi.acquireandplot(expt_num)
            else:
                if self.expt_cfg['singleshot']:
                    self.I,self.Q =  self.get_ss_data_pxi(self.expt_cfg,seq_data_file=seq_data_file)
                else:
                    self.I,self.Q = self.get_avg_data_pxi(self.expt_cfg,seq_data_file=seq_data_file,rotate_iq = self.rotate_iq,phi=self.iq_angle)
        except:print("Error in data acquisition from PXI")

        self.awg_stop(name)
        return self.I,self.Q

    def post_analysis(self,experiment_name,P='Q',show = False,check_sync = False, return_val=False):
        if check_sync:pass
        else:PA = PostExperiment(self.quantum_device_cfg, self.experiment_cfg, self.hardware_cfg, experiment_name, self.I ,self.Q, P,show)
        if return_val:
            return PA