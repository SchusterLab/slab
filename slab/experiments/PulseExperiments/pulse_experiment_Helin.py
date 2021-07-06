from slab import InstrumentManager
# from slab.instruments.awg import write_Tek5014_file
# from slab.instruments.awg.M8195A import upload_M8195A_sequence
# import keysight_pxi_load as ks_pxi
# from slab.instruments.keysight import keysight_pxi_load as ks_pxi
# from slab.instruments.keysight import KeysightLib as key
# from slab.instruments.keysight import keysightSD1 as SD1
from slab.instruments.awg.Tek70001 import write_Tek70001_sequence
# from slab.instruments.awg.Tek70001 import write_Tek70001_file
# from slab.instruments.awg import M8195A
from slab.instruments.pulseblaster.pulseblaster_Helin import *
from slab.instruments.awg import write_PXDAC4800_file
from slab.instruments.awg import AWG81180A
from slab.instruments.awg.PXDAC4800 import PXDAC4800
from slab.instruments.SignalCore import *
from slab import AttrDict, LocalInstruments
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
from slab.experiments.PulseExperiments_PXI.PostExperimentAnalysis import PostExperiment

class Experiment:
    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg,sequences=None, name=None):
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg
        im = InstrumentManager()
        # try: self.pxi =  ks_pxi.KeysightSingleQubit(self.experiment_cfg, self.hardware_cfg,self.quantum_device_cfg, sequences, name)
        # except: print("Not connected to keysight PXI")

        try: self.drive_los = [im[lo] for lo in self.hardware_cfg['drive_los']]
        except: print ("No drive function generator specified in hardware config")

        try: self.readout_los = [im[lo] for lo in self.hardware_cfg['readout_los']]
        except: print ("No readout function generator specified in hardware config")

        try: self.attens = [im[atten] for atten in self.hardware_cfg['attens']]
        except: print ("No digital attenuator specified in hardware config")

        try: self.trig = im['trig']
        except: print ("No trigger function generator specied in hardware cfg")

        try: self.flux1 = im[self.hardware_cfg['dc_flux']]
        except: print("No dc flux connected")

        try: self.sc = SignalCore(name="SignalCore",address="10001E48")
        except: print("No Signal Core generator connected")

        try: self.tek2 = im['TEK2']
        except: print("No tek2")

        # self.awg81180=AWG81180A (name='awg',address="192.168.14.134:5025",query_sleep=0)
        try: self.awg81180=AWG81180A (name='awg',address="192.168.14.134:5025",query_sleep=0)
        except: print("No AWG81180A")
        
        self.I = None
        self.Q = None
        self.prep_tek2 = False


    def initiate_pxi(self, name, sequences):
        try:self.tek2.stop()
        except:pass
        try:
            self.pxi.AWG_module.stopAll()
            self.pxi.m_module.stopAll()
            self.pxi.trig_module.stopAll()
        except:pass

        pxi_waveform_channels = self.hardware_cfg['awg_info']['keysight_pxi']['waveform_channels']
        pxi_sequences = {}
        for channel in pxi_waveform_channels:
            pxi_sequences[channel] = sequences[channel]
        try:
            self.pxi.configureChannels(self.hardware_cfg, self.experiment_cfg, name)
            self.pxi.loadAndQueueWaveforms(pxi_sequences)
        except:print("Error in configuring and loading sequences to PXI")

    def initiate_tek2(self, name,path, sequences):
        try:
            print("Connected to", self.tek2.get_id())
            tek2_waveform_channels = self.hardware_cfg['awg_info']['tek70001a']['waveform_channels']
            tek2_waveforms = [sequences[channel] for channel in tek2_waveform_channels]
            for waveform in tek2_waveforms:
                write_Tek70001_sequence(waveform,os.path.join(path, 'sequences/'), name,awg=self.tek2)
            self.tek2.prep_experiment()
        except:print("tek2 sequence not uploaded")

    def initiate_awg81180(self,sequences):
        # print(self.awg81180.get_id())
        print("Changing arb waveform")
        self.awg81180.set_output(False)
        self.awg81180.select_channel(1)
        self.awg81180.set_mode("USER")    
        # self.awg81180.set_amplitude(2.0)
        self.awg81180.set_offset(0.0)
        self.awg81180.delete_all()
        self.awg81180.select_sequence(1)
        awg81180_channels = self.hardware_cfg['awg_info']['awg81180']['waveform_channels']
        awg81180_waveforms = [sequences[channel] for channel in awg81180_channels]

        # arr=[]
        # print("Calculating waveform")
        # for i in range(20):
        #     b=zeros(100000)
        #     for j in range(i*5000):
        #         b[j]=j
        #     arr.append(b)
        print(len(awg81180_waveforms[0]))
        print("Convert to integer waveform")
        idata=[self.awg81180.convert_float_to_int_data(seg,float_range=[-1.0,1.0]) for seg in awg81180_waveforms[0]]    
        print("Uploading waveform")
        for ii,seg in enumerate(idata):
            self.awg81180.add_intsegment(seg,segnum=ii+1)
            time.sleep(1.0)
            self.awg81180.define_sequence_step(ii+1,ii+1)
            time.sleep(2.0)

        self.awg81180.set_to_sequence()
        self.awg81180.set_to_trace()
        self.awg81180.select_channel(1)
        self.awg81180.set_output(True)
        self.awg81180.select_channel(2)
        self.awg81180.set_output(False)
        self.awg81180.define_sequence_advance()

        # self.awg81180.define_sequence_step(step=1,seg_num=len(idata),loops=1,jump_flag=False)
        time.sleep(1)
        print("Finished uploading data.")

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

    def load_PXDAC4800_sequence(self, sequences, path, name, brdNum):
        pxdac_waveform_channels_num = self.hardware_cfg['awg_info']['pxdac' + str(brdNum)]['channels_num']
        pxdac_waveform_channels = self.hardware_cfg['awg_info']['pxdac' + str(brdNum)]['waveforms']
        pxdac_waveforms = []
        for channel in pxdac_waveform_channels:
            if not channel == None:
                pxdac_waveforms.append(sequences[channel['name']])
            else:
                pxdac_waveforms.append(np.zeros_like(sequences[pxdac_waveform_channels[0]['name']]))
        offset_bytes_list = write_PXDAC4800_file(pxdac_waveforms, os.path.join(path, name + '_%d.rd16' % brdNum),
                                                 name,
                                                 self.hardware_cfg['awg_info']['pxdac'+str(brdNum)]['iq_offsets_bytes'], self.hardware_cfg['awg_info']['pxdac'+str(brdNum)]['sample_size'])

        pxdac4800 = LocalInstruments().inst_dict['pxdac4800_%d' % brdNum]
        pxdac4800.load_sequence_file(os.path.join(path, name + '_%d.rd16' % brdNum), self.hardware_cfg['awg_info']['pxdac'+str(brdNum)])
        print("Sequence file uploaded")
        print("Waveform length: " + str(len(pxdac_waveforms[0][0])))
        pxdac4800.waveform_length = len(pxdac_waveforms[0][0])
        print("PXDAC4800 waveform length: " + str(pxdac4800.waveform_length))
        pxdac4800.run_experiment()


    def initiate_pxdac(self, name, path, sequences):
        for BrdNum in range(2,3,1):
            self.load_PXDAC4800_sequence(sequences, path, name, BrdNum)


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

    def awg_prep_pxdac(self):
        stop_pulseblaster()
        # LocalInstruments().inst_dict['pxdac4800_1'].stop()
        LocalInstruments().inst_dict['pxdac4800_2'].stop()
        try:self.tek2.stop()
        except:print('Error in stopping TEK2')
        # self.awg81180.stop()
        # time.sleep(10)

    def awg_run(self,run_pxi = True,name=None):
        if run_pxi:
            if 'sideband' in name:
                try:self.tek2.run()
                except:print("tek2 is not runnning")
            self.pxi.run()
        else:
            self.m8195a.start_output()
            time.sleep(1)
            self.tek.run()

    def awg_run_pxdac(self, name=None):
        # LocalInstruments().inst_dict['pxdac4800_1'].run_experiment()
        LocalInstruments().inst_dict['pxdac4800_2'].run_experiment()
        # self.awg81180.run()
        try:self.tek2.run()
        except:print("tek2 is not runnning")
        # time.sleep(10)
        run_pulseblaster()

    def awg_stop(self,name):
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
        if 'sideband' in name:
            try:self.tek2.stop()
            except:print('Error in stopping TEK2')

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
            self.quantum_device_cfg['readout']['length'] - 1).bit_length()
        self.hardware_cfg['alazar']['recordsPerBuffer'] = sequence_length
        self.hardware_cfg['alazar']['recordsPerAcquisition'] = int(
            sequence_length * min(averages, 500))
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

    def initiate_readout_LOs(self, freq=0):
        try:
            
            for ii, d in enumerate(self.readout_los):
                d.set_frequency((self.quantum_device_cfg['readout']['freq']+freq)*1e9)
                print ("Readout frequency = ",(self.quantum_device_cfg['readout']['freq']+freq),"GHz")
                d.set_power(self.quantum_device_cfg['readout_drive_lo_powers'][str(ii + 1)])
                d.set_ext_pulse(mod=True)
        except:print("Error in readout drive LO configuration")

    def initiate_signalcore(self, delta_freq = 0):
        self.sc.set_output_state(enable=True)
        self.sc.set_frequency(freq = (self.quantum_device_cfg['qubit']['1']['freq2'] - self.quantum_device_cfg['pulse_info']['1']['iq_freq2'] + delta_freq)*1e9, acknowledge = True)
        self.sc.set_power(pdBm = -15)

    def initiate_attenuators(self, atten = 0):
        try:
            for ii, d in enumerate(self.attens):
                d.set_attenuator(self.quantum_device_cfg['readout']['dig_atten']+atten)
                print("digi_atten set to", self.quantum_device_cfg['readout']['dig_atten']+atten)
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
        print('flux ramp started')
        time.sleep(5)
        print('flux ramp ended')

    def initiate_flux_pxdac(self,offset_current = 0):
        self.flux1.set_output(True)
        self.flux1.ramp_current((self.quantum_device_cfg['freq_flux']['1']['current_mA'] + offset_current) * 1e-3)
        print('flux ramp started')
        time.sleep(20)
        print('flux ramp ended')

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
                self.adc.acquire_singleshot_heterodyne_multitone_data_2(het_IFreqList, prep_function=self.awg_prep_pxdac,
                        start_function=self.awg_run_pxdac,excise=self.quantum_device_cfg['readout']['window'])
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
                                                                                self.expt_cfg.get('pi_calibration', False))
            data_2_cos_list, data_2_sin_list, data_2_list = get_singleshot_data(single_data2, 1,
                                                                                self.expt_cfg.get('pi_calibration', False))
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
        self.awg_prep_pxdac()

    def get_histogram_data_pxdac(self, sequence_length, acquisition_num, data_file, seq_data_file):

        len_exp = sequence_length
        num_bins = self.expt_cfg['numbins']
        ss_data1 = np.zeros((len_exp, num_bins))
        sss_data1 = np.zeros((len_exp, num_bins))
        ss_data2 = np.zeros((len_exp, num_bins))
        sss_data2 = np.zeros((len_exp, num_bins))
        ss_data = np.zeros((len_exp, num_bins,num_bins))
        total1 = np.zeros(len_exp)
        total2 = np.zeros(len_exp)
        all_data1 = np.zeros((len_exp,acquisition_num))
        all_data2 = np.zeros((len_exp,acquisition_num))
        ss_rotate = np.zeros((len_exp,acquisition_num))
        data_ang = np.zeros((len_exp,acquisition_num))
        histor = np.zeros((len_exp, num_bins))
        binsr = np.zeros((len_exp, num_bins+1))

        recordsnum = int(min(acquisition_num,500))
        numAcquisition = int(np.ceil(acquisition_num / recordsnum))
        ifrange=0

        for ii in tqdm(np.arange(numAcquisition)):
        # ss1, ss2 = adc.acquire_singleshot_data(prep_function=None, start_function=None,
        #                                        excise=self.cfg['readout']['window'])
            ss1, ss2 = self.adc.acquire_singleshot_data(prep_function=self.awg_prep_pxdac, start_function=self.awg_run_pxdac,
                                           excise=self.quantum_device_cfg['readout']['window'])

            ss11 = np.reshape(ss1, (recordsnum , len_exp)).T
            ss22 = np.reshape(ss2, (recordsnum , len_exp)).T

            if ifrange == 0:
                ew1 = max(np.abs(ss11.min()),np.abs(ss11.max()))
                ew2 = max(np.abs(ss22.min()),np.abs(ss22.max()))
                histo_range1 = [ss11.min() - ew1 * 0.5, ss11.max() + ew1 * 0.5]
                histo_range2 = [ss22.min() - ew2 * 0.5, ss22.max() + ew2 * 0.5]
                ifrange = 1

            for jj, ss in enumerate(ss11):
                sshisto1, ssbins1 = np.histogram(ss, bins=num_bins, range=histo_range1)
                ss_data1[jj] += sshisto1
                sss_data1[jj] = np.cumsum(ss_data1[[jj]])
                all_data1[jj][ii*500:(ii+1)*500] = ss

            for jj, ss in enumerate(ss22):
                sshisto2, ssbins2 = np.histogram(ss, bins=num_bins, range=histo_range2)
                ss_data2[jj] += sshisto2
                sss_data2[jj] = np.cumsum(ss_data2[[jj]])
                all_data2[jj][ii*500:(ii+1)*500] = ss

            for jj in range(len(ss11)):
                ss2d1 = np.array(ss11[jj])
                ss2d2 = np.array(ss22[jj])
                # print(ss2d1.shape)
                # print(ss2d2.shape)
                sshisto, ssbinsx, ssbinsy = np.histogram2d(ss2d1, ss2d2, bins=num_bins, range=np.array([histo_range1,histo_range2]))
                ss_data[jj] += sshisto
                
            for jj in range(len(ss11)):
                total1[jj] += ss11[jj].sum()
                total2[jj] += ss22[jj].sum()
            
        for jj in range(len_exp):
            total1[jj] = total1[jj]/acquisition_num
            total2[jj] = total2[jj]/acquisition_num

        # rotate to gf
        len_gf = np.sqrt((total1[2]-total1[0])**2+(total2[2]-total2[0])**2)
        rotate_ang = np.arctan((total2[2]-total2[0])/(total1[2]-total1[0]))
        rotate_offset = (total1[0]*total2[2]-total2[0]*total1[2])/(total2[2]-total2[0])

        for jj in range(len_exp):
            data_ang[jj] = np.arctan(all_data2[jj]/(all_data1[jj]-rotate_offset))
            proj_ang = rotate_ang - data_ang[jj]

            ss_rotate[jj] = np.sqrt((all_data1[jj]-rotate_offset)**2+all_data2[jj]**2)*np.cos(proj_ang)
        
            if jj ==0:
                ewr = max(np.abs(ss_rotate[jj].min()),np.abs(ss_rotate[jj].max()))
                histo_ranger = [ss_rotate[jj].min() - ewr * 0.25, ss_rotate[jj].max() + ewr * 0.25]

            histor[jj], binsr[jj] = np.histogram(ss_rotate[jj], bins=num_bins, range=histo_ranger)

        eh_contrast_data1 = abs(((sss_data1[3] - sss_data1[1]) / ss_data1[3].sum())).max()
        eh_contrast_data2 = abs(((sss_data2[3] - sss_data2[1]) / ss_data2[3].sum())).max()
        gf_contrast_data1 = abs(((sss_data1[0] - sss_data1[2]) / ss_data1[2].sum())).max()
        gf_contrast_data2 = abs(((sss_data2[0] - sss_data2[2]) / ss_data2[2].sum())).max()

        if not seq_data_file == None:
            self.slab_file = SlabFile(data_file)
            with self.slab_file as f:
                # f.append_pt('atten', atten)
                # f.append_line('freq', freqpts)
                f.append('ssdata1', ss_data1)
                f.append_line('ssbins1', ssbins1)
                f.append_pt('eh_contrast_data1', eh_contrast_data1)
                f.append_pt('eh_contrast_data2', eh_contrast_data2)
                f.append('ssdata2', ss_data2)
                f.append_line('ssbins2', ssbins2)
                f.append_pt('gf_contrast_data1', gf_contrast_data1)
                f.append_pt('gf_contrast_data2', gf_contrast_data2)
                f.append('ssdata', ss_data)
                f.append_line('ssbinsx', ssbinsx)
                f.append_line('ssbinsy', ssbinsy)
                f.append('total1', total1)
                f.append('total2', total2)
                f.append('all_data1', all_data1)
                f.append('all_data2', all_data2)
                f.append('rotate_ang', rotate_ang)
                f.append('rotate_offset', rotate_offset)
                f.append('histor', histor)
                f.append('binsr', binsr)
                
                f.close()

        # self.adc.close()
        self.awg_prep_pxdac()

    def get_avg_data_alazar(self, acquisition_num, data_file, seq_data_file):
        expt_data_ch1 = None
        expt_data_ch2 = None
        for ii in tqdm(np.arange(max(1, int(acquisition_num / 100)))):
            tpts, ch1_pts, ch2_pts = self.adc.acquire_avg_data_by_record(prep_function=self.awg_prep,
                                                                         start_function=self.awg_run,
                                                                         excise=
                                                                         self.quantum_device_cfg['readout'][
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
        self.awg_prep_pxdac()
        if not seq_data_file == None:
            self.slab_file = SlabFile(data_file)
            with self.slab_file as f:
                f.append_line('expt_avg_data_ch1', data_1_list)
                f.append_line('expt_avg_data_ch2', data_2_list)
                f.close()

    def get_avg_data_pxdac(self, acquisition_num, data_file, seq_data_file):
        expt_data_ch1 = None
        expt_data_ch2 = None
        for ii in tqdm(np.arange(max(1, int(acquisition_num / 100)))):
            tpts, ch1_pts, ch2_pts = self.adc.acquire_avg_data_by_record(prep_function=self.awg_prep_pxdac,
                                                                         start_function=self.awg_run_pxdac,
                                                                         excise=
                                                                         self.quantum_device_cfg['readout'][
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
        self.awg_prep_pxdac()
        if not seq_data_file == None:
            self.slab_file = SlabFile(data_file)
            with self.slab_file as f:
                f.append_line('expt_avg_data_ch1', data_1_list)
                f.append_line('expt_avg_data_ch2', data_2_list)
                f.close()

    def get_avg_data_pxdac_old(self, acquisition_num, data_file, seq_data_file, name = 'pxdac', adc_close = True):

        expt_data1 = None
        expt_data2 = None
        current_data1 = None
        current_data2 = None
        for ii in tqdm(np.arange(max(1, int(acquisition_num / 500)))):
            tpts, ch1_pts, ch2_pts = self.adc.acquire_avg_data_by_record(prep_function=self.awg_prep_pxdac,
                                                                    start_function=self.awg_run_pxdac,
                                                                         excise=self.quantum_device_cfg['readout']['window'])

            mag = np.sqrt(ch1_pts ** 2 + ch2_pts ** 2)
            # if not self.expt_cfg.get('pi_calibration',False):
            if expt_data1 is None:
                expt_data1 = ch1_pts
                expt_data2 = ch2_pts
            else:
                expt_data1 = (expt_data1 * ii + ch1_pts) / (ii + 1.0)
                expt_data2 = (expt_data2 * ii + ch2_pts) / (ii + 1.0)

            # else:
            #     zero_amp1 = np.mean(ch1_pts[-2])
            #     pi_amp1 = np.mean(ch1_pts[-1])
            #     current_data1 = (ch1_pts[:-2] - zero_amp) / (pi_amp - zero_amp)

            #     zero_amp2 = np.mean(ch2_pts[-2])
            #     pi_amp2 = np.mean(ch2_pts[-1])
            #     current_data2 = (ch2_pts[:-2] - zero_amp) / (pi_amp - zero_amp)

            #     if expt_data1 is None:
            #         expt_data1 = current_data1
            #         expt_data2 = current_data2
            #     else:
            #         expt_data1 = (expt_data1 * ii + current_data1) / (ii + 1.0)
            #         expt_data2 = (expt_data2 * ii + current_data2) / (ii + 1.0)

            expt_avg_data1 = np.mean(expt_data1, 1)
            expt_avg_data2 = np.mean(expt_data2, 1)

        if seq_data_file == None:
            self.slab_file = SlabFile(self.data_file)
            with self.slab_file as f:
                f.add('expt_2d1', expt_data1)
                f.add('expt_avg_data1', expt_avg_data1)
                f.add('expt_2d2', expt_data2)
                f.add('expt_avg_data2', expt_avg_data2)
                # f.add('expt_pts', self.expt_pts)
                f.close()
            # self.adc.close()

        if adc_close == True:
            self.adc.close()

        self.awg_prep_pxdac()
        if not seq_data_file == None:
            self.slab_file = SlabFile(data_file)
            with self.slab_file as f:
                f.append_line('expt_avg_data1', expt_avg_data1)
                f.append_line('expt_avg_data2', expt_avg_data2)
                f.close()

    def get_wf_data_pxdac(self, acquisition_num, data_file, seq_data_file, name = 'pxdac', adc_close = True, sequence_length = 100):

        window_length = self.quantum_device_cfg['readout']['window'][1] - self.quantum_device_cfg['readout']['window'][0]
        expt_data1 = np.zeros((max(1, int(acquisition_num / 500)),window_length,500*sequence_length))
        expt_data2 = np.zeros((max(1, int(acquisition_num / 500)),window_length,500*sequence_length))

        for ii in tqdm(np.arange(max(1, int(acquisition_num / 500)))):
            
            tpts, ch1_pts, ch2_pts = self.adc.acquire_singleshot_data2(prep_function=self.awg_prep_pxdac, 
                start_function=self.awg_run_pxdac, excise=self.quantum_device_cfg['readout']['window'])

            expt_data1[ii] = ch1_pts
            expt_data2[ii] = ch2_pts

        if seq_data_file == None:
            self.slab_file = SlabFile(self.data_file)
            with self.slab_file as f:
                f.add('expt_2d1', expt_data1)
                # f.add('expt_avg_data1', expt_avg_data1)
                f.add('expt_2d2', expt_data2)
                # f.add('expt_avg_data2', expt_avg_data2)
                # f.add('expt_pts', self.expt_pts)
                f.close()
            # self.adc.close()

        if adc_close == True:
            self.adc.close()

        self.awg_prep_pxdac()
        if not seq_data_file == None:
            self.slab_file = SlabFile(data_file)
            with self.slab_file as f:
                f.append_line('expt_2d1', expt_data1)
                f.append_line('expt_2d2', expt_data2)
                f.close()


    def get_avg_data_pxi(self,expt_cfg, seq_data_file):
        w = self.pxi.readout_window/self.pxi.dt_dig
        # expt_pts = np.arange(expt_cfg['start'],expt_cfg['stop'],expt_cfg['step'])

        try:pi_calibration = expt_cfg['pi_calibration']
        except:pi_calibration = False

        I,Q = self.pxi.acquire_avg_data(w,pi_calibration)
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
        self.initiate_drive_LOs()
        self.initiate_readout_LOs()
        self.initiate_attenuators()
        self.initiate_pxi(name, sequences)
        self.initiate_tek2(name,path,sequences)
        time.sleep(0.1)
        self.awg_run(run_pxi=True,name=name)

        try:
            if check_sync:self.pxi.acquireandplot(expt_num)
            else:
                if self.expt_cfg['singleshot']:
                    self.I,self.Q =  self.get_ss_data_pxi(self.expt_cfg,seq_data_file=seq_data_file)
                else:
                    self.I,self.Q = self.get_avg_data_pxi(self.expt_cfg,seq_data_file=seq_data_file)
        except:print("Error in data acquisition from PXI")

        self.awg_stop(name)
        return self.I,self.Q

    def run_experiment_pxdac(self, sequences, path, name, seq_data_file=None, update_awg=True):
        exp_period_ns = self.hardware_cfg['trigger']['period_us']*1000
        self.expt_name = name
        # self.initiate_awg81180(sequences)
        self.initiate_flux_pxdac()
        self.initiate_drive_LOs()
        self.initiate_readout_LOs()
        self.initiate_attenuators()
        # self.initiate_signalcore()

        sequence_length = len(sequences['charge1_I'])

        start_pulseblaster(exp_period_ns)

        if update_awg:
            self.initiate_pxdac(name, os.path.join(path, 'sequences\\'), sequences)
            self.initiate_tek2(name,path,sequences)
            time.sleep(1)

        self.expt_cfg = self.experiment_cfg[name]
        acquisition_num = self.expt_cfg['acquisition_num']

        self.initiate_alazar(sequence_length, acquisition_num)
        self.generate_datafile(path, name, seq_data_file)

        if name == 'ff_histogram' or name == 'ff_ss_histogram':
            self.get_histogram_data_pxdac(sequence_length, acquisition_num, self.data_file, seq_data_file)
        elif self.expt_cfg.get('singleshot', True):
            self.get_wf_data_pxdac(acquisition_num, self.data_file, seq_data_file, name = 'pxdac', adc_close = True, sequence_length = sequence_length)
        else:
            self.get_avg_data_pxdac_old(acquisition_num, self.data_file, seq_data_file)

        return self.data_file

    def run_experiment_seq_sweep(self, sequences, path, name, seq_data_file=None, update_awg=True, flux = 0, atten = 0, freq = 0, adc_close = False, tperiod = 0, flux_update=True):
        self.expt_name = name

        if name == 'ff_long_T1':
            close_pulseblaster()
            time.sleep(1.0)
            start_pulseblaster(tperiod*1000)
            pbon = True
        else:
            exp_period_ns = self.hardware_cfg['trigger']['period_us']*1000
            pbon = False

        # time.sleep(2)

        self.expt_cfg = self.experiment_cfg[name]
        acquisition_num = self.expt_cfg['acquisition_num']

        sequence_length = len(sequences['charge1_I'])

        if flux_update == True:
            self.initiate_flux_pxdac(offset_current=flux)
        
        # self.initiate_readout_LOs(freq)
        # self.initiate_attenuators(atten)
        # time.sleep(3.0)

        if update_awg:
            # self.initiate_flux_pxdac(offset_current=flux)
            # self.initiate_awg81180(sequences)
            # self.initiate_drive_LOs()
            self.initiate_readout_LOs()
            self.initiate_attenuators()
            self.initiate_tek2(name,path,sequences)
            time.sleep(1)

            if pbon == False:
                start_pulseblaster(exp_period_ns)

            self.initiate_pxdac(name, os.path.join(path, 'sequences\\'), sequences)
            self.initiate_alazar(sequence_length, acquisition_num)
            self.generate_datafile(path, name, seq_data_file)

        if name == 'ff_histogram' or name == 'ff_ss_histogram':
            self.get_histogram_data_pxdac(sequence_length, acquisition_num, self.data_file, seq_data_file)
        elif self.expt_cfg.get('singleshot', True):
            self.get_singleshot_data_alazar(sequence_length, acquisition_num, self.data_file, seq_data_file)
        else:
            self.get_avg_data_pxdac_old(acquisition_num, self.data_file, seq_data_file,name, adc_close= adc_close)

        return self.data_file

    def run_experiment_alazar_rspectr(self, sequences, path, name, seq_data_file=None, update_awg=True, freq=0):
        exp_period_ns = self.hardware_cfg['trigger']['period_us']*1000

        # self.initiate_flux_pxdac()
        # self.initiate_drive_LOs()
        self.initiate_readout_LOs(freq)
        self.initiate_attenuators()

        sequence_length = 1

        self.expt_cfg = self.experiment_cfg[name]
        acquisition_num = self.expt_cfg['acquisition_num']

        if freq==self.expt_cfg['start']:
            start_pulseblaster(exp_period_ns)
            self.initiate_pxdac(name, os.path.join(path, 'sequences\\'), sequences)
            self.initiate_alazar(sequence_length, acquisition_num)
            self.generate_datafile(path, name, seq_data_file)

        if self.expt_cfg.get('singleshot', True):
            self.get_singleshot_data_alazar(sequence_length, acquisition_num, self.data_file, seq_data_file)
        else:
            self.get_avg_data_pxdac_old(acquisition_num, self.data_file, seq_data_file,name, adc_close=False)

        time.sleep(2)

        return self.data_file

    def post_analysis(self,experiment_name,P='Q',show = False,check_sync = False):
        # if check_sync:pass
        # else:PA = PostExperiment(self.quantum_device_cfg, self.experiment_cfg, self.hardware_cfg, experiment_name, self.I ,self.Q, P,show)
        pass