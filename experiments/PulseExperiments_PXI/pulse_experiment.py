from slab import InstrumentManager
from slab.instruments.awg import write_Tek5014_file
from slab.instruments.awg.M8195A import upload_M8195A_sequence
# import keysight_pxi_load as ks_pxi
from slab.instruments.keysight import keysight_pxi_load as ks_pxi
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
from slab.experiments.PulseExperiments_PXI.get_data import get_iq_data, get_singleshot_data
from slab.experiments.PulseExperiments_PXI.PostExperimentAnalysis import PostExperiment
from slab.experiments.PulseExperiments_PXI.PostExperimentAnalysis import PostExperimentAnalyze
from slab.experiments.PulseExperiments_PXI.PostExperimentAnalysis import PostExperimentAnalyzeAndSave

class Experiment:
    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg,sequences=None, name=None):
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg
        im = InstrumentManager()
        self.qubits = ["A", "B"]
        import time
        #self.fluxbias = im['dacbox']
        #self.fluxbias.setvoltage(1,0)
        time.sleep(1)

        self.pxi =  ks_pxi.KeysightSingleQubit(self.experiment_cfg, self.hardware_cfg,self.quantum_device_cfg, sequences, name)

        try: self.drive_los = [im[lo] for lo in self.hardware_cfg['drive_los']]
        except: print ("No drive function generator specified in hardware config / failure to connect with im()")

        try: self.stab_los = [im[lo] for lo in self.hardware_cfg['stab_los']]
        except: print ("No stabilizer function generator specified in hardware config / failure to connect with im()")

        try: self.readout_los = [im[lo] for lo in self.hardware_cfg['readout_los']]
        except: print ("No readout function generator specified in hardware config / failure to connect with im()")

        try: self.readout_attens = [im[atten] for atten in self.hardware_cfg['readout_attens']]
        except: print ("No digital attenuator specified in hardware config / failure to connect with im()")

        try:self.drive_attens = [im[atten] for atten in self.hardware_cfg['drive_attens']]
        except:print("No digital attenuator specified in hardware config / failure to connect with im()")

        try: self.trig = im['triggrb']
        except: print ("No trigger function generator specied in hardware cfg / failure to connect with im()")

        self.data = None


    def initiate_pxi(self, name, sequences):
        try:
            for out_module in self.pxi.out_mods:
                out_module.stopAll()
                out_module.clearAll()
        except:pass
        pxi_waveform_channels = self.hardware_cfg['awg_info']['keysight_pxi']['waveform_channels']
        pxi_sequences = {}
        for channel in pxi_waveform_channels:
            pxi_sequences[channel] = sequences[channel]
        self.pxi.configureChannels(self.hardware_cfg, self.experiment_cfg, self.quantum_device_cfg, name)
        print('configureOK')
        self.pxi.loadAndQueueWaveforms(pxi_sequences)
        print('LoadandConfigureOK')

    def pxi_stop(self):
        try:
            for out_module in self.pxi.out_mods:
                out_module.stopAll()
                out_module.clearAll()
            self.pxi.DIG_module.stopAll()
            self.pxi.chassis.close()
        except:print('Error in stopping and closing PXI')

    # now setting for SignalCore
    def initiate_drive_LOs(self):
        try:
            #TODO this totally relieson order of drive LOs being A first, then B, etc. this should be a dictionary
            # somehow but I am currently pretty out of it so will have to wait until later - G 5/5/21
            for ii,d in enumerate(self.drive_los):
                drive_freq = self.quantum_device_cfg['qubit'][self.qubits[ii]]['freq'] - self.quantum_device_cfg[
                'pulse_info'][self.qubits[ii]]['iq_freq']
                d.set_frequency(drive_freq * 1e9)
                d.set_clock_reference(ext_ref=True)
                d.set_power(self.quantum_device_cfg['powers'][self.qubits[ii]]['drive_lo_powers'])
                d.set_output_state(True)
                d.set_rf_mode(val=0) # single RF tone on output 1
                d.set_standby(False)
                d.set_rf2_standby(True) # no output on RF 2
                rfparams = d.get_rf_parameters()
                time.sleep(0.2)
                settingparams = d.get_device_status()
                time.sleep(0.2)
                print(" ==== DRIVE LO SETTINGS ==== ")
                print("RF1 OUT ENABLED: %s"%settingparams.operate_status.rf1_out_enable)
                print("RF1 STANDBY: %s"%settingparams.operate_status.rf1_standby)
                print("RF1 EXT REF DETECTED: %s"%settingparams.operate_status.ext_ref_detect)
                print("RF1 FREQ: %s"%(rfparams.rf1_freq))
                print("RF1 LEVEL: %s"%(rfparams.rf_level))
        except:
            print ("Error in qubit drive LO configuration")
            raise

    def initiate_readout_LOs(self):
        try:
            for ii, d in enumerate(self.readout_los):
                readout_freq = self.quantum_device_cfg['readout'][self.qubits[ii]]['freq']*1e9
                d.set_frequency(readout_freq)
                d.set_clock_reference(ext_ref=True)
                d.set_power(self.quantum_device_cfg['powers'][self.qubits[ii]]['readout_drive_lo_powers'])
                d.set_output_state(True)
                d.set_rf_mode(val=0) # single RF tone on output 1
                d.set_standby(False)
                d.set_rf2_standby(True) # no output on RF 2
                rfparams = d.get_rf_parameters()
                settingparams = d.get_device_status()
                print(" ==== READOUT LO SETTINGS ==== ")
                print("RF1 OUT ENABLED: %s"%settingparams.operate_status.rf1_out_enable)
                print("RF1 STANDBY: %s"%settingparams.operate_status.rf1_standby)
                print("RF1 EXT REF DETECTED: %s"%settingparams.operate_status.ext_ref_detect)
                print("RF1 FREQ: %s"%(rfparams.rf1_freq))
                print("RF1 LEVEL: %s"%(rfparams.rf_level))
        except:
            print("Error in readout READOUT LO configuration")
            raise

    def initiate_stab_LOs(self):
        try:
            for ii, d in enumerate(self.stab_los):
                stab_freq = self.quantum_device_cfg['stabilizer_info']['freq']*1e9
                d.set_frequency(stab_freq)
                d.set_clock_reference(ext_ref=True)
                d.set_power(self.quantum_device_cfg['stabilizer_info']["stab_lo_power"])
                d.set_output_state(True)
                d.set_rf_mode(val=0) # single RF tone on output 1
                d.set_standby(False)
                d.set_rf2_standby(True) # no output on RF 2
                rfparams = d.get_rf_parameters()
                settingparams = d.get_device_status()
                print(" ==== STABILIZER LO SETTINGS ==== ")
                print("RF1 OUT ENABLED: %s"%settingparams.operate_status.rf1_out_enable)
                print("RF1 STANDBY: %s"%settingparams.operate_status.rf1_standby)
                print("RF1 EXT REF DETECTED: %s"%settingparams.operate_status.ext_ref_detect)
                print("RF1 FREQ: %s"%(rfparams.rf1_freq))
                print("RF1 LEVEL: %s"%(rfparams.rf_level))

        except:
            print("Error in readout STABILIZER LO configuration")
            raise

    def initiate_readout_attenuators(self):
        try:
            for ii, d in enumerate(self.readout_attens):
                d.set_attenuator(self.quantum_device_cfg['powers'][self.qubits[ii]]['readout_drive_digital_attenuation'])
                print("set readout attenuator")
        except:
            print("Error in readout digital attenuator configuration")

    def initiate_drive_attenuators(self):
        try:
            for ii, d in enumerate(self.drive_attens):
                d.set_attenuator(self.quantum_device_cfg['powers'][self.qubits[ii]]['drive_digital_attenuation'])
                print("set drive attenuator")
        except:
            print("Error in qubit drive attenuator configuration")


    def set_trigger(self):
        try:
            period = self.hardware_cfg['trigger']['period_us']
            self.trig.set_period(period*1e-6)
            print ("Trigger period set to ", period,"us")
        except:
            print("Error in trigger configuration")

    def save_cfg_info(self, f):
        f.attrs['quantum_device_cfg'] = json.dumps(self.quantum_device_cfg)
        f.attrs['experiment_cfg'] = json.dumps(self.experiment_cfg)
        f.attrs['hardware_cfg'] = json.dumps(self.hardware_cfg)
        f.close()

    def generate_datafile(self,path,name,seq_data_file = None):
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

    def get_avg_data_pxi(self,expt_cfg, name, seq_data_file):
        try:pi_calibration = expt_cfg['pi_calibration']
        except:pi_calibration = False

        data = self.pxi.acquire_avg_data(pi_calibration)
        if seq_data_file == None:
            self.slab_file = SlabFile(self.data_file)
            with self.slab_file as f:
                if "A" in self.experiment_cfg[name]["on_qubits"] and "B" in self.experiment_cfg[name]["on_qubits"]:
                    f.add('qbA_I', data[0][0])
                    f.add('qbA_Q', data[0][1])
                    f.add('qbB_I', data[1][0])
                    f.add('qbB_Q', data[1][1])
                else:
                    f.add('I', data[0][0])
                    f.add('Q', data[0][1])
        else:
            self.slab_file = SlabFile(seq_data_file)
            with self.slab_file as f:
                if "A" in self.experiment_cfg[name]["on_qubits"] and "B" in self.experiment_cfg[name]["on_qubits"]:
                    f.append_line('qbA_I', data[0][0])
                    f.append_line('qbA_Q', data[0][1])
                    f.append_line('qbB_I', data[1][0])
                    f.append_line('qbB_Q', data[1][1])
                else:
                    f.append_line('I', data[0][0])
                    f.append_line('Q', data[0][1])
        return data

    def get_ss_data_pxi(self,expt_cfg, name, seq_data_file):
        data = self.pxi.SSdata_many()
        if seq_data_file == None:
            self.slab_file = SlabFile(self.data_file)
            with self.slab_file as f:
                if "A" in self.experiment_cfg[name]["on_qubits"] and "B" in self.experiment_cfg[name]["on_qubits"]:
                    f.add('qbA_I', data[0][0])
                    f.add('qbA_Q', data[0][1])
                    f.add('qbB_I', data[1][0])
                    f.add('qbB_Q', data[1][1])
                else:
                    f.add('I', data[0][0])
                    f.add('Q', data[0][1])
        else:
            self.slab_file = SlabFile(seq_data_file)
            with self.slab_file as f:
                f.append_line('I', I.flatten())
                f.append_line('Q', Q.flatten())
                if "A" in self.experiment_cfg[name]["on_qubits"] and "B" in self.experiment_cfg[name]["on_qubits"]:
                    f.append_line('qbA_I', data[0][0].flatten())
                    f.append_line('qbA_Q', data[0][1].flatten())
                    f.append_line('qbB_I', data[1][0].flatten())
                    f.append_line('qbB_Q', data[1][1].flatten())
                else:
                    f.append_line('I', data[0][0].flatten())
                    f.append_line('Q', data[0][1].flatten())

        return data

    def get_traj_data_pxi(self,expt_cfg, name, seq_data_file):
        data = self.pxi.traj_data_many()
        if seq_data_file == None:
            self.slab_file = SlabFile(self.data_file)
            with self.slab_file as f:
                if "A" in self.experiment_cfg[name]["on_qubits"] and "B" in self.experiment_cfg[name]["on_qubits"]:
                    f.add('qbA_I', data[0][0])
                    f.add('qbA_Q', data[0][1])
                    f.add('qbB_I', data[1][0])
                    f.add('qbB_Q', data[1][1])
                else:
                    f.add('I', data[0][0])
                    f.add('Q', data[0][1])
        else:
            self.slab_file = SlabFile(seq_data_file)
            with self.slab_file as f:
                if "A" in self.experiment_cfg[name]["on_qubits"] and "B" in self.experiment_cfg[name]["on_qubits"]:
                    f.append_line('qbA_I', data[0][0])
                    f.append_line('qbA_Q', data[0][1])
                    f.append_line('qbB_I', data[1][0])
                    f.append_line('qbB_Q', data[1][1])
                else:
                    f.append_line('I', data[0][0])
                    f.append_line('Q', data[0][1])
        return data

    def get_traj_data_pxi_no_window(self,expt_cfg,seq_data_file):
        w=[0, self.hardware_cfg["awg_info"]["keysight_pxi"]["samplesPerRecord"]]
        data = self.pxi.traj_data_many(w=w)
        I= data[0][0]
        Q=data[0][1]
        I = np.average(I, axis=0)
        Q = np.average(Q, axis=0)
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
        return data

    def run_experiment_pxi(self, sequences, path, name, seq_data_file=None,update_awg=False,expt_num = 0,check_sync = False,save_errs = False):
        self.expt_cfg = self.experiment_cfg[name]
        self.generate_datafile(path,name,seq_data_file=seq_data_file)
        self.set_trigger()
        self.initiate_drive_LOs()
        self.initiate_readout_LOs()
        #self.initiate_stab_LOs()
        self.initiate_readout_attenuators()
        self.initiate_drive_attenuators()
        self.initiate_pxi(name, sequences)

        time.sleep(0.1)
        self.pxi.run()

        #TODO: not yet updated check_sync
        if check_sync:
            self.data = self.get_traj_data_pxi_no_window(self.expt_cfg, seq_data_file=seq_data_file)
            #I and Q in form of (avg_num, num_expt, sample_per_record)

        else:
            if self.expt_cfg['singleshot']:
                self.data =  self.get_ss_data_pxi(self.expt_cfg, name, seq_data_file=seq_data_file)
            elif self.expt_cfg['trajectory']:
                self.data = self.get_traj_data_pxi(self.expt_cfg, name, seq_data_file=seq_data_file)
            else:
                self.data = self.get_avg_data_pxi(self.expt_cfg, name, seq_data_file=seq_data_file)

        #
        self.pxi_stop()
        return self.data

    ##TODO: not yet updated post_analysis
    def post_analysis(self,path, experiment_name, cont_name, P='Q', phi=0):
        PA = PostExperimentAnalyze(self.quantum_device_cfg, self.experiment_cfg, self.hardware_cfg, path,
                                   experiment_name, self.data, P, phi)
        return PA.p

    def post_analysisandsave(self, path, experiment_name, cont_name, P='Q', phi=0, cont_data_file=None):
        PA = PostExperimentAnalyzeAndSave(self.quantum_device_cfg, self.experiment_cfg, self.hardware_cfg, path,
                                          experiment_name, self.data, P, phi, cont_data_file=cont_data_file,
                                          cont_name=cont_name)
        return PA.p

