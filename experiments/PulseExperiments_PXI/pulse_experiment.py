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
from slab.experiments.PulseExperiments_PXI.PostExperimentAnalysis import PostExperimentAnalyzeAndSave
import copy

class Experiment:
    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg,sequences=None, name=None):
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg
        im = InstrumentManager()
        self.setups = ["A", "B"]
        self.on_setups = self.quantum_device_cfg["setups"]
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
            for ii, s in enumerate(self.setups):
                d = self.drive_los[ii]
                if s in self.on_setups:
                    drive_freq =  self.quantum_device_cfg['qubit'][s]['freq'] - self.quantum_device_cfg[
                    'pulse_info'][s]['iq_freq']
                    d.set_frequency(drive_freq * 1e9)
                    d.set_power(self.quantum_device_cfg['powers'][s]['drive_lo_powers'])

                else:
                    d.set_frequency(self.hardware_cfg['lo_off_freq'] * 1e9)
                    d.set_power(self.hardware_cfg['lo_off_power'])

                d.set_clock_reference(ext_ref=True)
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
            for ii, s in enumerate(self.setups):
                d = self.readout_los[ii]
                if s in self.on_setups:
                    readout_freq = self.quantum_device_cfg['readout'][s]['freq'] * 1e9
                    d.set_frequency(readout_freq)
                    d.set_power(self.quantum_device_cfg['powers'][s]['readout_drive_lo_powers'])

                else:
                    d.set_frequency(self.hardware_cfg['lo_off_freq'] * 1e9)
                    d.set_power(self.hardware_cfg['lo_off_power'])

                d.set_clock_reference(ext_ref=True)
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
            for ii, s in enumerate(self.setups):
                d = self.readout_attens[ii]
                if s in self.on_setups:
                    d.set_attenuator(self.quantum_device_cfg['powers'][s]['readout_drive_digital_attenuation'])
                    print("set readout attenuator")
                else:
                    d.set_attenuator(30)
        except:
            print("Error in readout digital attenuator configuration")

    def initiate_drive_attenuators(self):
        try:
            for ii, s in enumerate(self.setups):
                d = self.drive_attens[ii]
                if s in self.on_setups:
                    d.set_attenuator(self.quantum_device_cfg['powers'][s]['drive_digital_attenuation'])
                    print("set drive attenuator")
                else:
                    d.set_attenuator(30)
                    print("set readout attenuator")

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
                if "A" in self.quantum_device_cfg["setups"] and "B" in self.quantum_device_cfg["setups"]:
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
                if "A" in self.quantum_device_cfg["setups"] and "B" in self.quantum_device_cfg["setups"]:
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
                if "A" in self.quantum_device_cfg["setups"] and "B" in self.quantum_device_cfg["setups"]:
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
                if "A" in self.quantum_device_cfg["setups"] and "B" in self.quantum_device_cfg["setups"]:
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
                if "A" in self.quantum_device_cfg["setups"] and "B" in self.quantum_device_cfg["setups"]:
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
                if "A" in self.quantum_device_cfg["setups"] and "B" in self.quantum_device_cfg["setups"]:
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
        data = self.pxi.traj_data_many_no_window(w=w)
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

    def run_experiment_pxi_resspec(self, sequences, path, name, seq_data_file=None,update_awg=False,expt_num = 0,check_sync = False,save_errs = False):
        data_path = os.path.join(path, 'data/')
        seq_data_file = os.path.join(data_path, get_next_filename(data_path, 'resonator_spectroscopy', suffix='.h5'))

        self.expt_cfg = self.experiment_cfg[name]
        self.generate_datafile(path,name,seq_data_file=seq_data_file)
        self.set_trigger()
        self.initiate_drive_LOs()
        #self.initiate_stab_LOs()
        self.initiate_readout_attenuators()
        self.initiate_drive_attenuators()
        self.initiate_pxi(name, sequences)
        self.initiate_readout_LOs()
        self.pxi.run()


        time.sleep(0.1)
        for qb in self.quantum_device_cfg["setups"]:
            read_freq = copy.deepcopy(self.quantum_device_cfg['readout'][qb]['freq'])
            for freq in np.arange(self.expt_cfg['start'] + read_freq, self.expt_cfg['stop'] + read_freq, self.expt_cfg['step']):
                self.quantum_device_cfg['readout'][qb]['freq'] = freq
                if self.expt_cfg['singleshot']:
                    self.data =  self.get_ss_data_pxi(self.expt_cfg, name, seq_data_file=seq_data_file)
                elif self.expt_cfg['trajectory']:
                    self.data = self.get_traj_data_pxi(self.expt_cfg, name, seq_data_file=seq_data_file)
                else:
                    self.data = self.get_avg_data_pxi(self.expt_cfg, name, seq_data_file=seq_data_file)


                self.pxi.DIG_module.stopAll()

                self.initiate_readout_LOs()

                self.pxi.configureDigChannels(self.hardware_cfg, self.experiment_cfg, self.quantum_device_cfg, name)
                self.pxi.DIG_ch_1.clear()
                self.pxi.DIG_ch_1.start()
                self.pxi.DIG_ch_2.clear()
                self.pxi.DIG_ch_2.start()
                self.pxi.DIG_ch_3.clear()
                self.pxi.DIG_ch_3.start()
                self.pxi.DIG_ch_4.clear()
                self.pxi.DIG_ch_4.start()
                time.sleep(0.1)

        #
        self.pxi_stop()
        return self.data

    def post_analysis(self, path, experiment_name, cont_name=None, P='Q', phi=0, cont_data_file=None):
        PA = PostExperimentAnalyzeAndSave(self.quantum_device_cfg, self.experiment_cfg, self.hardware_cfg, path,
                                          experiment_name, self.data, P, phi, cont_data_file=cont_data_file,
                                          cont_name=cont_name, save=False)
        return PA.p

    def post_analysisandsave(self, path, experiment_name, cont_name, P='Q', phi=0, cont_data_file=None):
        PA = PostExperimentAnalyzeAndSave(self.quantum_device_cfg, self.experiment_cfg, self.hardware_cfg, path,
                                          experiment_name, self.data, P, phi, cont_data_file=cont_data_file,
                                          cont_name=cont_name, save=True)
        return PA.p



def load_lattice_to_quantum_device(lattice_cfg_name, quantum_device_cfg_name, qb_id, setup_id):
    with open(quantum_device_cfg_name, 'r') as f:
        quantum_device_cfg = json.load(f)
    with open(lattice_cfg_name, 'r') as f:
        lattice_cfg = json.load(f)

    for category in quantum_device_cfg.keys():
        # check if category has "A" and "B" entries
        if isinstance(quantum_device_cfg[category], dict) and setup_id in quantum_device_cfg[category].keys():

            # if "A" and "B" are dictionaries where have to walk through keys
            if isinstance(quantum_device_cfg[category][setup_id], dict):
                for key in quantum_device_cfg[category][setup_id]:
                    try:
                        quantum_device_cfg[category][setup_id][key] = lattice_cfg[category][key][qb_id]
                    except:
                        print("[{}][{}] does not exist as a category and key that have as value a qubit list in "
                              "lattice config".format(category, key))

            # else just paste lattice into quantum_device [category][setup_id] directly
            else:
                try:
                    quantum_device_cfg[category][setup_id] = lattice_cfg[category][qb_id]
                except:
                    print("[{}] does not exist as a category that has as value a qubit listin lattice config".format(
                        category))

        # if category isn't a dictionary with setup_id, but has a matching list in lattice_cfg, fill it in
        else:
            if category in lattice_cfg.keys() and len(lattice_cfg[category]) == 8:
                quantum_device_cfg[category] = lattice_cfg[category][qb_id]

    with open(quantum_device_cfg_name, 'w') as f:
        json.dump(quantum_device_cfg, f, indent=2)


def load_quantum_device_to_lattice(lattice_cfg_name, quantum_device_cfg_name, qb_id, setup_id):
    with open(quantum_device_cfg_name, 'r') as f:
        quantum_device_cfg = json.load(f)
    with open(lattice_cfg_name, 'r') as f:
        lattice_cfg = json.load(f)

    for category in lattice_cfg.keys():
        # if category is directly a list of 8 qubit values, find the corresponsing entry in quantum device cfg and
        # stuff it in there
        if isinstance(lattice_cfg[category], list) and len(lattice_cfg[category]) == 8:

            # check if category even exists in quantum_device_config
            if category in quantum_device_cfg.keys():

                # check if needs to be stuffed into a setupid dict or just stuffed directly
                if isinstance(quantum_device_cfg[category], dict) and setup_id in quantum_device_cfg[category].keys():
                    lattice_cfg[category][qb_id] = quantum_device_cfg[category][setup_id]
                else:
                    lattice_cfg[category][qb_id] = quantum_device_cfg[category]

            else:
                print("[{}] not a category quantum device config".format(category))

        # if category is a dictionary, walk through the keys. if one of them is a list of eight qubit values,
        # stuff it in quantum device config
        elif isinstance(lattice_cfg[category], dict):
            for key in lattice_cfg[category]:
                if isinstance(lattice_cfg[category][key], list) and len(lattice_cfg[category][key]) == 8:

                    try:
                        if isinstance(quantum_device_cfg[category], dict) and setup_id in quantum_device_cfg[
                            category].keys():
                            lattice_cfg[category][key][qb_id] = quantum_device_cfg[category][setup_id][key]
                        else:
                            lattice_cfg[category][key][qb_id] = quantum_device_cfg[category][key]
                    except:
                        print("[{}][{}] not a category and key quantum device config".format(category, key))

    with open(lattice_cfg_name, 'w') as f:
        json.dump(lattice_cfg, f, indent=2)


def generate_quantum_device_from_lattice(lattice_cfg_name, qb_ids, setups=["A", "B"]):
    with open(lattice_cfg_name, 'r') as f:
        lattice_cfg = json.load(f)
        quantum_device_cfg = {}

        if len(qb_ids) == 1:
            qb_ids = qb_ids * 2

        for category in lattice_cfg.keys():
            # if category is directly a list of 8 qubit values, stuff it into setups "A" and "B"
            if isinstance(lattice_cfg[category], list) and len(lattice_cfg[category]) == 8:
                for i in range(len(qb_ids)):
                    quantum_device_cfg[category] = {}
                    quantum_device_cfg[category][setups[i]] = lattice_cfg[category][qb_ids[i]]

            # if category is a dictionary, walk through the keys.
            elif isinstance(lattice_cfg[category], dict):
                quantum_device_cfg[category] = {}
                for i in range(len(qb_ids)):
                    quantum_device_cfg[category][setups[i]] = {}
                for key in lattice_cfg[category]:
                    # if one of them is a list of eight qubit values,stuff it in quantum device config
                    if isinstance(lattice_cfg[category][key], list) and len(lattice_cfg[category][key]) == 8:
                        for i in range(len(qb_ids)):
                            quantum_device_cfg[category][setups[i]][key] = lattice_cfg[category][key][qb_ids[i]]
                    # else, just stuff it directly
                    else:
                        quantum_device_cfg[category][key] = lattice_cfg[category][key]

            # if category is other, just stuff it directly into quantum device config
            else:
                quantum_device_cfg[category] = lattice_cfg[category]

        return quantum_device_cfg

def generate_quantum_device_from_lattice_v2(lattice_cfg_name, qb_ids, setups=["A", "B"]):
    with open(lattice_cfg_name, 'r') as f:
        lattice_cfg = json.load(f)
        quantum_device_cfg = {}

        if len(qb_ids) == 1:
            qb_ids = qb_ids * 2

        quantum_device_cfg["setups"] = setups

        for category in lattice_cfg.keys():
            # if category is directly a list of 8 qubit values, stuff it into setups "A" and "B"
            if isinstance(lattice_cfg[category], list) and len(lattice_cfg[category]) == 8:
                for i in range(len(qb_ids)):
                    quantum_device_cfg[category] = {}
                    quantum_device_cfg[category][setups[i]] = lattice_cfg[category][qb_ids[i]]

            # if category is a dictionary, walk through the keys.
            elif isinstance(lattice_cfg[category], dict):
                quantum_device_cfg[category] = {}
                for i in range(len(qb_ids)):
                    quantum_device_cfg[category][setups[i]] = {}
                for key in lattice_cfg[category]:
                    # if one of them is a list of eight qubit values,stuff it in quantum device config
                    if isinstance(lattice_cfg[category][key], list) and len(lattice_cfg[category][key]) == 8:
                        for i in range(len(qb_ids)):
                            quantum_device_cfg[category][setups[i]][key] = lattice_cfg[category][key][qb_ids[i]]
                    #elif check if setup specific
                    elif key == setups[0]:
                        for key_set in lattice_cfg[category][key]:
                            quantum_device_cfg[category][setups[0]][key_set] = lattice_cfg[category][key][key_set][qb_ids[0]]
                    elif key == setups[1]:
                        for key_set in lattice_cfg[category][key]:
                            quantum_device_cfg[category][setups[1]][key_set] = lattice_cfg[category][key][key_set][qb_ids[1]]
                    # else, just stuff it directly
                    else:
                        quantum_device_cfg[category][key] = lattice_cfg[category][key]

            # if category is other, just stuff it directly into quantum device config
            else:
                quantum_device_cfg[category] = lattice_cfg[category]

        return quantum_device_cfg

def generate_quantum_device_from_lattice_v3(lattice_cfg_name, on_qubits = {'A':1,'B':2}):
    with open(lattice_cfg_name, 'r') as f:
        lattice_cfg = json.load(f)
        quantum_device_cfg = {}

        quantum_device_cfg["on_qubits"] = on_qubits
        setups = list[on_qubits.keys()]
        qb_ids  = list[on_qubits.values()]

        for category in lattice_cfg.keys():
            # if category is directly a list of 8 qubit values, stuff it into setups "A" and "B"
            if isinstance(lattice_cfg[category], list) and len(lattice_cfg[category]) == 8:
                for i in range(len(qb_ids)):
                    quantum_device_cfg[category] = {}
                    quantum_device_cfg[category][setups[i]] = lattice_cfg[category][qb_ids[i]]

            # if category is a dictionary, walk through the keys.
            elif isinstance(lattice_cfg[category], dict):
                quantum_device_cfg[category] = {}
                for i in range(len(qb_ids)):
                    quantum_device_cfg[category][setups[i]] = {}
                for key in lattice_cfg[category]:
                    # if one of them is a list of eight qubit values,stuff it in quantum device config
                    if isinstance(lattice_cfg[category][key], list) and len(lattice_cfg[category][key]) == 8:
                        for i in range(len(qb_ids)):
                            quantum_device_cfg[category][setups[i]][key] = lattice_cfg[category][key][qb_ids[i]]
                    #elif check if setup specific
                    elif key == setups[0]:
                        for key_set in lattice_cfg[category][key]:
                            quantum_device_cfg[category][setups[0]][key_set] = lattice_cfg[category][key][key_set][qb_ids[0]]
                    elif key == setups[1]:
                        for key_set in lattice_cfg[category][key]:
                            quantum_device_cfg[category][setups[1]][key_set] = lattice_cfg[category][key][key_set][qb_ids[1]]
                    # else, just stuff it directly
                    else:
                        quantum_device_cfg[category][key] = lattice_cfg[category][key]

            # if category is other, just stuff it directly into quantum device config
            else:
                quantum_device_cfg[category] = lattice_cfg[category]

        return quantum_device_cfg