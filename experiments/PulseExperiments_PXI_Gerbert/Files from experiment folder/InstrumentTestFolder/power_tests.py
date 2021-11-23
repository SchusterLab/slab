import platform
print(platform.python_version())
from slab.experiments.PulseExperiments_PXI.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_PXI.pulse_experiment import Experiment
from slab.experiments.PulseExperiments_PXI.pulse_experiment import generate_quantum_device_from_lattice_v2
from slab.instruments.keysight import keysight_pxi_load as ks_pxi
from slab.instruments.keysight import KeysightLib as key
from slab.instruments.keysight import keysightSD1 as SD1
from Keysight_card_control import *
from slab import InstrumentManager
import time
import json
import numpy as np

import os
path = os.getcwd()
path = "C:\\210412 - PHMIV3_56 - BF4 cooldown 2"

show = 'I'

lattice_cfg_name = '210526_sawtooth_lattice_device_config.json'
#lattice_cfg_name = '210510_sawtooth_lattice_device_config_wff.json'

with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)
with open(lattice_cfg_name, 'r') as f:
    lattice_cfg = json.load(f)

quantum_device_cfg = generate_quantum_device_from_lattice_v2(lattice_cfg_name, qb_ids=[0], setups=['A','B'])

class PowerTests:
    def __init__(self, quantum_device_cfg, experiment_cfg, hardware_cfg):
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg
        im = InstrumentManager()
        self.qubits = ["A", "B"]
        import time
        # self.fluxbias = im['dacbox']
        # self.fluxbias.setvoltage(1,0)
        time.sleep(1)

        try:
            self.drive_los = [im[lo] for lo in self.hardware_cfg['drive_los']]
        except:
            print("No drive function generator specified in hardware config / failure to connect with im()")

        try:
            self.stab_los = [im[lo] for lo in self.hardware_cfg['stab_los']]
        except:
            print("No stabilizer function generator specified in hardware config / failure to connect with im()")

        try:
            self.readout_los = [im[lo] for lo in self.hardware_cfg['readout_los']]
        except:
            print("No readout function generator specified in hardware config / failure to connect with im()")

        try:
            self.readout_attens = [im[atten] for atten in self.hardware_cfg['readout_attens']]
        except:
            print("No digital attenuator specified in hardware config / failure to connect with im()")

        try:
            self.drive_attens = [im[atten] for atten in self.hardware_cfg['drive_attens']]
        except:
            print("No digital attenuator specified in hardware config / failure to connect with im()")

        try:
            self.trig = im['triggrb']
        except:
            print("No trigger function generator specied in hardware cfg / failure to connect with im()")
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

    def run_power_test(self):
        self.initiate_drive_LOs()
        self.initiate_readout_LOs()
        self.initiate_readout_attenuators()
        self.initiate_drive_attenuators()


class KeyPowerTests:
    def __init__(self, experiment_cfg, hardware_cfg, quantum_device_cfg, on_qbs, num_avg, num_expt):  # 1000*10000 if you want to watch sweep by eye

        chassis = key.KeysightChassis(1,
                                      {4: key.ModuleType.OUTPUT,
                                       5: key.ModuleType.OUTPUT,
                                       6: key.ModuleType.OUTPUT,
                                       7: key.ModuleType.OUTPUT,
                                       8: key.ModuleType.OUTPUT,
                                       9: key.ModuleType.OUTPUT,
                                       10: key.ModuleType.INPUT})

        self.hardware_cfg = hardware_cfg
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.AWG_mod_no = hardware_cfg['awg_info']['keysight_pxi']['AWG_mod_no']  ## AWG_mod_no is qubit AWG output
        self.marker_mod_no = hardware_cfg['awg_info']['keysight_pxi'][
            'marker_mod_no']  ## marker is a pulse that's on for length that you want LO switch on
        self.stab_mod_no = hardware_cfg['awg_info']['keysight_pxi']['stab_mod_no']  ## IQ and marker for
        # stabilizer, + triggers Digitizer
        self.ff1_mod_no = hardware_cfg['awg_info']['keysight_pxi']['ff1_mod_no']  # 4 channels for Q0-Q3 fast flux
        self.ff2_mod_no = hardware_cfg['awg_info']['keysight_pxi']['ff2_mod_no']  # 4 channels for Q4-Q7 fast flux
        self.dig_mod_no = hardware_cfg['awg_info']['keysight_pxi']['dig_mod_no']  # digitizer card

        self.dt = hardware_cfg['awg_info']['keysight_pxi']['dt']
        self.dt_m = hardware_cfg['awg_info']['keysight_pxi']['dt_m']
        self.dt_dig = hardware_cfg['awg_info']['keysight_pxi']['dt_dig']
        self.dt_M3201A = hardware_cfg['awg_info']['keysight_pxi']['dt_M3201A']
        self.adc_range = hardware_cfg['awg_info']['keysight_pxi']['digtzr_vpp_range']

        self.qb_lo_conv = hardware_cfg['awg_info']['keysight_pxi']['qb_lo_conv']  # lo_delay convolves LO marker
        self.stab_lo_conv = hardware_cfg['awg_info']['keysight_pxi']['stab_lo_conv']  # lo_delay convolves LO marker
        self.hardware_delays = hardware_cfg['awg_info']['keysight_pxi']['channels_delay_hardware_10ns']
        self.trig_pulse_length = hardware_cfg['trig_pulse_len']['default']

        print("Module used for generating Q1 IQ  pulses = ", self.AWG_mod_no)
        print("Module used for generating digital markers for LO = ", self.marker_mod_no)
        print("Module used to trigger dig and for stabilizer  = ", self.stab_mod_no)
        print("Module used for generating fast flux pluses for Q0-Q3 = ", self.ff1_mod_no)
        print("Module used for generating fast flux pluses for Q4-Q7 = ", self.ff2_mod_no)
        self.out_mod_nums = [self.AWG_mod_no, self.marker_mod_no, self.stab_mod_no, self.ff1_mod_no, self.ff2_mod_no]

        self.on_qubits = on_qbs
        self.num_avg = num_avg
        self.num_expt = num_expt
        self.trigger_period = self.hardware_cfg['trigger']['period_us']
        self.DIG_sampl_record = hardware_cfg['awg_info']['keysight_pxi']['samplesPerRecord']
        self.totaltime = self.num_avg * self.num_expt * self.trigger_period * 1e-6 / 60.0

        self.chassis = chassis
        self.awg_channels = range(1, 5)
        self.dig_channels = range(1, 5)

        # Initialize AWG Cards!

        # initialize modules
        self.AWG_module = chassis.getModule(self.AWG_mod_no)
        self.m_module = chassis.getModule(self.marker_mod_no)
        self.stab_module = chassis.getModule(self.stab_mod_no)
        self.ff1_module = chassis.getModule(self.ff1_mod_no)
        self.ff2_module = chassis.getModule(self.ff2_mod_no)
        self.DIG_module = chassis.getModule(self.dig_mod_no)

        self.out_mods = [self.AWG_module, self.m_module, self.stab_module, self.ff1_module, self.ff2_module]

    def configureChannels(self):
        '''Configures the individual channels that are used in the experiment. This section may be modified as needed
        for other experiments. See documentation in KeysightLib for the configure() methods on KeysightChannelIn and
        KeysightChannelOut. Amplitude is in Vpp.'''

        hardware_cfg = self.hardware_cfg
        quantum_device_cfg = self.quantum_device_cfg
        amp_AWG = hardware_cfg['awg_info']['keysight_pxi']['amp_awg']
        amp_mark = hardware_cfg['awg_info']['keysight_pxi']['amp_mark']
        amp_stab = hardware_cfg['awg_info']['keysight_pxi']['amp_stab']
        amp_digtzr_trig = hardware_cfg['awg_info']['keysight_pxi']['amp_stab'][3] ##THIS IS ON PURPOSE
        amp_ff1 = hardware_cfg['awg_info']['keysight_pxi']['amp_ff1']
        amp_ff2 = hardware_cfg['awg_info']['keysight_pxi']['amp_ff2']
        IQ_dc_offsetA = quantum_device_cfg['pulse_info']['A']['IQ_dc']
        IQ_dc_offsetB = quantum_device_cfg['pulse_info']['B']['IQ_dc']
        DIG_ch_delays = hardware_cfg["awg_info"]['keysight_pxi']["DIG_channels_delay_samples"]
        qbA_freq = self.quantum_device_cfg["qubit"]["A"]["freq"]
        qbB_freq = self.quantum_device_cfg["qubit"]["B"]["freq"]
        rdA_freq = self.quantum_device_cfg["readout"]["A"]["freq"]
        rdB_freq = self.quantum_device_cfg["readout"]["B"]["freq"]
        A_iq_freq = self.quantum_device_cfg["pulse_info"]["A"]["iq_freq"]
        A_Q_phase = self.quantum_device_cfg["pulse_info"]["A"]["Q_phase"]
        B_iq_freq = self.quantum_device_cfg["pulse_info"]["B"]["iq_freq"]
        B_Q_phase = self.quantum_device_cfg["pulse_info"]["B"]["Q_phase"]


        num_avg = self.num_avg
        num_expt = self.num_expt
        print('num_exp = %s' %num_expt)

        print ("Configuring qubit IQ channels")
        self.AWG_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        set_card_ch_CW(self.AWG_mod_no, 1, amp_AWG[0], A_iq_freq)
        self.AWG_ch_1.configure(amplitude=amp_AWG[0], offset_voltage = IQ_dc_offsetA)
        self.AWG_ch_2.configure(amplitude=amp_AWG[1], offset_voltage = IQ_dc_offsetA)
        self.AWG_ch_3.configure(amplitude=amp_AWG[2], offset_voltage = IQ_dc_offsetB)
        self.AWG_ch_4.configure(amplitude=amp_AWG[3], offset_voltage = IQ_dc_offsetB)

        print("Configuring marker channels")
        self.m_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.m_ch_1.configure(amplitude=amp_mark[0])
        self.m_ch_2.configure(amplitude=amp_mark[1])
        self.m_ch_3.configure(amplitude=amp_mark[2])
        self.m_ch_4.configure(amplitude=amp_mark[3])

        print("Configuring trigger channels")
        self.stab_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.stab_ch_1.configure(amplitude=amp_stab[0])
        self.stab_ch_2.configure(amplitude=amp_stab[1])
        self.stab_ch_3.configure(amplitude=amp_stab[2])
        self.digtzr_trig_ch.configure(amplitude=amp_digtzr_trig)
        print ("Dig card trigger amplitude = ",amp_digtzr_trig)


        print ("Setting trigger mode for all channels of all output modules to External")
        for n in range(1, 5):
            self.AWG_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
            self.m_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
            self.stab_module.AWGtriggerExternalConfig(nAWG=n,externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
            self.ff1_module.AWGtriggerExternalConfig(nAWG=n,
                                                      externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,
                                                      triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)
            self.ff2_module.AWGtriggerExternalConfig(nAWG=n,
                                                      externalSource=SD1.SD_TriggerExternalSources.TRIGGER_EXTERN,
                                                      triggerBehavior=SD1.SD_TriggerBehaviors.TRIGGER_RISE)


        print ("Configuring digitizer. ADC range set to",self.adc_range, "Vpp")
        self.DIG_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN)
        self.DIG_ch_1.configure(full_scale=self.adc_range, delay=DIG_ch_delays[0], points_per_cycle=self.DIG_sampl_record,cycles=num_expt * num_avg, buffer_time_out=100000, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_2.configure(full_scale = self.adc_range,delay=DIG_ch_delays[1], points_per_cycle=self.DIG_sampl_record, buffer_time_out=100000, cycles=num_expt * num_avg, trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_3.configure(full_scale=self.adc_range, delay=DIG_ch_delays[2], points_per_cycle=self.DIG_sampl_record,cycles=num_expt * num_avg, buffer_time_out=100000, trigger_mode=SD1.SD_TriggerModes.EXTTRIG,use_buffering=True, cycles_per_return=num_expt)
        self.DIG_ch_4.configure(full_scale=self.adc_range, delay=DIG_ch_delays[3], points_per_cycle=self.DIG_sampl_record,
                                cycles=num_expt * num_avg, buffer_time_out=100000,
                                trigger_mode=SD1.SD_TriggerModes.EXTTRIG, use_buffering=True,
                                cycles_per_return=num_expt)


