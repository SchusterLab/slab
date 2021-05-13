
from slab.experiments.PulseExperiments_PXI.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_PXI.pulse_experiment import Experiment
# modified 4/14/21, set_output_state to set_output but THIS DOES NOT WORK
from slab import InstrumentManager
import json

with open('quantum_device_config.json', 'r') as f:
    quantum_device_cfg  = json.load(f)
with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)

im = InstrumentManager()

#hardware_cfg['LO_readB'] = ['SC_26C5']
SC = [im[lo] for lo in hardware_cfg['LO_readA']]
# SC = [im[lo] for lo in hardware_cfg['LO_qubitA']]

SC[0].set_frequency(6.2e9)
SC[0].set_power(0)
# readout_los[0].set_ext_pulse(mod=False)
SC[0].set_rf_mode(val=0) # single RF tone on output 1

SC[0].set_output_state(True)
SC[0].set_output_state(False)
# SC[0].set_output(True)
# # SC[0].set_output(False)

readParams = SC[0].get_rf_parameters()
print('Readout frequency '+ str(round(readParams.rf1_freq/1e9,5))+' GHz')
print('Readout power '+ str(round(readParams.rf_level,3))+' dbm')

# If want to merge commands into initiate_whatever_los in pulse_experiment.py, need to address
# the fact that the SignalCore.py command is set_output_state whereas the command used to
# initiate the LOs in general is set_output.

# turn off all LOs
SC = [im[lo] for lo in hardware_cfg['LO_qubitA']]
SC[0].set_power(-20)
SC[0].set_output_state(False)

SC = [im[lo] for lo in hardware_cfg['LO_readA']]
SC[0].set_power(-20)
SC[0].set_output_state(False)

SC = [im[lo] for lo in hardware_cfg['LO_cavityA']]
SC[0].set_power(-20)
SC[0].set_output_state(False)

SC = [im[lo] for lo in hardware_cfg['LO_readB']]
SC[0].set_power(-20)
SC[0].set_output_state(False)
