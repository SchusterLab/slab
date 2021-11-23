import json
import pickle
import os
from numpy import *

from slab import *
from slab.experiments.PulseExperiments_PXI.sequences_pxi import PulseSequences
from slab.experiments.PulseExperiments_PXI.pulse_experiment import Experiment
from slab.datamanagement import SlabFile
from DACInterface import AD5780_serial

# Create a data file to store contents of automate exp, such as experiment numbers, fits, etc
### SWEEP PARAMETER
qb_id = 5
automate_config_string = "Q%s"%qb_id + "_automate_config_no_amp.json"
path = os.getcwd()
name = "Q%s"%qb_id + 'automate'
data_path = os.path.join(path, 'data/')
auto_data_file = os.path.join(data_path, get_next_filename(data_path, name, suffix='.h5'))
slab_file = SlabFile(auto_data_file)

###Set Feedback Parameters
MAX_RABI_PI_LEN = 400  #in sigma
SIGMA_CUTOFF = 4 #for converting between pi len in sigma vs time

#Initialize DACBOX
DCBOX = 0
YOKO = 0
print("Initiating DAC")
if DCBOX == 1:
    dac = AD5780_serial()
    time.sleep(2)
    dac.init()
    time.sleep(2)
    dac.init()
    time.sleep(2)
print("DAC initiated")

#Open config files with all the parameters
with open(automate_config_string, 'r') as f:
    automate_cfg = json.load(f)
#TODO: Save to our auto_data file for future reference
#with slab_file as file:
#    file.append_line('automate_cfg', [automate_cfg])
#Set Sweep Parameters
# V = automate_cfg['voltage']


#Starting experiment!
# print("ramping DAC to "+ str(V[0]))
# dac.ramp3(qb_id+1,V[0],2,2)
# print("DAC ramped")

with open('quantum_device_config.json', 'r') as f:
    quantum_device_cfg = json.load(f)
with open('experiment_config.json', 'r') as f:
    experiment_cfg = json.load(f)
with open('hardware_config.json', 'r') as f:
    hardware_cfg = json.load(f)

# TODO: REMOVE REFERENCES TO DAC
for experiment_name in ['pulse_probe_iq']:

    if experiment_name == "pulse_probe_iq":
        qb_freq_ppiq = []
        for i in range(len(V)):
            # dac.ramp3(qb_id+1, V[i], 2, 2)

            # Write the values saved in automate_config to the config files
            with open(automate_config_string, 'r') as f:
                automate_cfg = json.load(f)
            PNAX_freq = automate_cfg['qb_freq_PNAX']
            PNAX_readout_freq = automate_cfg['readout_frequency'][i]
            quantum_device_cfg['readout']['freq'] = PNAX_readout_freq
            qb_atten = automate_cfg['ppiq_qubit_drive_digital_attenuation']
            qb_pulse = [250] * len(V)
            quantum_device_cfg['qubit']['1']['freq'] = PNAX_freq[i]
            quantum_device_cfg['qubit_drive_digital_attenuation'] = qb_atten[i]
            experiment_cfg['pulse_probe_iq']["pulse_length"] = qb_pulse[i]
            with open('quantum_device_config.json', 'w') as f:
                json.dump(quantum_device_cfg, f, indent=2)
            with open('experiment_config.json', 'w') as f:
                json.dump(experiment_cfg, f, indent=2)
            with open('hardware_config.json', 'w') as f:
                json.dump(hardware_cfg, f, indent=2)

            #Now that the right values are set, run the experiment
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, plot_visdom=False)
            sequences = ps.get_experiment_sequences(experiment_name)
            print("Sequences generated")
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

            #fit the data, and save it to auto_data_file
            fit_params = exp.post_analysisandsave(path, experiment_name, name, P='I', phi=0, cont_data_file=auto_data_file)
            qb_freq_ppiq.append(fit_params[2])

        #Update the automate_config file
        automate_cfg['qb_freq_ppiq'] = qb_freq_ppiq
        with open(automate_config_string, 'w') as f:
            json.dump(automate_cfg, f, indent=1)

    if experiment_name == "rabi":
        pi_len = []
        for i in range(len(V)):
            dac.ramp3(qb_id+1, V[i], 2, 2)

            # Write the values saved in automate_config to the config files
            with open(automate_config_string, 'r') as f:
                automate_cfg = json.load(f)
            ppiq_freq = automate_cfg['qb_freq_ppiq']
            qb_atten = automate_cfg['qubit_drive_digital_attenuation']
            qb_pulse = [250] * len(V)
            quantum_device_cfg['qubit']['1']['freq'] = ppiq_freq[i]
            quantum_device_cfg['qubit_drive_digital_attenuation'] = (qb_atten[i])
            print(i)
            print(qb_atten)
            experiment_cfg['pulse_probe_iq']["pulse_length"] = qb_pulse[i]
            with open('quantum_device_config.json', 'w') as f:
                json.dump(quantum_device_cfg, f, indent=2)
            with open('experiment_config.json', 'w') as f:
                json.dump(experiment_cfg, f, indent=2)
            with open('hardware_config.json', 'w') as f:
                json.dump(hardware_cfg, f, indent=2)

            #Now that the right values are set, run the experiment
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, plot_visdom=False)
            sequences = ps.get_experiment_sequences(experiment_name)
            print("Sequences generated")
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

            #fit the data, and save it to auto_data_file
            fit_params = exp.post_analysis(path, experiment_name, name, P='Q', phi=0)
            if quantum_device_cfg['pulse_info']['1']['pulse_type'] == "gauss":
                pi_time = SIGMA_CUTOFF * 1 / (2 * fit_params[1])
            else:
                pi_time = 1 / (2 * fit_params[1])

            ## If values aren't what you want, run again
            if False:
            #if pi_time>MAX_RABI_PI_LEN:
                print("PI TIME TOO LONG! Pi time is " + str(pi_time) + " when max pi time is " + str(MAX_RABI_PI_LEN) + "Adjusting drive power")
                power_diff = 20 * np.log10(MAX_RABI_PI_LEN/pi_time)
                qb_atten = automate_cfg['qubit_drive_digital_attenuation'][i]
                new_atten = qb_atten + power_diff
                if new_atten<0:
                    new_atten=0
                automate_cfg['qubit_drive_digital_attenuation'][i] = new_atten
                with open('quantum_device_config.json', 'w') as f:
                    json.dump(quantum_device_cfg, f, indent=2)

                exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
                exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

            fit_params = exp.post_analysisandsave(path, experiment_name, name, P='Q', phi=0, cont_data_file=auto_data_file)
            pi_len.append(1/(2*fit_params[1]))

        #Update the automate_config file
        automate_cfg['pi_len'] = pi_len
        with open(automate_config_string, 'w') as f:
            json.dump(automate_cfg, f, indent=1)

    if experiment_name == "t1":
        t1_list = []
        for i in range(len(V)):
            dac.ramp3(qb_id+1, V[i], 2, 2)

            # Write the values saved in automate_config to the config files
            with open(automate_config_string, 'r') as f:
                automate_cfg = json.load(f)
            ppiq_freq = automate_cfg['qb_freq_ppiq']
            pi_len = automate_cfg['pi_len']
            qb_atten = automate_cfg['qubit_drive_digital_attenuation']
            qb_pulse = [250] * len(V)
            quantum_device_cfg['qubit']['1']['freq'] = ppiq_freq[i]
            quantum_device_cfg['qubit_drive_digital_attenuation'] = qb_atten[i]
            experiment_cfg['pulse_probe_iq']["pulse_length"] = qb_pulse[i]
            quantum_device_cfg['pulse_info']['1']['pi_len'] = np.ceil(pi_len[i])
            quantum_device_cfg['pulse_info']['1']['pi_amp'] = pi_len[i] / np.ceil(pi_len[i])
            quantum_device_cfg['pulse_info']['1']['half_pi_len'] = np.ceil(pi_len[i]/2)
            quantum_device_cfg['pulse_info']['1']['half_pi_amp'] = (pi_len[i]/2) / np.ceil(pi_len[i]/2)
            with open('quantum_device_config.json', 'w') as f:
                json.dump(quantum_device_cfg, f, indent=2)
            with open('experiment_config.json', 'w') as f:
                json.dump(experiment_cfg, f, indent=2)
            with open('hardware_config.json', 'w') as f:
                json.dump(hardware_cfg, f, indent=2)

            #Now that the right values are set, run the experiment
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, plot_visdom=False)
            sequences = ps.get_experiment_sequences(experiment_name)
            print("Sequences generated")
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

            #fit the data, and save it to auto_data_file
            fit_params = exp.post_analysisandsave(path, experiment_name, name, P='I', phi=0, cont_data_file=auto_data_file)
            t1_list.append(fit_params[3])

        #Update the automate_config file
        automate_cfg['t1'] = t1_list
        with open(automate_config_string, 'w') as f:
            json.dump(automate_cfg, f, indent=1)

    if experiment_name == "ramsey":
        t2star_list = []
        qb_freq_ramsey_list = []
        for i in range(len(V)):
            dac.ramp3(qb_id+1, V[i], 2, 2)

            # Write the values saved in automate_config to the config files
            with open(automate_config_string, 'r') as f:
                automate_cfg = json.load(f)
            ppiq_freq = automate_cfg['qb_freq_ppiq']
            pi_len = automate_cfg['pi_len']
            qb_atten = automate_cfg['qubit_drive_digital_attenuation']
            qb_pulse = [250] * len(V)
            quantum_device_cfg['qubit']['1']['freq'] = ppiq_freq[i]
            quantum_device_cfg['qubit_drive_digital_attenuation'] = qb_atten[i]
            experiment_cfg['pulse_probe_iq']["pulse_length"] = qb_pulse[i]
            quantum_device_cfg['pulse_info']['1']['pi_len'] = np.ceil(pi_len[i])
            quantum_device_cfg['pulse_info']['1']['pi_amp'] = pi_len[i] / np.ceil(pi_len[i])
            quantum_device_cfg['pulse_info']['1']['half_pi_len'] = np.ceil(pi_len[i]/2)
            quantum_device_cfg['pulse_info']['1']['half_pi_amp'] = (pi_len[i]/2) / np.ceil(pi_len[i]/2)

            with open('quantum_device_config.json', 'w') as f:
                json.dump(quantum_device_cfg, f, indent=2)
            with open('experiment_config.json', 'w') as f:
                json.dump(experiment_cfg, f, indent=2)
            with open('hardware_config.json', 'w') as f:
                json.dump(hardware_cfg, f, indent=2)

            #Now that the right values are set, run the experiment
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, plot_visdom=False)
            sequences = ps.get_experiment_sequences(experiment_name)
            print("Sequences generated")
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

            #fit the data, and save it to auto_data_file
            fit_params = exp.post_analysisandsave(path, experiment_name, name, P='I', phi=0, cont_data_file=auto_data_file)
            t2star_list.append(fit_params[3])
            ramsey_freq = experiment_cfg['ramsey']['ramsey_freq']*1e3
            offset = ramsey_freq - fit_params[1]
            qb_freq_ramsey_list.append(ppiq_freq[i] + offset*10**-3)
        #Update the automate_config file
        automate_cfg['t2star'] = t2star_list
        automate_cfg['qb_freq_ramsey'] = qb_freq_ramsey_list
        with open(automate_config_string, 'w') as f:
            json.dump(automate_cfg, f, indent=1)

    ### Think about this more - Echo experiments can vary in pulses - sweeping pulses, qubits, and flux creates
    ### a rank 3 tensor!!! How to store that in .json?  list(list(list)))?  For now assume only 1 pulse
    if experiment_name == "echo":
        t2_list = []
        for i in range(len(V)):
            dac.ramp3(qb_id+1, V[i], 2, 2)

            # Write the values saved in automate_config to the config files
            with open(automate_config_string, 'r') as f:
                automate_cfg = json.load(f)
            ppiq_freq = automate_cfg['qb_freq_ppiq']
            pi_len = automate_cfg['pi_len']
            qb_atten = automate_cfg['qubit_drive_digital_attenuation']
            qb_pulse = [250] * len(V)
            quantum_device_cfg['qubit']['1']['freq'] = ppiq_freq[i]
            quantum_device_cfg['qubit_drive_digital_attenuation'] = qb_atten[i]
            experiment_cfg['pulse_probe_iq']["pulse_length"] = qb_pulse[i]
            quantum_device_cfg['pulse_info']['1']['pi_len'] = np.ceil(pi_len[i])
            quantum_device_cfg['pulse_info']['1']['pi_amp'] = pi_len[i] / np.ceil(pi_len[i])
            quantum_device_cfg['pulse_info']['1']['half_pi_len'] = np.ceil(pi_len[i]/2)
            quantum_device_cfg['pulse_info']['1']['half_pi_amp'] = (pi_len[i]/2) / np.ceil(pi_len[i]/2)

            with open('quantum_device_config.json', 'w') as f:
                json.dump(quantum_device_cfg, f, indent=2)
            with open('experiment_config.json', 'w') as f:
                json.dump(experiment_cfg, f, indent=2)
            with open('hardware_config.json', 'w') as f:
                json.dump(hardware_cfg, f, indent=2)

            #Now that the right values are set, run the experiment
            ps = PulseSequences(quantum_device_cfg, experiment_cfg, hardware_cfg, plot_visdom=False)
            sequences = ps.get_experiment_sequences(experiment_name)
            print("Sequences generated")
            exp = Experiment(quantum_device_cfg, experiment_cfg, hardware_cfg, sequences, experiment_name)
            exp.run_experiment_pxi(sequences, path, experiment_name, expt_num=0, check_sync=False)

            #fit the data, and save it to auto_data_file
            fit_params = exp.post_analysisandsave(path, experiment_name, name, P='I', phi=0, cont_data_file=auto_data_file)
            t2_list.append(fit_params[3])
        #Update the automate_config file
        automate_cfg['t2'] = t2_list
        with open(automate_config_string, 'w') as f:
            json.dump(automate_cfg, f, indent=1)

print ("Returning DCBOX to 0.0 mA")
for ii in range(8):
    dac.ramp3(ii+1,0.001,step=2,speed=2)
    time.sleep(0.5)
time.sleep(2)
dac.init()
time.sleep(2)
