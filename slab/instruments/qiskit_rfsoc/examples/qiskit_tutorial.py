"""
tutorial.py
"""
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from qiskit import assemble
from qiskit import IBMQ
from qiskit import pulse
from qiskit.pulse import Play
from qiskit.pulse import library as pulse_lib
from qiskit.tools.monitor import job_monitor

# IBMQ.save_account()

DPI = 300

# unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds

# samples need to be multiples of 16
def get_closest_multiple_of_16(num):
    return int(num + 8 ) - (int(num + 8 ) % 16)
#ENDDEF

def measure_instruction(backend, qubit):
    backend_config = backend.configuration()
    backend_defaults = backend.defaults()
    # Find out which group of qubits need to be acquired with this qubit
    meas_map_idx = None
    for i, measure_group in enumerate(backend_config.meas_map):
        if qubit in measure_group:
            meas_map_idx = i
            break
        #ENDIF
    #ENDFOR

    inst_sched_map = backend_defaults.instruction_schedule_map
    measure = inst_sched_map.get('measure', qubits=backend_config.meas_map[meas_map_idx])

    return measure
#ENDDEF

def freq_sweep(run=False):
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    backend = provider.get_backend('ibmq_armonk')
    backend_config = backend.configuration()
    backend_defaults = backend.defaults()
    dt = backend_config.dt

    # We will find the qubit frequency for the following qubit.
    qubit = 0

    # The sweep will be centered around the estimated qubit frequency.
    # The default frequency is given in Hz
    # warning: this will change in a future release
    center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]
    # print(f"Qubit {qubit} has an estimated frequency of {center_frequency_Hz / GHz} GHz.")

    # scale factor to remove factors of 10 from the data
    scale_factor = 1e-14

    # We will sweep 40 MHz around the estimated frequency
    frequency_span_Hz = 40 * MHz
    # in steps of 1 MHz.
    frequency_step_Hz = 1 * MHz

    # We will sweep 20 MHz above and 20 MHz below the estimated frequency
    frequency_min = center_frequency_Hz - frequency_span_Hz / 2
    frequency_max = center_frequency_Hz + frequency_span_Hz / 2
    # Construct an np array of the frequencies for our experiment
    frequencies_GHz = np.arange(frequency_min / GHz, 
                                frequency_max / GHz, 
                                frequency_step_Hz / GHz)

    # print(f"The sweep will go from {frequency_min / GHz} GHz to {frequency_max / GHz} GHz \
    # in steps of {frequency_step_Hz / MHz} MHz.")

    # Drive pulse parameters (us = microseconds)
    # This determines the actual width of the gaussian
    drive_sigma_us = 0.075
    # This is a truncating parameter, because gaussians don't have a natural finite length
    drive_samples_us = drive_sigma_us*8
    # The width of the gaussian in units of dt
    drive_sigma = get_closest_multiple_of_16(drive_sigma_us * us /dt)
    # The truncating parameter in units of dt
    drive_samples = get_closest_multiple_of_16(drive_samples_us * us /dt)
    drive_amp = 0.05
    # Drive pulse samples
    drive_pulse = pulse_lib.gaussian(duration=drive_samples,
                                     sigma=drive_sigma,
                                     amp=drive_amp,
                                     name='freq_sweep_excitation_pulse')

    measure = measure_instruction(backend, qubit)

    ### Collect the necessary channels
    drive_chan = pulse.DriveChannel(qubit)
    meas_chan = pulse.MeasureChannel(qubit)
    acq_chan = pulse.AcquireChannel(qubit)

    # Create the base schedule
    # Start with drive pulse acting on the drive channel
    schedule = pulse.Schedule(name='Frequency sweep')
    schedule += Play(drive_pulse, drive_chan)
    # The left shift `<<` is special syntax meaning to
    # shift the start time of the schedule by some duration
    schedule += measure << schedule.duration

    # Create the frequency settings for the sweep (MUST BE IN HZ)
    frequencies_Hz = frequencies_GHz*GHz
    # schedule_frequencies = [{drive_chan: freq} for freq in frequencies_Hz]
    schedule_frequencies = [{drive_chan: center_frequency_Hz} for i in range(5)]
    

    # num_shots_per_frequency = 1024
    num_shots_per_frequency = 1
    frequency_sweep_program = assemble(schedule,
                                       backend=backend, 
                                       meas_level=1,
                                       meas_return='avg',
                                       shots=num_shots_per_frequency,
                                       schedule_los=schedule_frequencies)

    print(frequency_sweep_program)
    
    if run:
        data_file_path = "freq_sweep.h5"
        if not os.path.isfile(data_file_path):
            job = backend.run(frequency_sweep_program)
            job_monitor(job)
            frequency_sweep_results = job.result(timeout=120)

            sweep_values = np.zeros(len(frequency_sweep_results.results), dtype=np.complex64)
            for i in range(len(frequency_sweep_results.results)):
                # Get the results from the ith experiment
                res = frequency_sweep_results.get_memory(i)*scale_factor
                # Get the results for `qubit` from this experiment
                sweep_values[i] = res[qubit]
            #ENDFOR
            with h5py.File(data_file_path, "w") as data_file:
                data_file["sweep_values"] = sweep_values
        else:
            with h5py.File(data_file_path, "r") as data_file:
                sweep_values = data_file["sweep_values"][()]
            #ENDWITH
        #ENDIF
        # plot real part of sweep values
        # plt.scatter(frequencies_GHz, np.real(sweep_values), color='black')
        # plt.xlim([min(frequencies_GHz), max(frequencies_GHz)])
        # plt.xlabel("Frequency [GHz]")
        # plt.ylabel("Measured signal [a.u.]")
        # plt.show()
    #ENDIF
#ENDDEF


def rabi():
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    backend = provider.get_backend('ibmq_armonk')
    backend_config = backend.configuration()
    backend_defaults = backend.defaults()
    dt = backend_config.dt

    qubit = 0
    center_frequency_Hz = backend_defaults.qubit_freq_est[qubit]
    
    ### Collect the necessary channels
    drive_chan = pulse.DriveChannel(qubit)
    meas_chan = pulse.MeasureChannel(qubit)
    acq_chan = pulse.AcquireChannel(qubit)
    
    measure = measure_instruction(backend, qubit)
    
    # Rabi experiment parameters
    # num_rabi_points = 50
    num_rabi_points = 1

    drive_sigma_us = 0.075
    # This is a truncating parameter, because gaussians don't have a natural finite length
    drive_samples_us = drive_sigma_us*8
    # The width of the gaussian in units of dt
    drive_sigma = get_closest_multiple_of_16(drive_sigma_us * us /dt)
    # The truncating parameter in units of dt
    drive_samples = get_closest_multiple_of_16(drive_samples_us * us /dt)
    # Drive amplitude values to iterate over: 50 amplitudes evenly spaced from 0 to 0.75
    drive_amp_min = 0
    drive_amp_max = 0.75
    drive_amps = np.linspace(drive_amp_min, drive_amp_max, num_rabi_points)

    rabi_schedules = []
    for drive_amp in drive_amps:
        rabi_pulse = pulse_lib.gaussian(
            duration=drive_samples, amp=drive_amp, 
            sigma=drive_sigma, name=f"Rabi drive amplitude = {drive_amp}")
        this_schedule = pulse.Schedule(name=f"Rabi drive amplitude = {drive_amp}")
        this_schedule += Play(rabi_pulse, drive_chan)
        # Reuse the measure instruction from the frequency sweep experiment
        this_schedule += measure << this_schedule.duration
        rabi_schedules.append(this_schedule)
        break

    num_shots_per_point = 1024

    rabi_experiment_program = assemble(rabi_schedules,
                                       backend=backend,
                                       meas_level=1,
                                       meas_return='avg',
                                       shots=num_shots_per_point,
                                       schedule_los=[{drive_chan: center_frequency_Hz}]
                                       * num_rabi_points)
    print(rabi_experiment_program)
    print(type(rabi_experiment_program))
#ENDDEF

if __name__ == "__main__":
    rabi()
#ENDIF


