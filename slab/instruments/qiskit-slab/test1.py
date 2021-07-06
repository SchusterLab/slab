"""
test1.py - use a qobj to load a waveform
"""

from enum import Enum
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from qiskit import assemble
from qiskit import IBMQ
from qiskit import pulse
from qiskit.tools.monitor import job_monitor

sys.path.append("/Users/thomaspropson/repos/qsystem0/pynq")
from qsystem0_asm2 import ASM_Program

DPI = 300

# unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds

class PulseType(Enum):
    SAMPLE = 1
    PARAMETRIC = 2
    ACQUIRE = 3
#ENDCLASS

class Pulse(object):
    """
    Flat, intermediate representation of an instruction.

    Fields:
    addr :: int - the memory address where the pulse is stored
    channel_name :: str - qiskit pulse's name for the channel
    duration :: int - sample length
    name :: str - the name of the pulse
    pulse_type :: PulseType - the type of the pulse
    t0 :: int - the start time of the pulse in the schedule
    tf :: int - the final time of the pulse in the schedule
    parameters :: dict - parameters for a parametric pulse
    _samples :: array - samples for the pulse, lazily constructed
    """
    def __init__(self, channel_name, duration, name, pulse_type, t0, addr=0,
                 parameters=None, samples=None):
        super().__init__(self)
        self.addr = addr
        self.channel_name = channel_name
        self.duration = duration
        self.name = name
        self.t0 = t0
        self.tf = t0 + duration
        self.pulse_type = pulse_type
        self.parameters = parameters
        self._samples = samples
    #ENDDEF

    def __eq__(self, pulse):
        return (
            type(pulse) == Pulse
            and self.name == pulse.name
            and self.parameters == pulse.parameters
        )
    #ENDDEF

    def samples(self):
        samples_ = None
        if self._samples is not None:
            # TODO: implement, probably want backend to implement
            samples_ = None
        else:
            samples_ = self._samples
        #ENDIF
        return samples_    
    #ENDDEF
#ENDCLASS

def pulses_from_instruction(instruction, qobj_dict):
    name = instruction["name"]
    t0 = instruction["t0"]
    pulses = []
    # check for acquire pulse
    if name == "acquire":
        for i in range(len(instruction["qubits"])):
            qubit_idx = instruction["qubits"][i]
            addr = instruction["memory_slot"][i]
            channel_name = "acquire{}".format(qubit_idx)
            duration = instruction["duration"]
            t0 = instruction["t0"]
            pulse = Pulse(channel_name, duration, name, PulseType.ACQUIRE, t0, addr=addr)
            pulses.append(pulse)
        #ENDFOR
    #ENDIF
    # check for sample pulse
    if len(pulses) == 0:
        sample_pulse_lib = qobj_dict["config"]["pulse_library"]
        for i in range(len(sample_pulse_lib)):
            pulse_spec = sample_pulse_lib[i]
            if pulse_spec["name"] == name:
                pulse = Pulse(channel, pulse_spec["duration"], name, PulseType.SAMPLE, t0,
                              samples=pulse_spec["samples"])
                pulses.append(pulse)
            #ENDIF
        #ENDFOR
    #ENDIF
    # check for parametric pulse
    parametric_pulse_lib = qobj_dict["config"]["parametric_pulses"]
    if len(pulses) == 0:
        for i in range(len(parametric_pulse_lib)):
            if name == parametric_pulse_lib[i]:
                channel = instruction.get("channel")
                pulse = Pulse(channel, instruction["parameters"]["duration"], name, True, t0,
                              parameters=instruction["parameters"])
                pulses.append(pulse)
            #ENDIF
        #ENDFOR
    #ENDIF
    if len(pulses) == 0:
        raise(Exception("pulses_from_instruction failed"))
    #ENDIF
    return pulses
#ENDDEF

def pulses_from_expt_config(expt_config, qobj_dict):
    # aggregate pulses
    pulses = {}
    for channel_name in backend_config.channels.keys():
        pulses[channel_name] = list()
    #ENDFOR
    for instruction in expt_config["instructions"]:
        pulses_ = pulses_from_instruction(instruction, qobj_dict)
        # assign memory address
        for pulse in pulses_:
            # if this is an acquire pulse it doesn't need to be assigned a memory address
            if pulse.pulse_type == PulseType.ACQUIRE:
                continue
            # if there is already a pulse in this channel, grab the next
            # available memory address
            elif length(pulses[pulse.channel_name]) > 0:
                pulse_prev = pulses[pulse.channel_name][-1]
                addr = pulse_prev.addr + pulse_prev.duration
                # check if pulses have more samples than can fit into memory
                # TODO: get max length from signal generator
                if addr + pulse.duration > 16 * (2 ** 12):
                    raise Error("pulses on channel {} exceed memory capacity"
                                "".format(pulse.channel_name))
                #ENDIF
                pulse.addr = addr
            # otherwise, this pulse goes at address 0
            else:
                addr = 0
                pulse.addr = addr
            #ENDIF
            pulses[pulse.channel_name].append(pulse)
        #ENDFOR
    #ENDFOR
    return pulses
#ENDDEF

class QiskitPulseProgram(object):
    def __init__(self, program, pulses):
        super().__init__(self)
        self.program = program
        self.pulses = pulses
    #ENDDEF
#ENDCLASS

def build_asm_program(pulses):
    p = ASM_Program()
    p.synci(1000)
    p.end()
    return p
#ENDDEF

def get_closest_multiple_of_16(num):
    return int(num + 8 ) - (int(num + 8 ) % 16)
#ENDDEF

def rabi():
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    backend = provider.get_backend('ibmq_armonk')
    backend_config = backend.configuration()
    backend_defaults = backend.defaults()
    dt = backend_config.dt

    # collect channels
    qubit_idx = 0
    drive_chan = pulse.DriveChannel(qubit_idx)
    meas_chan = pulse.MeasureChannel(qubit_idx)
    acq_chan = pulse.AcquireChannel(qubit_idx)

    # measure pulse
    res_freq_Hz = backend_defaults.meas_freq_est[qubit_idx]
    meas_duration = 22400
    meas_sigma = 64
    meas_amp = -0.3584733362723958 + 0.05040701520361846j
    meas_width = 22144
    meas_pulse = pulse.library.GaussianSquare(meas_duration, meas_amp, meas_sigma, meas_width)

    # Rabi experiment parameters
    qubit_freq_Hz = backend_defaults.qubit_freq_est[qubit_idx]
    drive_sigma_us = 0.075
    # The width of the gaussian in units of dt
    drive_sigma = get_closest_multiple_of_16(drive_sigma_us * us /dt)
    # This is a truncating parameter, because gaussians don't have a natural finite length
    drive_samples_us = drive_sigma_us * 8
    # The truncating parameter in units of dt
    drive_samples = get_closest_multiple_of_16(drive_samples_us * us /dt)
    # Drive amplitude values to iterate over: 50 amplitudes evenly spaced from 0 to 0.75
    num_rabi_points = 1
    drive_amp_min = 0
    drive_amp_max = 0.75
    drive_amps = np.linspace(drive_amp_min, drive_amp_max, num_rabi_points)

    rabi_schedules = []
    for drive_amp in drive_amps:
        rabi_pulse = pulse.library.Gaussian(
            duration=drive_samples, amp=drive_amp, 
            sigma=drive_sigma, name="Rabi drive amplitude = {}".format(drive_amp)
        )
        schedule = pulse.Schedule(name="Rabi drive amplitude = {}".format(drive_amp))
        schedule += pulse.Play(rabi_pulse, drive_chan)
        meas_start = schedule.duration
        schedule += pulse.Play(meas_pulse, meas_chan) << meas_start
        schedule += pulse.Acquire(meas_duration, acq_chan,
                                  pulse.MemorySlot(0)) << meas_start
        rabi_schedules.append(schedule)
    #ENDFOR

    num_shots_per_point = 1024

    rabi_experiment_program = assemble(
        rabi_schedules, backend=backend, meas_level=1,
        meas_return='avg', shots=num_shots_per_point,
        schedule_los=[{drive_chan: qubit_freq_Hz,
                       meas_chan: res_freq_Hz}] * num_rabi_points
    )

    qobj_dict = rabi_experiment_program.to_dict()
    
    # RFSOC
    programs = list()
    for expt_config in qobj_dict["experiments"]:
        pulses = pulses_from_expt_config(expt_config, qobj_dict)
        program = QiskitPulseProgram(pulses)
        programs.append(program)
    #ENDFOR

    return qobj_dict, backend_config
    
#ENDDEF

if __name__ == "__main__":
    rabi()
#ENDIF

