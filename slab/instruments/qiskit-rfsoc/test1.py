"""
test1.py - run a rabi
"""

from enum import Enum
import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from slab import generate_file_path
from qiskit import assemble, pulse
from qiskit.tools.monitor import job_monitor
from qiskit.providers import ProviderV1 as ProviderInterface
from qiskit.providers.backend import BackendV1 as BackendInterface
from qiskit.providers.models import (
    BackendStatus, BackendProperties, PulseDefaults, GateConfig,
    QasmBackendConfiguration, PulseBackendConfiguration
)
from qiskit.providers.options import Options
from qiskit.qobj.utils import MeasLevel, MeasReturnType

sys.path.append("/home/xilinx") # for pynq
sys.path.append("/home/xilinx/repos/qsystem0/pynq")
from qsystem0 import PfbSoc
from qsystem0_asm2 import ASM_Program

DPI = 300

# unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
GHz = 1.0e9 #Hz
MHz = 1.0e6 #Hz
us = 1.0e-6 #s
ns = 1.0e-9 #s

# TODO: unclear that configuration and defaults should
# be passed to __init__. I think some pointer to where to fetch
# them form would be better. This pointer would be found in the provider.
class SLabBackendInterface(BackendInterface):
    def __init__(self, configuration, defaults, provider):
        super().__init__(configuration=configuration, provider=provider)
        self._defaults = defaults
    #ENDDEF

    def configuration(self, refresh=False):
        return self._configuration
    #ENDDEF

    def defaults(self, refresh=False):
        return self._defaults
    #ENDDEF

    @classmethod
    def _default_options(cls):
        return Options(
            shots=1024, memory=False,
            qubit_lo_freq=None, meas_lo_freq=None,
            schedule_los=None,
            meas_level=MeasLevel.KERNELED,
            meas_return=MeasReturnType.AVERAGE,
            memory_slots=None, memory_slot_size=100,
            rep_time=None, rep_delay=None,
            init_qubits=True
        )
    #ENDDEF

    def run(self, qobj):
        qobj_dict = qobj.to_dict()
        # TODO: post to actual backend
        return qobj_dict
    #ENDDEF
#ENDCLASS

class SLabProviderInterface(ProviderInterface):
    """
    References:
    [0] https://github.com/Qiskit/qiskit-ibmq-provider/blob/master/
        qiskit/providers/ibmq/accountprovider.py#L43
    """
    def __init__(self):
        super().__init__()
        self._backends = self._discover_remote_backends()
    #ENDDEF

    def _discover_remote_backends(self, timeout=60):
        # TODO: raw_config and raw_defaults should be fetched in BackendInterface
        # this function should search network for backends, then pass that info
        # to BackendInterface-s which are instantiated by this function
        # https://github.com/Qiskit/qiskit-terra/blob/main/qiskit
        # /providers/models/backendconfiguration.py#L493
        raw_config = {
            "backend_name": "bf3-rfsoc",
            "backend_version": "0.1",
            "n_qubits": 2,
            "basis_gates": [],
            "gates": [],
            "local": False,
            "simulator": False,
            "conditional": False,
            "open_pulse": True,
            "memory": True,
            "max_shots": np.iinfo(np.int64).max,
            "coupling_map": [[1], [0]],
            "n_uchannels": 0,
            "u_channel_lo": [],
            "meas_levels": [1],
            "qubit_lo_range": [[0,10],[0,10]], #TODO
            "meas_lo_range": [[0,10]], #TODO
            "dt": 1 / 6.144,
            "dtm": 1 / 6.144,
            "rep_times": [],
            "meas_kernels": [], #TODO
            "discriminators": [], #TODO
            "rep_delay_range": [0., np.finfo(np.float64).max],
            "dynamic_reprate_enabled": True,
            "default_rep_delay": 0.,
        }
        config = PulseBackendConfiguration.from_dict(raw_config)
        # https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/
        # providers/models/pulsedefaults.py#L164
        raw_defaults = {
            "qubit_freq_est": [5., 5.], #TODO
            "meas_freq_est": [5.], #TODO
            "buffer": 0,
            "pulse_library": [
                {"name": "gaussian", "samples": [0]},
                {"name": "gaussian_square", "samples": [0]},
            ],
            "cmd_def": [],
        }
        defaults = PulseDefaults.from_dict(raw_defaults)
        backend_ = SLabBackendInterface(config, defaults, self)
        
        # TODO: get backends_list by searching network
        backends_list = [backend_]
        backends_dict = {}
        for backend in backends_list:
            backends_dict[backend.name()] = backend
        #ENDFOR
        return backends_dict
    #ENDDEF

    def backends(self, name=None, filters=None, **kwargs):
        return self._backends
    #ENDDEF

    def get_backend(self, name):
        backend = self._backends.get(name, None)
        return backend
    #ENDDEF
#ENDCLASS


class InstructionType(Enum):
    SAMPLE = 1
    PARAMETRIC = 2
    ACQUIRE = 3
#ENDCLASS

class Instruction(object):
    """
    Flat, intermediate representation of an instruction.

    Fields:
    addr :: int - the memory address where the pulse is stored
    ch_idx :: int - channel that the pulse should be played on
    duration :: int - sample length
    name :: str - the name of the pulse
    inst_type :: InstructionType - the type of the pulse
    t0 :: int - the start time of the pulse in the schedule
    tf :: int - the final time of the pulse in the schedule
    parameters :: dict - parameters for a parametric pulse
    _samples :: array - samples for the pulse, lazily constructed
    """
    def __init__(self, ch_idx, duration, name, inst_type, t0, addr=0,
                 parameters=None, samples=None):
        super().__init__()
        self.addr = addr
        self.ch_idx = ch_idx
        self.duration = duration
        self.name = name
        self.t0 = t0
        self.tf = t0 + duration
        self.inst_type = inst_type
        self.parameters = parameters
        self._samples = samples
    #ENDDEF

    def __eq__(self, inst):
        return (
            type(inst) == Instruction
            and self.name == inst.name
            and self.parameters == inst.parameters
        )
    #ENDDEF

    def samples(self, backend):
        samples_ = None
        if self._samples is None:
            samples_ = backend.parametric_samples(self.name, self.parameters)
        else:
            samples_ = self._samples
        #ENDIF
        return samples_    
    #ENDDEF
#ENDCLASS

class Backend(object):
    def __init__(self):
        super().__init__()
    #ENDDEF

    # TODO: implement this, likely want to pull from a more
    # centralized place. This can probably be moved out of here
    # and come from a pulse lib that inherits from ibm's pulse lib
    def parametric_samples(self, name, params, type_=np.int16):
        if name == "gaussian":
            samples = 0
        #ENDIF
        return samples.astype(type_)
    #ENDDEF
#ENDCLASS

class MyBackend(Backend):
    # TODO: these constants probably go in a config file that is read in
    TPROC_UNITY_GAIN = 32767
    TPROC_INITIAL_TIME_OFFSET = 1000
    ADC_TRIG_OFFSET = 256
    DAC_MAX_MEMORY = 16 * (2 ** 12)
    CH_NAME_IDX = {
        "d0": 4, # qubit
        "d1": 2, # cavity
        "m0": 3, # readout
        "a0": 1, # acquire readout
    }
    
    def __init__(self):
        super().__init__()
        self.soc = PfbSoc("/home/xilinx/repos/qsystem0/pynq/qsystem_0.bit")
        self.gens = {
            1: soc.gen0,
            2: soc.gen1,
            3: soc.gen2,
            4: soc.gen3,
        }
    #ENDDEF

    def run(self, qobj_dict):
        programs = list()
        for expt_dict in qobj_dict["experiments"]:
            program = QiskitPulseProgram(qobj_dict, expt_dict, self)
            programs.append(program)
            program.run()
        #ENDFOR
        # TOOD: run each program, send output back to user
    #ENDDEF
#ENDCLASS

class QiskitPulseProgram(object):
    def __init__(self, qobj_dict, expt_dict, backend):
        super().__init__()
        self.qobj_dict = qobj_dict
        self.expt_dict = expt_dict
        self.backend = backend
        self.insts = self._get_insts()
        self.asm_program = self._get_asm_program()
    #ENDDEF

    def _get_insts_from_qinst(self, qinst):
        name = qinst["name"]
        t0 = qinst["t0"]
        insts = []
        # check for acquire pulse
        if name == "acquire":
            for i in range(len(qinst["qubits"])):
                qubit_idx = qinst["qubits"][i]
                addr = qinst["memory_slot"][i]
                ch_name = "a{}".format(qubit_idx)
                ch_idx = self.backend.CH_NAME_IDX[ch_name]
                duration = qinst["duration"]
                t0 = qinst["t0"]
                inst = Instruction(ch_idx, duration, name, InstructionType.ACQUIRE, t0, addr=addr)
                insts.append(inst)
            #ENDFOR
        #ENDIF
        # check for sample pulse
        sample_pulse_lib = self.qobj_dict["config"]["pulse_library"]
        if len(insts) == 0:
            for i in range(len(sample_pulse_lib)):
                pulse_spec = sample_pulse_lib[i]
                if pulse_spec["name"] == name:
                    ch_name = qinst.get("ch")
                    ch_idx = self.backend.CH_NAME_IDX[ch_name]
                    samples = pulse_spec["samples"]
                    duration = len(samples)
                    inst = Instruction(ch_idx, duration, name, InstructionType.SAMPLE, t0,
                                       samples=samples)
                    insts.append(inst)
                #ENDIF
            #ENDFOR
        #ENDIF
        # check for parametric pulse
        parametric_pulse_lib = self.qobj_dict["config"]["parametric_pulses"]
        if len(insts) == 0:
            for i in range(len(parametric_pulse_lib)):
                if name == parametric_pulse_lib[i]:
                    ch_name = qinst.get("ch")
                    ch_idx = self.backend.CH_NAME_IDX[ch_name]
                    inst = Instruction(ch_idx, qinst["parameters"]["duration"], name,
                                       InstructionType.PARAMETRIC, t0,
                                  parameters=qinst["parameters"])
                    insts.append(inst)
                #ENDIF
            #ENDFOR
        #ENDIF
        # TODO: implement more instruction types like set_freq, set_phase, etc.
        if len(insts) == 0:
            raise(Exception("_get_insts_from_qinst failed"))
        #ENDIF
        return insts
    #ENDDEF

    def _get_insts(self):
        insts = {}
        for ch_name in self.backend.CH_NAME_IDX.keys():
            ch_idx = self.backend.CH_NAME_IDX[ch_name]
            insts[ch_idx] = list()
        #ENDFOR
        for qinst in self.expt_dict["instructions"]:
            # get, possibly multiple, `Instruction` objects from the qiskit instruction
            insts_ = self._get_insts_from_qinst(qinst)
            # assign memory addresses to the insts
            for inst in insts_:
                # if this is an acquire pulse it has already been assigned a memory address
                if inst.inst_type == InstructionType.ACQUIRE:
                    continue
                # if there is already a pulse in this channel, grab the next
                # available memory address
                elif len(insts[inst.ch_idx]) > 0:
                    inst_prev = insts[inst.ch_idx][-1]
                    addr = inst_prev.addr + inst_prev.duration
                    # check if insts have more samples than can fit into memory
                    # TODO: get max length from signal generator
                    if addr + inst.duration > self.backend.DAC_MAX_MEMORY:
                        raise Error("pulses on channel {} exceed memory capacity"
                                    "".format(inst.ch_idx))
                    #ENDIF
                    inst.addr = addr
                # otherwise, this inst goes at address 0
                else:
                    inst.addr = 0
                #ENDIF
                # append this inst to the channel
                insts[inst.ch_idx].append(inst)
            #ENDFOR
        #ENDFOR
        return insts
    #ENDDEF

    def _get_asm_program(self):
        """
        Build a tproc ASM program from the insts.
        """
        p = ASM_Program()
        # sync to begin
        p.synci(self.backend.TPROC_INITIAL_TIME_OFFSET)
        # loop `shots` number of times
        # TODO: remove hardcoding register and register page
        ri = 13
        rip = 0
        p.regwi(rip, ri, self.qobj_dict["config"]["shots"] - 1)
        p.label("LOOP_I")
        # get default freq for all DAC channels
        meas_freqs = self.qobj_dict["config"]["meas_lo_freq"]
        qubit_freqs = self.qobj_dict["config"]["qubit_lo_freq"]
        # get program specific freq for all DAC channels if set
        if "config" in self.expt_dict:
            expt_config_dict = self.expt_dict["config"]
            if "meas_lo_freq" in expt_config_dict:
                meas_freqs = expt_config_dict["meas_lo_freq"]
            #ENDIF
            if "qubit_lo_freq" in expt_config_dict:
                qubit_freqs = expt_config_dict["qubit_lo_freq"]
            #ENDIF
        #ENDIF
        # set freq for all DAC channels
        for (i, freq) in enumerate(qubit_freqs):
            ch_name = "d{}".format(i)
            ch_idx = self.backend.CH_NAME_IDX[ch_name]
            rp = p.ch_page(ch_idx)
            r_freq = p.sreg(ch_idx, "freq")
            p.regwi(rp, r_freq, freq, "freq")
        #ENDFOR
        for (i, freq) in enumerate(meas_freqs):
            ch_name = "m{}".format(i)
            ch_idx = self.backend.CH_NAME_IDX[ch_name]
            rp = p.ch_page(ch_idx)
            r_freq = p.sreg(ch_idx, "freq")
            p.regwi(rp, r_freq, freq, "freq")
        #ENDFOR
        # set phase to 0 and gain to 1 for all DAC channels
        for ch_name in self.backend.CH_NAME_IDX.keys():
            ch_char = ch_name[0]
            if ch_char == 'm' or ch_char == 'd':
                ch_idx = self.backend.CH_NAME_IDX[ch_name]
                rp = p.ch_page(ch_idx)
                r_phase = p.sreg(ch_idx, "phase")
                r_gain = p.sreg(ch_idx, "gain")
                p.regwi(rp, r_phase, 0, "phase")
                p.regwi(rp, r_gain, self.backend.TPROC_UNITY_GAIN, "gain")
            #ENDIF
        #ENDFOR
        # execute instructions
        # for each channel
        for i, ch_idx in enumerate(self.insts.keys()):
            # for each inst in the channel
            for inst in self.insts[ch_idx]:
                # play sample and parametric pulses
                if (inst.inst_type == InstructionType.SAMPLE
                    or inst.inst_type == InstructionType.PARAMETRIC):
                    p.set_wave(ch=ch_idx, addr=inst.addr, length=inst.duration, t=inst.t0,
                               stdysel=1, mode=0, outsel=0, play=True)
                # listen to acquire
                elif inst.inst_type == InstructionType.ACQUIRE:
                    # TODO: support acquiring on different ADCs
                    t0 = inst.t0 + self.backend.ADC_TRIG_OFFSET
                    tf = t0 + inst.duration
                    p.seti_trigger(t0, t1=0, t14=1, t15=0)
                    p.seti_trigger(tf, t1=0, t14=0, t15=0)
                # TODO: handle other inst types
                #ENDIF
            #ENDFOR
        #ENDFOR
        # sync all channels and wait until next experiment
        p.delay(p.us2cycles(self.qobj_dict["config"]["rep_delay"]))
        # end loop
        p.loopnz(rip, ri, "LOOP_I")
        # end program
        p.end()
        return p
    #ENDDEF

    def run(self):
        """
        Execute the program.
        References:
        [0] https://github.com/openquantumhardware/qsystem0/blob/main/pynq/averager_program.py#L72
        """
        # load sample and parametric pulses
        for i, ch_idx in enumerate(self.insts.keys()):
            for inst in self.insts[ch_idx]:
                if (inst.inst_type == InstructionType.SAMPLE
                    or inst.inst_type == InstructionType.PARAMETRIC):
                    samples = inst.samples(self.backend)
                    gen = self.backend.gens[ch_idx]
                    gen.load(samples, addr=inst.addr)
                #ENDIF
            #ENDFOR
        #ENDFOR
        # readout configuration to route input without frequency translation
        self.backend.soc.readout.set_out(sel="product")
        # configure a buffer for every acquire statement
        for i, ch_idx in enumerate(self.insts.keys()):
            for inst in insts[ch_idx]:
                # TODO: currently assumes there is only one acquire statement
                if inst.inst_type == InstructionType.ACQUIRE:
                    avg_buf = self.backend.soc.avg_bufs[ch_idx]
                    avg_buf.config_buf(address=inst.addr, length=inst.duration)
                    avg_buf.config_avg(address=inst.addr, length=inst.duration)
                    avg_buf.enable_buf()
                    avg_buf.enable_avg()
                #ENDIF
            #ENDFOR
        #ENDFOR
        # load ASM program
        self.backend.soc.tproc.load_asm_program(self.asm_program)
        # run ASM program
        self.backend.soc.tproc.stop()
        self.backend.soc.tproc.single_write(addr=1, data=0)
        self.backend.soc.tproc.start()
        # wait for completion
        while True:
            addr = soc.tproc.single_read(addr=1)
            if addr == 1:
                di, dq = soc.getAccumulated(addr=addr, length=1)
            #ENDIF
        #ENDWHILE
        return di, dq
    #ENDDEF
#ENDCLASS

def get_closest_multiple_of_16(num):
    return int(num + 8 ) - (int(num + 8 ) % 16)
#ENDDEF

def rs():
    provider = SLabProviderInterface()
    backend = provider.get_backend("bf3-rfsoc")
    backend_config = backend.configuration()
    backend_defaults = backend.defaults()
    dt = backend_config.dt #s

    qubit_idx = 0
    cavity_chan = pulse.DriveChannel(1)
    meas_chan = pulse.MeasureChannel(qubit_idx)
    acq_chan = pulse.AcquireChannel(qubit_idx)
    
    meas_amp = 0.125
    meas_duration = get_closest_multiple_of_16(2000 * 16) #dts
    meas_pulse = pulse.library.Constant(meas_duration, meas_amp)
    
    meas_freq_count = 1
    meas_freq_start = 90 * MHz #Hz
    meas_freq_stop = 110 * MHz #Hz
    meas_freqs = np.linspace(meas_freq_start, meas_freq_stop, meas_freq_count)
    
    schedule = pulse.Schedule(name="Frequency sweep")
    schedule += pulse.Play(meas_pulse, meas_chan)
    schedule += pulse.Acquire(meas_duration, acq_chan, pulse.MemorySlot(0))
    
    # fig = schedule.draw()
    # plot_file_path = generate_file_path(".", "rs_sched", "png")
    # plt.savefig(plot_file_path)
    # print(f"plotted schedule to {plot_file_path}")

    num_shots = 1000
    rep_delay = 200 * us #s
    schedule_los = list()
    for meas_freq in meas_freqs:
        schedule_los.append({
            meas_chan: meas_freq,
        })
    #ENDFOR    
    program = assemble(
        schedule, backend=backend, meas_level=1,
        meas_return="avg", shots=num_shots,
        schedule_los=schedule_los,
        rep_delay=rep_delay,
    )
    job = backend.run(program)

    # TODO: this should happen server side
    backend_ = MyBackend()
    ret = backend_.run(job)
    
    return
#ENDDEF

def rabi():
    provider = SLabProviderInterface()
    backend = provider.get_backend("bf3-rfsoc")
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

    program = assemble(
        rabi_schedules, backend=backend, meas_level=1,
        meas_return='avg', shots=num_shots_per_point,
        schedule_los=[{drive_chan: qubit_freq_Hz,
                       meas_chan: res_freq_Hz}] * num_rabi_points
    )
    
    job = backend.run(program)

    # TODO: this should happen server side
    backend_ = MyBackend()
    ret = backend_.run(job)

    return ret
#ENDDEF

if __name__ == "__main__":
    rabi()
#ENDIF

