"""
rfsoc_experiment.py
"""

from enum import Enum
import logging

import numpy as np
from qiskit.qobj.utils import MeasLevel, MeasReturnType

from ..experiment import PulseExperiment
from .qsystem0_asm2 import ASM_Program
from .qsystem0 import freq2reg

class InstructionType(Enum):
    SAMPLE = 1
    PARAMETRIC = 2
    ACQUIRE = 3
    SET_FREQUENCY = 4
    SET_PHASE = 5
#ENDCLASS

class Instruction(object):
    """
    Flat, intermediate representation of an instruction.

    Fields:
    addr :: int - the memory address where the pulse is stored
    ch_idx :: int - channel that the pulse should be played on
    constant :: bool - flag for constant parametric pulse
    duration :: int - sample length in DAC units
    frequency :: float - the frequency for a set frequency instruction
    memory :: int - qiskit storage identifier
    name :: str - the name of the pulse
    inst_type :: InstructionType - the type of the pulse
    t0 :: int - the start time of the pulse in the schedule in DAC units
    tf :: int - the final time of the pulse in the schedule in DAC units
    parameters :: dict - the params for a parametric pulse
    phase :: float - the phase for a set phase instruction
    rdds :: bool - flag if this operation is for a readout DDS
    samples :: array - samples for the pulse, lazily constructed
    """
    def __init__(self, ch_idx, duration, name, inst_type, t0, addr=0,
                 constant=False, frequency=0., memory=0, parameters=None,
                 phase=0., samples=None, rdds=False):
        super().__init__()
        self.addr = addr
        self.ch_idx = ch_idx
        self.constant = constant
        self.duration = duration
        self.frequency = frequency
        self.name = name
        self.t0 = t0
        self.tf = t0 + duration
        self.inst_type = inst_type
        self.parameters = parameters
        self.phase = phase
        self.rdds = rdds
        self.memory = memory
        self.samples = samples
    #ENDDEF

    def __eq__(self, inst):
        return (
            type(inst) == Instruction
            and self.name == inst.name
            and self.parameters == inst.parameters
        )
    #ENDDEF
#ENDCLASS

class RFSoCExperiment(PulseExperiment):
    def __init__(self, qobj, qexpt, backend, log_extra):
        # initialize
        super().__init__(qobj, qexpt, backend, log_extra)
        # parse
        self.insts = self._insts(log_extra)
        self.asm_program = self._asm_program()
        self.backend.logger.log(logging.DEBUG, "asm\n{}".format(self.asm_program), extra=log_extra)
    #ENDDEF

    def _inst_freq(self, ch_idx, frequency, t0, insts, stats, log_extra, prefix="parsed"):
        """
        This is a helper function for building an InstructionType.SET_FREQUENCY function
        """
        duration = 0
        name = "setf"
        inst = Instruction(ch_idx, duration, name, InstructionType.SET_FREQUENCY, t0,
                           frequency=frequency)
        insts[ch_idx].append(inst)
        stats[ch_idx]["frequency"] = frequency
        self.backend.logger.log(logging.DEBUG, "{} InstructionType.SET_FREQUENCY ch: {}, "
                                "t0: {}, f: {}".format(prefix, ch_idx, t0, frequency),
                                extra=log_extra)
    #ENDDEF

    def _inst_pulse(self, constant, duration, inst_type, name, parameters,
                    samples, t0, qinst, insts, stats, log_extra):
        ch_name = qinst.ch
        ch_idx = self.backend.ch_name_idx[ch_name]
        addr = stats[ch_idx]["addr"]
        if addr + duration > self.backend.dac_max_memory:
            raise Error("maximum memory exceeded on channel {} DAC"
                        "".format(ch_idx))
        #ENDIF
        inst = Instruction(ch_idx, duration, name, inst_type, t0,
                           addr=addr, constant=constant, parameters=parameters,
                           samples=samples)
        #ENDIF
        insts[ch_idx].append(inst)
        stats[ch_idx]["t"] = t0 + duration
        stats[ch_idx]["addr"] = stats[ch_idx]["addr"] + duration
        self.backend.logger.log(logging.DEBUG, "parsed {} ch: {}, t0: {}, d: {}, n: {}"
                                "".format(inst_type, ch_idx, t0, duration, name),
                                extra=log_extra)
        #ENDIF
    #ENDDEF

    def _insts(self, log_extra):
        """
        This function takes an experiment in a qiskit pulse representation
        and parses it into an `Instruction` representation.
        """
        # a list of `Instruction`s for each channel
        insts = {}
        # running parameters for each channel to aid in parsing
        stats = {}
        # initialize insts, stats
        for ch_name in self.backend.ch_name_idx.keys():
            ch_idx = self.backend.ch_name_idx[ch_name]
            insts[ch_idx] = list()
            stats[ch_idx] = {
                "frequency": 0,
                "phase": 0,
                "t": 0,
                "addr": 0,
            }
        #ENDFOR
        # get default freq for all DAC channels
        meas_freqs = self.config.meas_lo_freq
        qubit_freqs = self.config.qubit_lo_freq
        # set default freq for all DAC channels
        for (i, frequency) in enumerate(qubit_freqs):
            ch_name = "d{}".format(i)
            ch_idx = self.backend.ch_name_idx[ch_name]
            t0 = 0
            self._inst_freq(ch_idx, frequency, t0, insts, stats, log_extra, prefix="init")
        #ENDFOR
        for (i, frequency) in enumerate(meas_freqs):
            ch_name = "m{}".format(i)
            ch_idx = self.backend.ch_name_idx[ch_name]
            t0 = 0
            self._inst_freq(ch_idx, frequency, t0, insts, stats, log_extra, prefix="init")
            # TODO: this will change in a new RFSoC firmware version
            # a more general approach for handling something
            # like this could be implemented by a method in self.backend
            # set the corresponding readout dds
            rdds_ch_idx = self.backend.ch_idx_rdds[ch_idx]
            self._inst_freq(rdds_ch_idx, frequency, t0, insts, stats, log_extra, prefix="init rdds")
        #ENDFOR
        # parse experiment instructions
        for qinst in self.qexpt.instructions:
            name = qinst.name
            t0 = qinst.t0
            # check for delay
            if name == "delay":
                # we don't need to account for delays,
                # they will be reflected in `t0` in future instructions
                continue
            # check for acquire
            elif name == "acquire":
                for i in range(len(qinst.qubits)):
                    qubit_idx = qinst.qubits[i]
                    memory = qinst.memory_slot[i]
                    ch_name = "a{}".format(qubit_idx)
                    ch_idx = self.backend.ch_name_idx[ch_name]
                    duration = qinst.duration
                    # TODO the case where there is more than one acquire statement on
                    # each channel is not handled in the current RFSoC firmware or
                    # this parser
                    inst = Instruction(ch_idx, duration, name, InstructionType.ACQUIRE,
                                       t0, memory=memory)
                    insts[ch_idx].append(inst)
                    stats[ch_idx]["t"] = t0 + duration
                    self.memory_count += 1
                    self.backend.logger.log(logging.DEBUG, "parsed InstructionType.ACQUIRE ch: {} "
                                            "d: {}, t0: {}".format(ch_idx, duration, t0),
                                            extra=log_extra)
                #ENDFOR
            # check for set/shift frequency
            elif name == "setf" or name == "shiftf":
                ch_name = qinst.ch
                ch_idx = self.backend.ch_name_idx[ch_name]
                if name == "setf":
                    frequency = qinst.frequency
                else:
                    frequency = stats[ch_idx]["frequency"] + qinst.frequency
                #ENDIF
                self._inst_freq(ch_idx, frequency, t0, insts, stats, log_extra)
                # if this is a measurement channel, change the readout dds frequency
                if ch_name[0] == 'm':
                    rdds_ch_idx = self.backend.ch_idx_rdds[ch_idx]
                    self._inst_freq(rdds_ch_idx, frequency, t0, insts, stats, log_extra,
                                    prefix="parsed rdds")
                #ENDIF
            # check for set/shift phase
            elif name == "setp" or name == "shiftp":
                ch_name = qinst.ch
                ch_idx = self.backend.ch_name_idx[ch_name]
                if name == "setp":
                    phase = qinst.phase
                elif name == "shiftp":
                    phase = stats[ch_idx]["phase"] + qinst.phase
                #ENDIF
                inst = Instruction(ch_idx, 0, name, InstructionType.SET_PHASE, t0,
                                   phase=phase)
                insts[ch_idx].append(inst)
                stats[ch_idx]["phase"] = phase
                self.backend.logger.log(logging.DEBUG, "parsed InstructionType.SET_PHASE ch: {}, "
                                        "t0: {}, p: {}".format(ch_idx, t0, phase),
                                        extra=log_extra)
            # check for parametric pulse
            elif name == "parametric_pulse":
                pulse_shape = qinst.pulse_shape
                constant = True if pulse_shape == "constant" else False
                duration = qinst.parameters["duration"]
                inst_type = InstructionType.PARAMETRIC
                parameters = qinst.parameters
                samples = self.backend.samples(pulse_shape, parameters)
                self._inst_pulse(constant, duration, inst_type, pulse_shape, parameters,
                                 samples, t0, qinst, insts, stats, log_extra)
            # check for sample pulse, otherwise instruction unrecognized, raise exception
            else:
                # check for sample pulse
                inst_type = None
                for i in range(len(self.config["pulse_library"])):
                    pulse_spec = self.config["pulse_library"][i]
                    if pulse_spec["name"] == name:
                        constant = False
                        samples = pulse_spec["samples"]
                        duration = len(samples)
                        inst_type = InstructionType.SAMPLE
                        parameters = None
                        self._inst_pulse(constant, duration, inst_type, name,
                                         parameters, samples, t0, qinst, insts,
                                         stats, log_extra)
                    #ENDIF
                #ENDFOR
                # instruction unrecognized, raise exception
                if inst_type is None:
                    raise(Exception("Unrecognized instruction: {}".format(qinst.name)))        
                #ENDIF
            #ENDIF
        #ENDFOR
        return insts
    #ENDDEF

    def _asm_program(self):
        """
        This function takes an experiment in an `Instruction` representation
        and generates a tproc ASM program.
        """
        p = ASM_Program()
        # declare constants
        p_i = self.backend.misc_page[0]
        r_i = self.backend.misc_reg[0]
        # sync to begin
        p.synci(self.backend.tproc_initial_cycle_offset, "init delay")
        # loop `shots` number of times
        p.memri(p_i, r_i, 0, "shots")
        p.label("LOOP_I")
        # set phase to 0 and gain to 1 for all DAC channels
        for ch_name in self.backend.ch_name_idx.keys():
            ch_char = ch_name[0]
            if ch_char == 'm' or ch_char == 'd':
                ch_idx = self.backend.ch_name_idx[ch_name]
                p_ch = self._ch_page(ch_idx)
                r_phase = self._ch_reg(ch_idx, "phase")
                r_gain = self._ch_reg(ch_idx, "gain")
                p.regwi(p_ch, r_phase, 0, "ch {} phase".format(ch_idx))
                p.regwi(p_ch, r_gain, self.backend.tproc_max_gain, "ch {} gain".format(ch_idx))
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
                    # determine registers
                    p_ch = self._ch_page(ch_idx)
                    r_freq = self._ch_reg(ch_idx, "freq")
                    r_phase = self._ch_reg(ch_idx, "phase")
                    r_addr = self._ch_reg(ch_idx, "addr")
                    r_gain = self._ch_reg(ch_idx, "gain")
                    r_mode = self._ch_reg(ch_idx, "mode")
                    r_t = self._ch_reg(ch_idx, "t")
                    # write address
                    p.regwi(p_ch, r_addr, inst.addr, "ch {} addr".format(ch_idx))
                    # determine mode code
                    # write gain for special case of a constant pulse to minimize memory impact
                    # TODO: `np.isreal` is checking that the complex part of
                    # the amplitude is zero. this check will be deprecated in future
                    # firmware versions; currently, however, the DDS can't have a complex value
                    if inst.constant and not np.isreal(inst.parameters["amp"]):
                        raise Exception("DDS does not currently support complex amplitudes.")
                    #ENDIF
                    if inst.constant:
                        gain_val = int(inst.parameters["amp"].real * self.backend.tproc_max_gain)
                        p.regwi(p_ch, r_gain, gain_val, "ch {} gain".format(ch_idx))
                        mode_code = self._mode_code(
                            self.backend.stdysel_zero, self.backend.mode_oneshot,
                            self.backend.outsel_dds, inst.duration // self.backend.tproc_to_dac
                        )
                    else:
                        mode_code = self._mode_code(
                            self.backend.stdysel_zero, self.backend.mode_oneshot,
                            self.backend.outsel_tdds, inst.duration // self.backend.tproc_to_dac
                        )
                    #ENDIF
                    p.regwi(p_ch, r_mode, mode_code, "ch {} mode".format(ch_idx))
                    # write time
                    p.regwi(p_ch, r_t, inst.t0 // self.backend.tproc_to_dac,
                            "ch {} t".format(ch_idx))
                    # schedule pulse
                    p.set(ch_idx, p_ch, r_freq, r_phase, r_addr, r_gain, r_mode, r_t,
                          "ch {} play".format(ch_idx))
                    # reset gain to unity if a constant pulse was played
                    if inst.constant:
                        p.regwi(p_ch, r_gain, self.backend.tproc_max_gain,
                                "ch {} gain".format(ch_idx))
                    #ENDIF
                # listen to acquire
                elif inst.inst_type == InstructionType.ACQUIRE:
                    # determine registers
                    p_ch = self._ch_page(ch_idx)
                    p_out = self._ch_reg(ch_idx, "out")
                    # determine times
                    t_start = inst.t0 // self.backend.tproc_to_dac + self.backend.adc_trig_offset
                    t_stop = (t_start + inst.duration // self.backend.tproc_to_dac
                              + self.backend.acquire_pad)
                    # determine bit codes
                    bits_start = self._trigger_bits(0, 1, 0, 0, 0, 0)
                    bits_stop = self._trigger_bits(0, 0, 0, 0, 0, 0)
                    # schedule average buffer capture start
                    p.regwi(p_ch, p_out, bits_start, "start average buffer bits")
                    p.seti(ch_idx, p_ch, p_out, t_start, "start average buffer")
                    # schedule average buffer capture stop
                    p.regwi(p_ch, p_out, bits_stop, "stop average buffer bits")
                    p.seti(ch_idx, p_ch, p_out, t_stop, "stop average buffer")
                elif inst.inst_type == InstructionType.SET_FREQUENCY:
                    p_ch = self._ch_page(ch_idx)
                    r_freq = self._ch_reg(ch_idx, "freq")
                    # qobj frequency is in GHz, `freq2reg` accepts MHz
                    # readout dds frequency should be in ADC cycles
                    if inst.rdds:
                        freq_val = freq2reg(self.backend.soc.fs_adc, 1e3 * inst.frequency)
                    else:
                        freq_val = freq2reg(self.backend.soc.fs_dac, 1e3 * inst.frequency)
                    p.regwi(p_ch, r_freq, freq_val,
                            "ch {} freq".format(ch_idx))
                elif inst.inst_type == InstructionType.SET_PHASE:
                    p_ch = self._ch_page(ch_idx)
                    r_phase = self._ch_reg(ch_idx, "phase")
                    # TODO should have phase2reg function in qsystem0
                    phase_mod2pi = np.arctan2(np.sin(inst.phase), np.cos(inst.phase)) + np.pi
                    phase_reg = int(np.floor(phase_mod2pi / (2 * np.pi)
                                             * self.backend.tproc_max_phase))
                    p.regwi(p_ch, r_phase, phase_reg,
                            "ch {} phase".format(ch_idx))
                else:
                    raise Error("Unrecognized instruction type {}".format(inst.inst_type))
                #ENDIF
            #ENDFOR
        #ENDFOR
        # wait until next experiment
        p.synci(p.us2cycles(self.config.rep_delay), "rep delay")
        # end loop
        p.loopnz(p_i, r_i, "LOOP_I")
        # end program
        p.end()
        return p
    #ENDDEF

    def _ch_page(self, ch_idx):
        return self.backend.ch_idx_page[ch_idx]
    #ENDDEF

    def _ch_reg(self, ch_idx, name):
        return self.backend.ch_idx_reg[ch_idx][name]
    #ENDDEF

    def _mode_code(self, stdysel, mode, outsel, length):
        return (stdysel << 15) | (mode << 14) | (outsel << 12) | (length <<0)
    #ENDDEF

    def _trigger_bits(self, b15, b14, b3, b2, b1, b0):
        """
        b15 - full speed buffer
        b14 - average buffer
        b3 - PMOD4
        b2 - PMOD3
        b1 - PMOD2
        b0 - PMOD1
        """
        return (b15 << 15) | (b14 << 14) | (b3 << 3) | (b2 << 2) | (b1 << 1) | (b0 << 0) 
    #ENDDEF

    def _run(self, shots):
        """
        Execute the program.
        References:
        [0] https://github.com/openquantumhardware/qsystem0/blob/main/pynq/averager_program.py#L72
        """
        # load pulses and configure buffers
        for i, ch_idx in enumerate(self.insts.keys()):
            for inst in self.insts[ch_idx]:
                # skip loading constant pulses
                if inst.constant:
                    continue
                #ENDIF
                # load pulse
                elif (inst.inst_type == InstructionType.SAMPLE
                    or inst.inst_type == InstructionType.PARAMETRIC):
                    samples = inst.samples
                    gen = self.backend.gens[ch_idx]
                    gen.load(samples, addr=inst.addr)
                #ENDIF
                # configure average buffer
                elif inst.inst_type == InstructionType.ACQUIRE:
                    readout = self.backend.readouts[ch_idx]
                    # readout configuration to route input without frequency translation
                    readout.set_out(sel="product")
                    avg_buf = self.backend.avg_bufs[ch_idx]
                    duration_dac = inst.duration
                    duration_adc_dec = duration_dac // (self.backend.adc_to_dac
                                                        * self.backend.decimation)
                    avg_buf.config(address=inst.addr, length=duration_adc_dec)
                    avg_buf.enable()
                #ENDIF
            #ENDFOR
        #ENDFOR
        # load ASM program
        self.backend.soc.tproc.load_asm_program(self.asm_program)
        # write number of shots into memory
        self.backend.soc.tproc.single_write(addr=0, data=shots - 1)
        # run ASM program
        self.backend.soc.tproc.stop()
        self.backend.soc.tproc.start()
        # get memory for each acquire statement
        # TODO assumes ["meas_return"] == "avg"
        memory = np.zeros(self.memory_count, dtype=np.complex128)
        for i, ch_idx in enumerate(self.insts.keys()):
            for inst in self.insts[ch_idx]:
                if inst.inst_type == InstructionType.ACQUIRE:
                    # get raw data for each shot
                    ch_name = self.backend.ch_idx_name[ch_idx]
                    i_data, q_data = self.backend.soc.getAccumulated(
                        address=inst.addr,
                        length=shots
                    )
                    data = i_data + 1j * q_data
                    # average over shots if necessary
                    if self.config.meas_return == MeasReturnType.AVERAGE:
                        data = np.sum(data) / data.shape[0]
                    #ENDIF
                    memory[inst.memory] = data
                #ENDIF
            #ENDIF
        #ENDIF

        return memory
    #ENDDEF
#ENDCLASS
