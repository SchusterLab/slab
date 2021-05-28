"""
rfsoc_experiment.py
"""

from .experiment import PulseExperiment

class RFSoCExperiment(PulseExperiment):
    def __init__(self, qobj_dict, expt_dict, backend):
        # initialize
        super().__init__(qobj_dict, expt_dict, backend)
        self.shots_per_set = qobj_dict["config"]["shots_per_set"]
        self.shots = qobj_dict["config"]["shots"]
        self.sets = int(np.ceil(shots / shots_per_set))
        # parse
        self.insts = self._insts()
        self.asm_program = self._asm_program()
    #ENDDEF

    def _inst_freq(self, ch_idx, frequency, t0, insts, stats, prefix="parsed"):
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
              "t0: {}, f: {}".format(prefix, ch_idx, t0, frequency))
    #ENDDEF

    def _insts(self):
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
        # set default freq for all DAC channels
        for (i, frequency) in enumerate(qubit_freqs):
            ch_name = "d{}".format(i)
            ch_idx = self.backend.ch_name_idx[ch_name]
            t0 = 0
            self._inst_freq(ch_idx, frequency, t0, insts, stats, prefix="init")
        #ENDFOR
        for (i, frequency) in enumerate(meas_freqs):
            ch_name = "m{}".format(i)
            ch_idx = self.backend.ch_name_idx[ch_name]
            t0 = 0
            self._inst_freq(ch_idx, frequency, t0, insts, stats, prefix="init")
            # TODO: this will change in a new RFSoC firmware version
            # a more general approach for handling something
            # like this could be implemented by a method in self.backend
            # set the corresponding readout dds
            rdds_ch_idx = self.backend.ch_idx_rdds[ch_idx]
            self._inst_freq(rdds_ch_idx, frequency, t0, insts, stats, prefix="init rdds")
        #ENDFOR
        # parse experiment instructions
        for qinst in self.expt_dict["instructions"]:
            name = qinst["name"]
            t0 = qinst["t0"]
            # check for delay
            if name == "delay":
                # we don't need to account for delays,
                # they will be reflected in `t0` in future instructions
                continue
            # check for acquire
            elif name == "acquire":
                for i in range(len(qinst["qubits"])):
                    qubit_idx = qinst["qubits"][i]
                    addr = qinst["memory_slot"][i]
                    ch_name = "a{}".format(qubit_idx)
                    ch_idx = self.backend.ch_name_idx[ch_name]
                    duration = qinst["duration"]
                    inst = Instruction(ch_idx, duration, name, InstructionType.ACQUIRE,
                                       t0, addr=addr)
                    insts[ch_idx].append(inst)
                    stats[ch_idx]["t"] = t0 + duration
                    # logging.log(11, f"parsed InstructionType.ACQUIRE ch: {ch_idx}, "
                    #            "d: {duration}, t0: {t0}")
                    print("parsed InstructionType.ACQUIRE ch: {}, "
                          "t0: {}, d: {}".format(ch_idx, t0, duration))
                #ENDFOR
            # check for set/shift frequency
            elif name == "setf" or name == "shiftf":
                ch_name = qinst["ch"]
                ch_idx = self.backend.ch_name_idx[ch_name]
                if name == "setf":
                    frequency = qinst["frequency"]
                else:
                    frequency = stats[ch_idx]["frequency"] + qinst["frequency"]
                #ENDIF
                self._inst_freq(ch_idx, frequency, t0, insts, stats)
                # if this is a measurement channel, change the readout dds frequency
                if ch_name[0] == 'm':
                    rdds_ch_idx = self.backend.ch_idx_rdds[ch_idx]
                    self._inst_freq(rdds_ch_idx, frequency, t0, insts, stats, prefix="parsed rdds")
                #ENDIF
            # check for set/shift phase
            elif name == "setp" or name == "shiftp":
                ch_name = qinst["ch"]
                ch_idx = self.backend.ch_name_idx[ch_name]
                if name == "setp":
                    phase = qinst["phase"]
                elif name == "shiftp":
                    phase = stats[ch_idx]["phase"] + qinst["phase"]
                #ENDIF
                inst = Instruction(ch_idx, 0, name, InstructionType.SET_PHASE, t0,
                                   phase=phase)
                insts[ch_idx].append(inst)
                stats[ch_idx]["phase"] = phase
                # logging.log(11, )
                print("parsed InstructionType.SET_PHASE ch: {}, "
                      "t0: {}, p: {}".format(ch_idx, t0, phase))
            # check for sample/parametric pulse
            else:
                sample_pulse_lib = self.qobj_dict["config"]["pulse_library"]
                parametric_pulse_lib = self.qobj_dict["config"]["parametric_pulses"]
                constant = False
                duration = 0
                inst_type = None
                parameters = samples = None
                # check for sample pulse
                for i in range(len(sample_pulse_lib)):
                    pulse_spec = sample_pulse_lib[i]
                    if pulse_spec["name"] == name:
                        samples = pulse_spec["samples"]
                        duration = len(samples)
                        inst_type = InstructionType.SAMPLE
                    #ENDIF
                #ENDFOR
                # check for parametric pulse
                if name in parametric_pulse_lib:
                    if name == "constant":
                        constant = True
                    #ENDIF
                    duration = qinst["parameters"]["duration"]
                    inst_type = InstructionType.PARAMETRIC
                    parameters = qinst["parameters"]
                #ENDFOR
                # parse the pulse if one was found
                if inst_type is not None:
                    ch_name = qinst.get("ch")
                    ch_idx = self.backend.ch_name_idx[ch_name]
                    addr = stats[ch_idx]["addr"]
                    if addr + duration > self.backend.dac_max_memory:
                        raise Error("maximum memory exceeded on channel {} DAC"
                                    "".format(ch_idx))
                    #ENDIF
                    inst = Instruction(ch_idx, duration, name, inst_type, t0,
                                       addr=addr, constant=constant,
                                       parameters=parameters, samples=samples)
                    #ENDIF
                    insts[ch_idx].append(inst)
                    stats[ch_idx]["t"] = t0 + duration
                    stats[ch_idx]["addr"] = stats[ch_idx]["addr"] + duration
                    self.backend.logger.log(logging.DEBUG, "parsed {} ch: {}, "
                          "t0: {}, d: {}, n: {}".format(inst_type, ch_idx, t0, duration, name))
                    #ENDIF
                else:
                    raise(Exception("Unrecognized instruction: {}".format(qinst["name"])))        
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
        r_i = self.backend.misc_register[0]
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
                    # TODO: the np.isreal check will be deprecated in future firmware versions
                    # currently, only the DDS can have complex values, not the table
                    constant = inst.constant and np.isreal(inst.parameters["amp"])
                    if constant:
                        gain_val = int(inst.parameters["amp"] * self.backend.tproc_max_gain)
                        p.regwi(p_ch, r_gain, gain_val, "ch {} gain".format(ch_idx))
                        mode_code = self._mode_code(
                            STDYSEL_ZERO, MODE_ONESHOT,
                            OUTSEL_T, inst.duration // self.backend.tproc_to_dac
                        )
                    else:
                        mode_code = self._mode_code(
                            STDYSEL_ZERO, MODE_ONESHOT,
                            OUTSEL_TDDS, inst.duration // self.backend.tproc_to_dac
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
                    if constant:
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
        p.synci(p.us2cycles(self.qobj_dict["config"]["rep_delay"]), "rep delay")
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

    def run_set(self, set_idx):
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
                    samples = inst.samples(self.backend)
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
                    duration_adc_dec = duration_dac // (ADC_TO_DAC * DECIMATION)
                    avg_buf.config(address=inst.addr, length=duration_adc_dec)
                    avg_buf.enable()
                #ENDIF
            #ENDFOR
        #ENDFOR
        # load ASM program
        self.backend.soc.tproc.load_asm_program(self.asm_program)
        # write number of shots into memory
        shots_completed = set_idx * self.shots_per_set
        if shots_completed + self.shots_per_set > self.shots:
            shots = self.shots - shots_completed
        else:
            shots = self.shots_per_set
        #ENDIF
        self.backend.soc.tproc.single_write(addr=0, data=shots - 1)
        # run ASM program
        self.backend.soc.tproc.stop()
        self.backend.soc.tproc.start()
        # get results for each acquire statement
        results = dict()
        for i, ch_idx in enumerate(self.insts.keys()):
            for inst in self.insts[ch_idx]:
                if inst.inst_type == InstructionType.ACQUIRE:
                    # get raw data for each shot
                    ch_name = self.backend.ch_idx_name[ch_idx]
                    i_data, q_data = self.backend.soc.getAccumulated(
                        address=inst.addr,
                        length=self.qobj_dict["config"]["shots"]
                    )
                    data = i_data + 1j * q_data
                    # average over shots if necessary
                    if self.qobj_dict["config"]["meas_return"] == "avg":
                        data = np.sum(data) / data.shape[0]
                    #ENDIF
                    # create result list if one does not exist
                    if not ch_name in results:
                        results[ch_name] = list()
                    #ENDIF
                    # append result
                    results[ch_name].append(data)
                #ENDIF
            #ENDIF
        #ENDIF
        return results
    #ENDDEF
#ENDCLASS
