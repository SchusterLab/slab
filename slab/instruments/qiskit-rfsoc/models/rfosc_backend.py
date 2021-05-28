"""
rfsoc_backend.py
"""

from .backend import SLabBackend
from .rfsoc_experiment import RFSoCExperiment

RFSOC_BACKEND_KEYS = [
    "ch_name_idx", "ch_idx_rdds", "ch_idx_page", "ch_idx_reg",
    "misc_page", "misc_reg", "tproc_initial_cycle_offset",
    "adc_trig_offset", "acquire_pad",
]

class RFSoCBackend(SLabBackend):
    def __init__(config_file_path, log_path):
        super().__init__(config_file_path, log_path, RFSoCExperiment):
        # set keys from config
        for (k, v) in self.config_dict.items():
            setattr(self, k, v)
        #ENDFOR
        # TODO do RFSoC initialization, add config key for path
        self.soc = PfbSoC()
        # TODO this shouldn't be hardcoded
        self.gens = {
            1: self.soc.gen0,
            2: self.soc.gen1,
            3: self.soc.gen2,
            4: self.soc.gen3,
        }
        self.avg_bufs = {
            0: self.soc.avg_buf,
        }
        self.readouts = {
            0: self.soc.readout,
        }
        # TODO these should be fetched from self.backend.soc
        self.tproc_to_dac = 16
        self.adc_to_dac = 2
        self.decimation = 8
        self.tproc_max_phase = 2 ** 16 - 1
        self.dac_max_memory = 16 * (2 ** 12)
        self.tproc_max_gain = 2 ** 15 - 1
        # mode constants
        # generator outputs last sample after pulse terminates
        self.stdysel_last = 0
        # generator outputs zeros after pulse terminates
        self.stdysel_zero = 1
        # generator stops playing after pulse terminates
        self.mode_oneshot = 0
        # generator plays pulse on loop
        self.mode_periodic = 1
        # output is product of table and dds
        self.outsel_tdds = 0
        # output is dds
        self.outsel_dds = 1
        # output is table only
        self.outsel_t = 2
        # output is zero
        self.outsel_zero = 3
    #ENDDEF

    def required_config_keys():
        return super().required_config_keys() + RFSOC_BACKEND_KEYS
    #ENDDEF
#ENDCLASS

