"""
rfsoc_backend.py
"""

from ..backend import SLabBackend
from .rfsoc_experiment import RFSoCExperiment
from .qsystem0 import PfbSoc

RFSOC_BACKEND_KEYS_REQUIRED = [
    "ch_name_idx", "ch_idx_rdds", "ch_idx_page", "ch_idx_reg",
    "misc_page", "misc_reg", "tproc_initial_cycle_offset",
    "adc_trig_offset", "acquire_pad", "rfsoc_binary_file_path"
]

class RFSoCBackend(SLabBackend):
    # !TODO document args
    def __init__(self, config_file_path, experiment_class, default_shots_per_set,
                 log_level, log_path, result_timeout, result_timeout_sleep,
                 ch_name_idx, ch_idx_rdds, ch_idx_page, ch_idx_reg, misc_page, misc_reg,
                 tproc_initial_cycle_offset, adc_trig_offset, acquire_pad,
                 rfsoc_binary_file_path, **kwargs):
        """
        See SLabBackend class for all args and kwargs not listed here.

        args:
        ch_name_idx :: dict - a dictionary which 
        ch_idx_rdds
        ch_idx_page
        ch_idx_reg
        misc_page
        misc_reg
        tproc_initial_cycle_offset
        adc_trig_offset
        acquire_pad
        """
        # initialize fields
        super().__init__(config_file_path, experiment_class, default_shots_per_set,
                         log_level, log_path, result_timeout, result_timeout_sleep, **kwargs)
        self.ch_name_idx = ch_name_idx
        ch_idx_name = dict()
        for (k, v) in self.ch_name_idx.items():
            ch_idx_name[v] = k
        #ENDFOR
        self.ch_idx_name = ch_idx_name
        self.ch_idx_rdds = ch_idx_rdds
        self.ch_idx_page = ch_idx_page
        self.ch_idx_reg = ch_idx_reg
        self.misc_page = misc_page
        self.misc_reg = misc_reg
        self.tproc_initial_cycle_offset = tproc_initial_cycle_offset
        self.adc_trig_offset = adc_trig_offset
        self.acquire_pad = acquire_pad
        # RFSoC initialization
        self.soc = PfbSoc(rfsoc_binary_file_path)
        # TODO the rest of these fields should be fetched from self.soc
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

    @classmethod
    def from_file(cls, config_file_path):
        return super(RFSoCBackend, cls).from_file(config_file_path, RFSoCExperiment)
    #ENDDEF

    @classmethod
    def required_config_keys(cls):
        return super(RFSoCBackend, cls).required_config_keys() + RFSOC_BACKEND_KEYS_REQUIRED
    #ENDDEF
#ENDCLASS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config-file-path", type=str)
    parser.add_argument("--log-path", default=".", type=str)
    args = vars(parser.parse_args())
    backend = RFSoCBackend(args["config-file-path"], args["log_path"])
    backend.serve()
#ENDDEF

if __name__ == "__main__":
    main()
#ENDIF
