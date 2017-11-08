__author__ = 'Nelson'

from MultimodeFluxSideBandFreqSweepExperiment3 import *
import gc
import json
from slab.instruments.Alazar import Alazar

flux_freq_pts = arange(2.515,3.0,0.001)

path = r"S:\_Data\150716 - 2D multimode (Chip 1)\data"
prefix = 'MM_Flux_Sideband_Rabi_Freq_Sweep_3_'
data_file =  os.path.join(path, get_next_filename(path, prefix, suffix='.h5'))
config_file = os.path.join(path, "..\\config" + ".json")
with open(config_file, 'r') as fid:
    cfg_str = fid.read()

cfg = AttrDict(json.loads(cfg_str))

print "Prep Card"
adc = Alazar(cfg['alazar'])

for ii, flux_freq in enumerate(flux_freq_pts):

    mm_flux_sideband_rabi_freq_sweep=MultimodeFluxSideBandFreqSweepExperiment3(path=path,prefix=prefix,config_file='..\\config.json',flux_freq=flux_freq, data_file=data_file)
    # if ii == 0:
    #     mm_flux_sideband_rabi_freq_sweep.plotter.clear()
    if mm_flux_sideband_rabi_freq_sweep.ready_to_go:
        mm_flux_sideband_rabi_freq_sweep.go(adc=adc)

    mm_flux_sideband_rabi_freq_sweep = None
    del mm_flux_sideband_rabi_freq_sweep

    gc.collect()

