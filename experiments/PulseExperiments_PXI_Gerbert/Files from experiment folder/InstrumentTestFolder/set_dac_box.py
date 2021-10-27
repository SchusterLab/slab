from slab import *
from slab.instruments import *
dac = AD5780_serial()
flux_list = [0, 0, 0, 0, 0, 0, 0.0, 0]

#flux_list = [4, 0,  0,  0,  0, 0, 0,  0]
#flux_list = [0.0]*8

dac.parallelramp(flux_list,stepsize = 8,steptime = 1)

