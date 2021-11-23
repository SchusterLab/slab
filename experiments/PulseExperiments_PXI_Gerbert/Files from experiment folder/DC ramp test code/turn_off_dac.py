from slab import *
from DACInterface import AD5780_serial
dac = AD5780_serial()
time.sleep(2)
dac.init()
time.sleep(2)
dac.init()
time.sleep(2)