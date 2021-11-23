from Keysight_card_control import *
from slab.instruments import *
from DACInterface import AD5780_serial

#set_card_off(mod_nb, ch_nb)
#set_card_ch_DC_V(mod_nb, ch_nb, amp, dc_offset)

set_card_ch_DC_V(mod_nb=6, ch_nb=1, amp=0.0, dc_offset=0)


set_card_off(mod_nb=6, ch_nb=1)
dac = AD5780_serial()
time.sleep(2)
dac.init()
time.sleep(2)
dac.init()
time.sleep(2)
dac.ramp3(5,-1.6,2,2)