__author__ = 'Nelson'

from .spinapi import *

# Enable the log file
pb_set_debug(1)

print(("Using SpinAPI Library version %s" % pb_get_version()))
print(("Found %d boards in the system.\n" % pb_count_boards()))

pb_select_board(0)

if pb_init() != 0:
	print(("Error initializing board: %s" % pb_get_error()))
	exit()

# time settings
unit_inst_time = 10 # ns
exp_period = 200*1000 # 100us

awg_trig_time = 0
awg_trig_len = 100

card_trig_time = 2000
card_trig_len = 100

readout_trig_time = card_trig_time
readout_trig_len = 1000 # has to be longer than card_trig_len


# calculated time value
awg_trig_delay_loop = awg_trig_len/unit_inst_time
card_and_readout_trig_delay_loop = card_trig_len/unit_inst_time
readout_trig_delay_loop = (readout_trig_len-card_trig_len)/unit_inst_time

awg_card_idle_time = card_trig_time - (awg_trig_time + awg_trig_len)
awg_card_idle_delay_loop = awg_card_idle_time/unit_inst_time

total_time = readout_trig_time + readout_trig_len
period_idle_time = exp_period - total_time

# delay loop
delay_loop = 20

# ports hex value
idle = 0x0
awg_trig = 0x1
card_trig = 0x2
readout_trig = 0x4
drive_trig = 0x8

# Configure the core clock
pb_core_clock(500.0)



# Program the pulse program
pb_start_programming(PULSE_PROGRAM)

# awg trigger
start = pb_inst_pbonly64(awg_trig, Inst.LONG_DELAY,awg_trig_delay_loop,unit_inst_time)

# idle between end of awg trig and card trig
pb_inst_pbonly64(idle, Inst.LONG_DELAY, awg_card_idle_delay_loop, unit_inst_time)

# card and readout trig starts at the same time
pb_inst_pbonly64(card_trig+readout_trig, Inst.LONG_DELAY,card_and_readout_trig_delay_loop,unit_inst_time)

# readout trig after card trig (assume readout trig is longer)
pb_inst_pbonly64(readout_trig, Inst.LONG_DELAY,readout_trig_delay_loop,unit_inst_time)

# branch to start; Outputs are off
pb_inst_pbonly64(idle, Inst.BRANCH, start, period_idle_time)

pb_stop_programming()

# Trigger the board
pb_reset()
pb_start()

pb_close()