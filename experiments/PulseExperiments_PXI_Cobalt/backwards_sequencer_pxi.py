import numpy as np
import visdom

try:
    from .backwards_pulse_classes import Gauss_B, Idle_B, Ones_B, Square_B
except:
    from backwards_pulse_classes import Gauss_B, Idle_B, Ones_B, Square_B

class BackwardsSequencer():

    def __init__(self, channels, channels_awg, awg_info, channels_delay):
        self.channels = channels
        self.channels_awg = channels_awg
        self.awg_info = awg_info
        self.channels_delay = channels_delay


        channels_awg_info = {}

        for channel in channels:
            channels_awg_info[channel] = awg_info[channels_awg[channel]]

        self.channels_awg_info = channels_awg_info

        self.pulse_array_list = {}
        self.multiple_sequences = []

    def new_sequence(self, sequences):
        self.pulse_array_list = {}


        # # M8195a trig
        # try:
        #     self.append('m8195a_trig', Ones(time=sequences.hardware_cfg['trig_pulse_len']['m8195a']))
        # except:
        #     pass
        # Removed sideband cooling, check sequencer_pxi and make mods if you would like to restore it.

        self.sync_channels_time(sequences.channels)

    def append(self, channel, pulse):
        pulse.generate_pulse_array(t0=self.get_time(channel), dt=self.channels_awg_info[channel]['dt'])
        pulse_array = pulse.pulse_array
        self.pulse_array_list[channel].append(pulse_array)

    def get_time(self, channel):
        # Subtracted the time delay because we're... going backwards
        return self.channels_awg_info[channel]['dt'] * len(np.concatenate(self.pulse_array_list[channel])) - self.channels_awg_info[channel]['time_delay']

    # MODIFY THIS TO COUNT THE RIGHT WAY
    def append_idle_to_time(self, channel, time):
        # Know that idle is backwards! It'll be the same length though.
        current_time = self.get_time(channel)
        extra_time = time - current_time - 0.0001*self.channels_awg_info[channel]['dt']
        # the "- delta*self.channels_awg_info[channel]['dt']" is to add numerical stability
        # when expect time - current_time ==  I (an integer) multiple of clock rate, i.e. expect numpy array with lenth I
        # if time - current_time is slightly larger than I * dt,
        # np.arange(0, I * dt + eps, self.dt) will result a numpy array with length I+1

        if extra_time > 0:
            self.append(channel, Idle(time=extra_time, dt=self.channels_awg_info[channel]['dt']))

    # A new one to stick the idles on the front end
    # NEEDS WORK IN ORDER TO FUNCTION PROPERLY
    def pad_sequence_front(self):
        for channel in self.channels:
            idle = Idle(time=100, dt=self.channels_awg_info[channel]['dt'])
            idle.generate_pulse_array()
            # MODIFY THIS TO APPEND RATHER THAN EQUATE ONCE APPEND SYNTAX IS WRITTEN
            self.pulse_array_list[channel] = [idle.pulse_array]