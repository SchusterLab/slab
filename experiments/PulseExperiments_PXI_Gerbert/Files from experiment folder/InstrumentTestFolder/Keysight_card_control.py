from slab.instruments.keysight import KeysightLib as key
from slab.instruments.keysight import keysightSD1 as SD1
import numpy as np
import time


def set_card_ch_DC_V(mod_nb, ch_nb, amp, dc_offset):
    """
    Turns card off ie sets DC voltage to zero, and then sets it to Vpp/2
    :param mod_nb:
    :param ch_nb:
    :param amp:
    :param dc_offset:
    :return:
    """
    #since input takes Vpp
    Vpp = amp*2

    # initiate chassis, module, and channel
    chassis = key.KeysightChassis(1, {mod_nb: key.ModuleType.OUTPUT})
    flux_module = chassis.getModule(mod_nb)
    flux_ch = chassis.getChannel(mod_nb, ch_nb)

    # stop anything currently happening
    try:
        err = flux_module.channelWaveShape(ch_nb, SD1.SD_Waveshapes.AOU_OFF)
        if err < 0:
            raise key.KeysightError("Cannot set to off  mode", err)
    except:pass

    #initialize trigger direction of module (not sure if I actually need this?)
    flux_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN) #sets module trigger so trigger by ext

    #set channel to output DC
    err = flux_module.channelWaveShape(ch_nb, SD1.SD_Waveshapes.AOU_DC)
    if err < 0:
        raise key.KeysightError("Cannot set to DC mode", err)

    #set amplitude of DC voltage
    flux_ch.setAmplitude(Vpp)

    #set DC offset of DC voltage
    err1 = flux_module.channelOffset(ch_nb, dc_offset)
    if err1 < 0:
        raise key.KeysightError("Error setting offset voltage", err1)

    flux_module.close()
    chassis.close()

def set_card_ch_CW(mod_nb, ch_nb, amp, freq):
    """
    Turns card off ie sets DC voltage to zero, and then sets it to freq
    :param mod_nb:
    :param ch_nb:
    :param amp:
    :param freq:
    :return:
    """
    #since input takes Vpp
    Vpp = amp*2

    # initiate chassis, module, and channel
    chassis = key.KeysightChassis(1, {mod_nb: key.ModuleType.OUTPUT})
    flux_module = chassis.getModule(mod_nb)
    flux_ch = chassis.getChannel(mod_nb, ch_nb)

    # stop anything currently happening
    try:
        err = flux_module.channelWaveShape(ch_nb, SD1.SD_Waveshapes.AOU_OFF)
        if err < 0:
            raise key.KeysightError("Cannot set to off  mode", err)
    except:pass

    #initialize trigger direction of module (not sure if I actually need this?)
    flux_module.triggerIOconfig(SD1.SD_TriggerDirections.AOU_TRG_IN) #sets module trigger so trigger by ext

    #set channel to output DC
    err = flux_module.channelWaveShape(ch_nb, SD1.SD_Waveshapes.AOU_SINUSOIDAL)
    if err < 0:
        raise key.KeysightError("Cannot set to sinusoidal mode", err)

    #set amplitude
    flux_ch.setAmplitude(Vpp)
    # set frequency
    flux_ch.setFreq(freq)
    flux_module.close()
    chassis.close()

def set_card_off(mod_nb, ch_nb):
    # get chassis, module
    chassis = key.KeysightChassis(1, {mod_nb: key.ModuleType.OUTPUT})
    flux_module = chassis.getModule(mod_nb)
    err = flux_module.channelWaveShape(ch_nb, SD1.SD_Waveshapes.AOU_OFF)
    if err < 0:
        raise key.KeysightError("Cannot set to off  mode", err)
    flux_module.close()
    chassis.close()

def ramp_card_ch_DC_V(mod_nb, ch_nb, target_V, ramp_step, step_speed):
    """
    Sets card ch to DC=0, and then ramps
    :param mod_nb:
    :param ch_nb:
    :param ramp_step:
    :param step_speed: in seconds
    :param target_V:
    :return:
    """
    if np.abs(target_V)>1.5:
        print("Error: target V out of range, greater than 1.5. Terminating.")
        return
    # get chassis, module, and channel
    chassis = key.KeysightChassis(1, {mod_nb: key.ModuleType.OUTPUT})
    flux_module = chassis.getModule(mod_nb)
    flux_ch = chassis.getChannel(mod_nb, ch_nb)

    # stop anything currently happening
    try:
        err = flux_module.channelWaveShape(ch_nb, SD1.SD_Waveshapes.AOU_OFF)
        if err < 0:
            raise key.KeysightError("Cannot set to off  mode", err)
    except:
        pass

    #set channel to output DC
    err = flux_module.channelWaveShape(ch_nb, SD1.SD_Waveshapes.AOU_DC)
    if err < 0:
        raise key.KeysightError("Cannot set to DC mode", err)

    # set amplitude of DC voltage to zero
    flux_ch.setAmplitude(0)

    #ramp amplitude of DC voltage
    nb_steps = int(np.abs(np.ceil(target_V/ramp_step)))
    if nb_steps<2:
        nb_steps=2

    v_steps = np.linspace(0, target_V, nb_steps)
    for i in v_steps:
        flux_ch.setAmplitude(i*2) #times 2 since we want to set dc voltage, and card sets Vpp
        time.sleep(step_speed)
        print("set amplitude of channel to")
        print(i)
    flux_module.close()
    chassis.close()

#set_card_ch_DC_V(4, 2, 1.000, 0)
set_card_ch_DC_V(4, 4, 1.000, 0)


