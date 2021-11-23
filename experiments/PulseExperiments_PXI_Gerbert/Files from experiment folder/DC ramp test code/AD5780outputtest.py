from slab.instruments import InstrumentManager
#from alex_run_seq_experiment import *
from slab.instruments.AD5780DAC.AD5780 import AD5780
from slab.instruments.voltsource import *
from DACInterface import *

from numpy import *
import time


freq_dc = [4.75e9]*8

#=======================================

def set_flux_direct(flux_actual= None):

    im = InstrumentManager()

    # dac = AD5780(address='192.168.14.158')
    # print(dac.get_id().strip())

    dac = AD5780_serial()
    print("DAC connected vai USB: ", dac.ard.is_open)
    time.sleep(1)


    if True:
        # ramping codes
        pt = flux_actual
        print("Driving ADC at ( %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f) V" % (
            pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7]))

        for jj in range(0, 8):
            if jj == 0:
                dac.ramp3(dacnum=8, voltage=pt[jj], step=5, speed=5)
                time.sleep(0.5)
            # pass f4
            elif jj == 4:
                pass
            else:
                dac.ramp3(dacnum=jj, voltage=pt[jj], step=5, speed=5)
                time.sleep(0.5)

        print("Driving YOKO at (%.3f) mA" % (pt[4] * 1000))
        dcflux2.ramp_current(pt[4], 0.0016)

        time.sleep(2)

    return flux_actual

def set_flux_inverted(flux_target= None):

    ###
    if flux_target == None:
        # 0905
        flux_target = [0]*8

    print('flux target (additional offset) =', flux_target)
    # flux_actual = dot(Minv, flux_target)

    diag_start = array([1.3, 0, -0.87506893, -1.76006077, 0.00193846, 1.01371724,
                        -0.6663699, 0.14317566])

    diag_offset = [0, -0.5, 0.65, 1.4, 0.108, 0, 0.3, 0.05]

    pt_target = list(array(flux_target) + array(diag_offset))
    targetNoF0 = pt_target[1:8]
    ptNoF0 = list(dot(Minv, targetNoF0))

    # pt = [0.0] + ptNoF0
    # 1101
    pt = [0.0, 0.0] + ptNoF0[1:]
    print('flux target (additional offset) inverted =', pt)

    flux_actual = array(pt) + array(diag_start)
    print('flux_actual =', list(flux_actual))

    if True:
        set_flux_direct(flux_actual = flux_actual)

    return flux_actual

def set_q_freq_inverted(freq_jump=[0]*8):


    # move freq_cal in opposite direction!
    # freq_cal = [0, 4.809e9, 4.761e9, 4.755e9,
    #             4.7445e9, 4.791e9, 4.803e9, 4.8e9]

    # 7 site chain
    freq_cal = [0, 4.800e9, 4.752e9, 4.745e9,
                4.814e9, 4.741e9, 4.773e9, 4.77e9]

    freq_cal = [0, 4.800e9, 4.752e9-100e6, 4.745e9-0e6,
                4.814e9, 4.741e9-100e6, 4.773e9-100e6, 4.77e9-100e6]

    # # 1026 try to move Q4 away
    # freq_cal = [0, 4.800e9+0e6, 4.752e9+200e6, 4.745e9+200e6,
    #             4.814e9+400e6, 4.741e9, 4.773e9, 4.77e9]
    #
    # # 1031 move Q4/Q5 away, push Q6/7 up for readout
    # freq_cal = [0, 4.800e9+0e6, 4.752e9+200e6, 4.745e9+200e6,
    #             4.814e9+400e6, 4.741e9+300e6, 4.773e9-280e6, 4.77e9-0e6]

    # # 7 site chain & forced F1 to be zero
    # freq_cal = [0, 4.800e9, 4.752e9-50e6, 4.745e9,
    #             4.814e9-70e6, 4.741e9, 4.773e9-200e6, 4.77e9]

    flux_cal = [0]*8
    freq_target = [0] + [4.741e9]*7

    for ii in range(8):
        # skip Q4 in case inversion blows up, change it in cal
        if ii not in [4]:
            freq_target[ii] += freq_jump[ii]
    freq_target[0] = 0
    print('freq target with jump =', array(freq_target)/1e9, 'GHz')

    flux_target = flux_cal[:]
    for ii in range(1,8):
        flux_target[ii] += (freq_target[ii]-freq_cal[ii])/1e9/flux_slope[ii]
    # not using F0 in inversion
    flux_target[0] = 0
    print('flux target (offset) with jump =', flux_target)

    if True:

        flux_actual = set_flux_inverted(flux_target = flux_target)
        return flux_actual

    else:
        return -1

def get_fast_flux_inverted(freq_jump=[0]*8):

    freq_target = [0]*8 # + [4.741e9]*7

    for ii in range(8):
        freq_target[ii] += freq_jump[ii]
    freq_target[0] = 0
    print('fast_flux: freq target with jump =', array(freq_target)/1e9, 'GHz')

    fast_flux_target = zeros(8)
    for ii in range(1,8):
        fast_flux_target[ii] += (freq_target[ii]-freq_dc[ii])/fast_flux_slope[ii]

    print('fast_flux target (no inversion) = ', fast_flux_target)

    fast_flux_inverted = [0.0] + list(dot(Minv_fastflux, fast_flux_target[1:8]))
    print('fast_flux target (with inversion) =', fast_flux_inverted)

    if  max(abs(array(fast_flux_inverted))) > 0.99:
        raise Exception('error: max fast flux > 0.99, setting fast flux to [0] !')
        #return [0]*8
    else:
        return fast_flux_inverted

def get_fast_flux_inverted_relative(freq_jump=[0]*8):

    freq_target = freq_jump
    freq_target[0] = 0
    print('fast_flux: freq jump =', array(freq_target)/1e6, 'MHz')

    fast_flux_target = zeros(8)
    for ii in range(1,8):
        fast_flux_target[ii] += (freq_target[ii])/fast_flux_slope[ii]

    print('fast_flux target (no inversion) = ', fast_flux_target)

    fast_flux_inverted = [0.0] + list(dot(Minv_fastflux, fast_flux_target[1:8]))
    print('fast_flux target (with inversion) =', fast_flux_inverted)

    if  max(abs(array(fast_flux_inverted))) > 0.99:
        raise Exception('error: max fast flux > 0.99, setting fast flux to [0] !')
        #return [0]*8
    else:
        return fast_flux_inverted

def get_rel_qfreq_from_fast_flux(fast_flux=[0]*8):

    return fast_flux_slope * concatenate(([0],dot(M_fastflux, array(fast_flux[1:]))))

if __name__ == "__main__":


    # set_q_freq_inverted(freq_jump=[0,0,0,0, 0,0,0,0 ])

    # tt = [1.5, 0, 2.18, 2.88, (0.43), 3.5, 2.0, 2.6]
    # tt[3] = 2.8
    # set_flux_inverted(flux_target= tt)

    # # # q2 at 4745129098.31
    # set_flux_direct([1.3, 0.0, -0.00264499412544561, -0.15592545007497804,
    #                  0.0039310989355102008, 3.223778670522344+0.5, 0.89653885130291411+0.9, 1.3521529428407582+0.9])
    #
    # # 11-09 try readout on Q0
    # set_flux_direct([1.3, 0.0, -0.00264499412544561, -0.15592545007497804,
    #                  0.0039310989355102008, 3.223778670522344 + 0.5, 0.89653885130291411 + 0.9,
    #                  1.3521529428407582 + 0.9])

    # set_flux_direct([0]*8)

    # 02 23 - Q67 at 4.75 ish
    # set_flux_direct([-1.0, 0, 0.0, 0.0,   0.0e-3, -0.5, 2.6, 3.2])

    # set_flux_direct([-1.0, 0, 0.0, 0.0,   0.0e-3, -0.5, 2.2, 2.7])

    # Q6 5.004, Q7
    # set_flux_direct([-1.0, 0, 0.0, 0.0,   0.0e-3, -0.5, 3.5, 3.9])

    # 8 site 0227
    # set_flux_direct([0.38, -0.0, -1.0, -1.0,   19e-3, 2.5, 1.0, 1.5])

    # 0319
    # Q7 read
    # set_flux_direct([ -0.5, 0, 0, 0.0,  0e-3, -1, -2, 4.07])
    # Q6 read
    # set_flux_direct([-0.5, 0., 0., 0., 0.e-3, -1., 2.567, -0.5]) # @ 4.94
    # set_flux_direct([-0.5, 0., 0., 0., 0.e-3, -1., 2.867, -0.5]) # @ 5.014
    # set_flux_direct([-0.5, 0., 0., 0., 0.e-3, -1., 2.717, -0.5]) # @ 4.977

    # Q6 read high clean pulse
    # set_flux_direct([-0.8, 0., 0., 0., 0.e-3, -1., 4.037, -0.5]) # @ 5.155 # final good readout
    # Q6 read high clean pulse - with Q7 nearby
    # set_flux_direct([-0.8, 0., 0., 0., 0.e-3, -1., 3.082, 3.5]) # @ 1 site 2Q Q7 detuned by 100 fro Q6


    # FINAL DATA LOCS ===========================
    # ===========================================

    # till 0510 - final 2q scheme location
    # set_flux_direct([-0.8, 0., 0., 0., 0.e-3, -1., 2.1, 2.5]) # @ 0425 - 1 site 2Q Q7 detuned by 100 fro Q6

    # 0510 1Q scheme
    # set_flux_direct([-0.8, 0., 0., 0., 0.e-3, -1., -2.5, 3.195]) # Q7 at 4.738

    # 0512 final 8 site from pnax
    # set_flux_direct([0.6 ,0., -1.5, -1.5 ,    0.0155, 1., 0., 0.5])
    # set_flux_direct([0.7 , -0.15, -1.5, -1.5 ,    0.0155, 1., 0., 0.5])

    #
    # set_flux_direct([0.75 , -0.15, -1.5, -1.5 ,    0.015, 1., -0.3, 0.5])

    # 05 16 chain
    # set_flux_direct([0.82 , -0.2, -1.5, -1.5 ,    0.014, 1., -0.3, 2.5])

    set_flux_direct([0]*8)

    # ===========================================
    # FINAL DATA LOCS ===========================



    # set_flux_direct([-0.8, 0., 0., 0., 0.e-3, 3.0, -2.57, 4.0])  # @ Q6 at LSS


    # Q7 at lattice location
    # set_flux_direct([-0.8, 0., 0., 0., 0.e-3, -1., -2.5, 3.16]) # Q7 at 4.7315
    # set_flux_direct([-0.8, 0., 0., 0., 0.e-3, -1., -2.5, 3.23]) # Q7 at 4.747
    # set_flux_direct([-0.8, 0., 0., 0., 0.e-3, -1., -2.5, 3.09]) # Q7 at 4.713
    # set_flux_direct([-0.8, 0., 0., 0., 0.e-3, -1., -2.5, 2.7]) # Q7 at 4.619
    # set_flux_direct([-0.8, 0., 0., 0., 0.e-3, -1., -2.5, 3.85]) # Q7 at 4.898

    # Q67 debug zeno
    # set_flux_direct([-0.5, 0., 0., 0., 0.e-3, -1., 2.0-0.04, 2.0])


    # ================================
    # 180125 - 1 qubit scheme read Q7
    # set_flux_direct([-4, 0, -1, -1,     0, 4, 4, -2.0])

    # 180128 two qubit scheme read Q6
    # set_flux_direct([-4, 0, -1, -1,     0, 5, -1.5, 1.0])

    # 180201 tune up lattice

    # set_flux_direct([0.8, -0.4, 0, 0, 0.0225, 3.5, 1.5, 2.5])

    # move to get larger slope
    # set_flux_direct([0.8, -0.4, 0, 0, 0.021, 3.5, 1.5, 2.5])


    # # other place where Q4 is near 4, all others near 4.75
    # tt = [ 1.126458, 0,  -0.035644 , 0.025276 , 0.001451 , 2.431067 , 0.874454 , 1.42643 ]
    # set_flux_direct(tt)

    # set_flux_inverted()
    # qIdx = 7
    # jump = [0e6] * 8
    # for ii in range(8):
    #     if ii is not qIdx:
    #         jump[ii] -= 50e6
    #
    # set_q_freq_inverted(jump)

    # set_q_freq_inverted()

    # get_fast_flux_inverted()

    # ff = [0]*8
    # fast_flux_detune = array([1]*8)
    # print get_qfreq_from_fast_flux(fast_flux_detune)