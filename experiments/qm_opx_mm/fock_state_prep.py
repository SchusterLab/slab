from configuration_IQ import config, storage_cal_file, qubit_cal_file, ge_IF, two_chi, two_chi_2, two_chi_vec
import numpy as np
from h5py import File
import scipy
import os
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface

def transfer_function(omegas_in, cavity=False, qubit=True, storage_cal_file=storage_cal_file[1], qubit_cal_file=qubit_cal_file):
    # takes input array of omegas and converts them to output array of amplitudes,
    # using a calibration h5 file defined in the experiment config
    # pull calibration data from file, handling properly in case of multimode cavity

    if cavity==True:
        fn_file = storage_cal_file
    elif qubit==True:
        fn_file = qubit_cal_file

    with File(fn_file, 'r') as f:
        omegas = np.array(f['omegas'])
        amps = np.array(f['amps'])
    # assume zero frequency at zero amplitude, used for interpolation function
    omegas = np.append(omegas, -omegas)
    amps = np.append(amps, -amps)
    omegas = np.append(omegas, 0.0)
    amps = np.append(amps, 0.0)

    o_s = [x for y, x in sorted(zip(amps, omegas))]
    a_s = np.sort(amps)

    # interpolate data, transfer_fn is a function that for each omega returns the corresponding amp
    transfer_fn = scipy.interpolate.interp1d(o_s, a_s)
    output_amps = []
    max_interp_index = np.argmax(omegas)

    for i in range(len(omegas_in)):
        # if frequency greater than calibrated range, assume a proportional relationship (high amp)
        if np.abs(omegas_in[i]) > omegas[max_interp_index]:
            output_amps.append(omegas_in[i] * amps[max_interp_index] / omegas[max_interp_index])
            # output_amps.append(amps[max_interp_index])
        else:  # otherwise just use the interpolated transfer function
            output_amps.append(transfer_fn((omegas_in[i])))
    return np.array(output_amps)*1.025814693410431

def oct_to_opx_amp(opx_config=config, fock_state=1):
    """Given the Fock state to prepare, it obtains the oct amplitudes in terms of the opx amp and
    updates the waveforms in the opx config"""

    path = os.getcwd()

    if fock_state==0:
        pulse_filename = path+"//oct_pulses//g"+str(1)+".h5"
        a_max = 0.0
    else:
        pulse_filename = path+"//oct_pulses//g"+str(fock_state)+".h5"
        a_max = 0.45 #Max peak-peak amplitude out of OPX

    with File(pulse_filename,'r') as a:
        Iq = np.array(a['uks'][-1][0], dtype=float)
        Qq = np.array(a['uks'][-1][1], dtype=float)
        Ic = np.array(a['uks'][-1][2], dtype=float)
        Qc = np.array(a['uks'][-1][3], dtype=float)
        a.close()

    Iq = transfer_function(Iq, qubit=True)
    Qq = transfer_function(Qq, qubit=True)
    Ic = transfer_function(Ic, qubit=False, cavity=True)
    Qc = transfer_function(Qc, qubit=False, cavity=True)

    Iq = [float(x*a_max) for x in Iq]
    Qq = [float(x*a_max) for x in Qq]
    Ic = [float(x*a_max) for x in Ic]
    Qc = [float(-x*a_max) for x in Qc] #We need to multiply this with a -ve sign since we are using the LSB with a +ve IF frequency

    config['pulses']['qoct_pulse']['length'] = len(Iq)
    config['pulses']['soct_pulse']['length'] = len(Ic)

    config['waveforms']['qoct_wf_i']['samples'] = Iq
    config['waveforms']['qoct_wf_q']['samples'] = Qq
    config['waveforms']['soct_wf_i']['samples'] = Ic
    config['waveforms']['soct_wf_q']['samples'] = Qc

    return len(Iq)

def opx_amp_to_alpha(cav_amp=1.0, cav_len=250, storage_cal_file=storage_cal_file[1]):
    # takes input array of amps and length and converts them to output array of alphas,
    # using a calibration h5 file defined in the experiment config
    # pull calibration data from file, handling properly in case of multimode cavity
    with File(storage_cal_file, 'r') as f:
        omegas = np.array(f['omegas'])
        amps = np.array(f['amps'])

    # assume zero frequency at zero amplitude, used for interpolation function
    omegas = np.append(omegas, 0.0)
    amps = np.append(amps, 0.0)

    o_s = omegas
    a_s = amps

    # interpolate data, transfer_fn is a function that for each amp returns the corresponding omega
    transfer_fn = scipy.interpolate.interp1d(a_s, o_s)

    omega_desired = transfer_fn(cav_amp)
    alpha = omega_desired * cav_len

    """Returns alpha in the cavity"""
    return alpha/1.025814693410431

def oct_to_opx_amp_test(opx_config=config, pulse_filename=None, qctrl=False):
    """Given the Fock state to prepare, it obtains the oct amplitudes in terms of the opx amp and
    updates the waveforms in the opx config"""

    path = os.getcwd()

    if 'g0' in pulse_filename:
        a_max = 0.0
    else:
        a_max = 0.45 #Max peak-peak amplitude out of OPX

    if qctrl:
        with File(pulse_filename,'r') as a:
            Iq = np.array(a['Iq'], dtype=float)
            Qq = np.array(a['Qq'], dtype=float)
            Ic = np.array(a['Ic'], dtype=float)
            Qc = np.array(a['Qc'], dtype=float)
            a.close()
    else:
        with File(pulse_filename,'r') as a:
            Iq = np.array(a['uks'][-1][0], dtype=float)
            Qq = np.array(a['uks'][-1][1], dtype=float)
            Ic = np.array(a['uks'][-1][2], dtype=float)
            Qc = np.array(a['uks'][-1][3], dtype=float)
            a.close()

    Iq = transfer_function(Iq, qubit=True)
    Qq = transfer_function(Qq, qubit=True)
    Ic = transfer_function(Ic, qubit=False, cavity=True)
    Qc = transfer_function(Qc, qubit=False, cavity=True)

    Iq = [float(x*a_max) for x in Iq]
    Qq = [float(x*a_max) for x in Qq]
    Ic = [float(x*a_max) for x in Ic]
    Qc = [float(-x*a_max) for x in Qc] #We need to multiply this with a -ve sign since we are using the LSB with a +ve IF frequency

    config['pulses']['qoct_pulse']['length'] = len(Iq)
    config['pulses']['soct_pulse']['length'] = len(Ic)

    config['waveforms']['qoct_wf_i']['samples'] = Iq
    config['waveforms']['qoct_wf_q']['samples'] = Qq
    config['waveforms']['soct_wf_i']['samples'] = Ic
    config['waveforms']['soct_wf_q']['samples'] = Qc

    return len(Iq)

def alpha_pulse_len_cal(alpha, cav_amp=1.0, cal_file=storage_cal_file[1]):
    # takes input array of omegas and converts them to output array of amplitudes,
    # using a calibration h5 file defined in the experiment config
    # pull calibration data from file, handling properly in case of multimode cavity
    with File(cal_file, 'r') as f:
        omegas = np.array(f['omegas'])
        amps = np.array(f['amps'])
    # assume zero frequency at zero amplitude, used for interpolation function
    omegas = np.append(omegas, 0.0)
    amps = np.append(amps, 0.0)

    o_s = omegas
    a_s = amps

    # interpolate data, transfer_fn is a function that for each omega returns the corresponding amp
    transfer_fn = scipy.interpolate.interp1d(a_s, o_s)

    omega_desired = transfer_fn(cav_amp)

    pulse_length = (1.025814693410431*alpha/omega_desired)
    """Returns time in units of 4ns for FPGA"""
    return abs(pulse_length)//4+1

def snap_seq(fock_state=0, phase=0.1):

    opx_amp = 1.0

    if fock_state==0:
        play("CW"*amp(0.0),'storage_mode1', duration=alpha_pulse_len_cal(1.143, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(0.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-0.0),'storage_mode1', duration=alpha_pulse_len_cal(-0.58, cav_amp=opx_amp))

    elif fock_state==1:
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(1.143, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        frame_rotation_2pi(0.05, 'storage_mode1')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(-0.58, cav_amp=opx_amp))

    elif fock_state==2:
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.497, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        # frame_rotation_2pi(0.045, 'storage_mode1')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(1.133, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + two_chi + (1*(1-1))*two_chi_2)
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        # frame_rotation_2pi(0.025, 'storage_mode1')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.432, cav_amp=opx_amp))
        # update_frequency('qubit_mode0', ge_IF[0])

    elif fock_state==3:
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.531, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        # frame_rotation_2pi(0.025, 'storage_mode1')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.559, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + two_chi + (1*(1-1))*two_chi_2)
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        # frame_rotation_2pi(0.025, 'storage_mode1')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.946, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + 2*two_chi + (2*(2-1))*two_chi_2)
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        # frame_rotation_2pi(0.005, 'storage_mode1')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.358, cav_amp=opx_amp))
        # update_frequency('qubit_mode0', ge_IF[0])

    elif fock_state==4:
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.843, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.637, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + two_chi + (1*(1-1))*two_chi_2)
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.443, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + 2*two_chi + (2*(2-1))*two_chi_2)
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.827, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        update_frequency('qubit_mode0', ge_IF[0] + 3*two_chi + (3*(3-1))*two_chi_2)
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        frame_rotation_2pi(0.025, 'storage_mode1')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.311, cav_amp=opx_amp))
        # update_frequency('qubit_mode0', ge_IF[0])

    elif fock_state==5:
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.567, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.921, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + two_chi + (1*(1-1))*two_chi_2)
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.442, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + 2*two_chi + (2*(2-1))*two_chi_2)
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.385, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        update_frequency('qubit_mode0', ge_IF[0] + 3*two_chi + (3*(3-1))*two_chi_2)
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.735, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        update_frequency('qubit_mode0', ge_IF[0] + 4*two_chi + (4*(4-1))*two_chi_2)
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.278, cav_amp=opx_amp))
        # update_frequency('qubit_mode0', ge_IF[0])

    elif fock_state==6:
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.5485, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.8411, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + two_chi+ (1*(1-1))*two_chi_2)
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.7779, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + 2*two_chi + (2*(2-1))*two_chi_2)
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.088, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        update_frequency('qubit_mode0', ge_IF[0] + 3*two_chi+ (3*(3-1))*two_chi_2)
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.5938, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        update_frequency('qubit_mode0', ge_IF[0] + 4*two_chi + (4*(4-1))*two_chi_2)
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.6313, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        update_frequency('qubit_mode0', ge_IF[0] + 5*two_chi + (5*(5-1))*two_chi_2)
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.2257, cav_amp=opx_amp))
        # update_frequency('qubit_mode0', ge_IF[0])

def snap_seq_test(fock_state=0, phase=0.1):

    opx_amp = 1.0

    if fock_state==0:
        play("CW"*amp(0.0),'storage_mode1', duration=alpha_pulse_len_cal(1.143, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(0.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-0.0),'storage_mode1', duration=alpha_pulse_len_cal(-0.58, cav_amp=opx_amp))

    elif fock_state==1:
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(1.143, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        frame_rotation_2pi(0.05, 'storage_mode1')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(-0.58, cav_amp=opx_amp))

    elif fock_state==2:
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.497, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        # frame_rotation_2pi(0.045, 'storage_mode1')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(1.133, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + two_chi_vec[fock_state-1])
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        # frame_rotation_2pi(0.025, 'storage_mode1')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.432, cav_amp=opx_amp))
        # update_frequency('qubit_mode0', ge_IF[0])

    elif fock_state==3:
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.531, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        # frame_rotation_2pi(0.025, 'storage_mode1')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.559, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + two_chi_vec[fock_state-2])
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        # frame_rotation_2pi(0.025, 'storage_mode1')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.946, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + two_chi_vec[fock_state-1])
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        # frame_rotation_2pi(0.005, 'storage_mode1')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.358, cav_amp=opx_amp))
        # update_frequency('qubit_mode0', ge_IF[0])

    elif fock_state==4:
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.843, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.637, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + two_chi_vec[fock_state-3])
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.443, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + two_chi_vec[fock_state-2])
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.827, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        update_frequency('qubit_mode0', ge_IF[0] + two_chi_vec[fock_state-1])
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        frame_rotation_2pi(0.025, 'storage_mode1')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.311, cav_amp=opx_amp))
        # update_frequency('qubit_mode0', ge_IF[0])

    elif fock_state==5:
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.567, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.921, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + two_chi_vec[fock_state-4])
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.442, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + two_chi_vec[fock_state-3])
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.385, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        update_frequency('qubit_mode0', ge_IF[0] + two_chi_vec[fock_state-2])
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.735, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        update_frequency('qubit_mode0', ge_IF[0] + two_chi_vec[fock_state-1])
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.278, cav_amp=opx_amp))
        # update_frequency('qubit_mode0', ge_IF[0])

    elif fock_state==6:
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.5485, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.8411, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + two_chi_vec[fock_state-5])
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.7779, cav_amp=opx_amp))
        update_frequency('qubit_mode0', ge_IF[0] + two_chi_vec[fock_state-4])
        align('storage_mode1', 'qubit_mode0')
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.088, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        update_frequency('qubit_mode0', ge_IF[0] + two_chi_vec[fock_state-3])
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.5938, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        update_frequency('qubit_mode0', ge_IF[0] + two_chi_vec[fock_state-2])
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(-opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.6313, cav_amp=opx_amp))
        align('storage_mode1', 'qubit_mode0')
        update_frequency('qubit_mode0', ge_IF[0] + two_chi_vec[fock_state-1])
        play("res_2pi"*amp(1.0), 'qubit_mode0')
        align('storage_mode1', 'qubit_mode0')
        play("CW"*amp(opx_amp),'storage_mode1', duration=alpha_pulse_len_cal(0.2257, cav_amp=opx_amp))
        # update_frequency('qubit_mode0', ge_IF[0])
