from configuration_IQ import config, storage_cal_file, qubit_cal_file
import numpy as np
from h5py import File
import scipy
import os

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
    return np.array(output_amps)

def oct_to_opx_amp(opx_config=config, fock_state=1):
    """Given the Fock state to prepare, it obtains the oct amplitudes in terms of the opx amp and
    updates the waveforms in the opx config"""
    path = os.getcwd()
    pulse_filename = path+"//oct_pulses//g"+str(fock_state)+".h5"

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
    a_max = 0.45 #Max peak-peak amplitude out of OPX

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
    return alpha