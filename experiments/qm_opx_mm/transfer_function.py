import numpy as np
from h5py import File

def transfer_function(oct_file=oct_filename, cal_file = cal_filename):
    with File(oct_filename,'r') as a:
        Iq = np.array(a['uks'][-1][0], dtype=float)
        Qq = np.array(a['uks'][-1][1], dtype=float)
        Ic = np.array(a['uks'][-1][2], dtype=float)
        Qc = np.array(a['uks'][-1][3], dtype=float)
        a.close()
    with File(cal_filename,'r') as a:
        omegas = a['omgas']
        amps = a['amps']
        a.close()








def transfer_function(self, omegas_in, mode_index=0):
    # takes input array of omegas and converts them to output array of amplitudes,
    # using a calibration h5 file defined in the experiment config
    # pull calibration data from file, handling properly in case of multimode cavity
    if channel == 'cavity':
        fn_file = self.experiment_cfg['amp_vs_freq_transfer_function_calibration_files'][channel][mode_index]
    elif channel == 'cavity_list_weak':
        fn_file = self.experiment_cfg['amp_vs_freq_transfer_function_calibration_files'][channel][mode_index]
    else:
        fn_file = self.experiment_cfg['amp_vs_freq_transfer_function_calibration_files'][channel]
    with File(fn_file, 'r') as f:
        omegas = f['omegas'][()]
        amps = f['amps'][()]
    # assume zero frequency at zero amplitude, used for interpolation function
    omegas = np.append(omegas, -omegas)
    amps = np.append(amps, -amps)
    omegas = np.append(omegas, 0.0)
    amps = np.append(amps, 0.0)
    o_s = [x for y, x in sorted(zip(amps, omegas))]
    a_s = np.sort(amps)

    # interpolate data, transfer_fn is a function that for each omega returns the corresponding amp
    transfer_fn = interpolate.interp1d(o_s, a_s)
    output_amps = []
    max_interp_index = np.argmax(omegas)
    for i in range(len(omegas_in)):
        # if frequency greater than calibrated range, assume a proportional relationship (high amp)
        if np.abs(omegas_in[i]) > omegas[max_interp_index]:
            output_amps.append(omegas_in[i] * amps[max_interp_index] / omegas[max_interp_index])
        else:  # otherwise just use the interpolated transfer function
            output_amps.append(transfer_fn((omegas_in[i])))
    return np.array(output_amps)