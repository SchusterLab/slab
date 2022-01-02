import numpy as np
import pandas as pd
from h5py import File
######################
# AUXILIARY FUNCTIONS:
######################
def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    gauss_wave = gauss_wave - 0.5*(gauss_wave[0]+gauss_wave[-1])
    return [float(x) for x in gauss_wave]

def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1-g**2)*(2*c**2-1))
    return [float(N * x) for x in [(1-g)*c, (1+g)*s, (1-g)*s, (1+g)*c]]
################
# CONFIGURATION:
################
####---------------------####
qubit_LO = 4.8681*1e9
qubit_freq = [4.961396736931055*1e9, 4.961396736931055*1e9, 4.961396736931055*1e9, 4.961396736931055*1e9]
ge_IF = [int(qubit_freq[i] - qubit_LO) for i in range(4)]

gauss_len = 40
gauss_amp = 0.45

pi_len = 40
pi_amp = 0.6340

half_pi_len = pi_len
half_pi_amp = 0.3107

pi_len_resolved = 3000
pi_amp_resolved = 0.0081

pi_amps =[0.0, 0.6340, 0.0, 0.0]
pi_2_amps = [0.0, 0.3107, 0.0, 0.0]
res_pi_amps = [0.0081, 0.0081, 0.0, 0.0]
res_pi_len = [100, 3000, 100, 100]

two_chi = [-2.191*1e6,  -1.279*1e6, 0, 0]

ef_IF = -140e6
pi_ef_len = 40
pi_ef_amp = 0.6479
pi_ef_len = pi_ef_len
pi_2_ef_amp = pi_ef_amp/2

qubit_params = {'freqs': [4.961396736931055*1e9, 4.961396736931055*1e9, 4.961396736931055*1e9, 4.961396736931055*1e9],
                'ge_IF': [int(qubit_freq[i] - qubit_LO) for i in range(4)],
                'qubit_LO': 4.8681*1e9,
                'pi_amps': [0.0, 0.6340, 0.0, 0.0],
                'pi_2_amps': [0.0, 0.3107, 0.0, 0.0] ,
                'res_pi_amps': [0.0081, 0.0081, 0.0, 0.0],
                'pi_len': 40, #ns
                'pi_2_len': 40,
                'res_pi_len' : 3000,
                'chi_storage': [-2.191*1e6,  -1.279*1e6, 0, 0],
                'ef_IF': -140e6,
                'pi_ef_amp': 0.6479,
                'pi_ef_len': 40,
                'pi_2_ef_amp': 0.6479/2,
                'pi_2_ef_len': 40,
                }

####---------------------####
rr_LO = 7.8897 *1e9 + 10e6

rr_freq = 7.789406035466071*1e9
rr_IF = int(rr_LO - rr_freq)
opt_readout = "C:\\_Lib\\python\\slab\\experiments\\qm_opx_mm\\pulses\\00013_readout_optimal_pulse.h5"
with File(opt_readout,'r') as a:
    opt_amp = 1.0*np.array(a['I_wf'])

readout_len = 2000
opt_len = len(opt_amp)
readout_params = {'rr_freq': 7.789406035466071*1e9,
                  'rr_LO': 7.8897 *1e9 + 10e6,
                  'rr_IF': int(7.8897 *1e9 + 10e6 - 7.789406035466071*1e9),
                  'sq_amp': 0.4,
                  'readout_length': 2000,
                  'optimal_amp': opt_amp,
                  'optimal_len': opt_len,
                  'disc_file': 'ge_disc_params_sq.npz',
                  'disc_file_opt': 'ge_disc_params_opt.npz',
                  'biased_th_sq':0.005,
                  'biased_th_opt':0.001,
                  'pump_amp':0.0
                  }
####---------------------####
usable_modes = [0, 2, 3, 5, 7]
"""All the usable modes are speced below"""
storage_freq = [5.4605359569639695*1e9, 5.965085584240055*1e9, 6.511215233526263*1e9, 6.7611215233526263*1e9]
storage_LO = [5.56e9, 6.061e9, 6.61e9, 6.861e9]
storage_IF = [int(abs(storage_freq[i]-storage_LO[i])) for i in range(4)]
storage_mode = 1
storage_params = {
    'storage_mode': 1,
    'storage_LO': storage_LO,
    'storage_freq': storage_freq,
     'storage_IF':storage_IF,
     'iq_offset': [[-0.028, -0.044], [-0.025, -0.035], [-0.028, -0.0225], [-0.029, -0.023]],
     'chiby2pi_e': two_chi,
     'chi_2_by2pi': [0, 3.6317466677315835e-6, 1.6492042423932318e-6, 1.0194765942550532e-6],
     'st_self_kerr': [0, 5e3/2, 0, 0]
     }
####---------------------####
oct_len = 1000
opx_config = {

    'version': 1,

    'controllers': {

        'con1': {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': 0.010},#0.0095},  # qubit I
                2: {'offset':  -0.010},#-0.064},  # qubit Q
                3: {'offset': -0.010},  # RR I
                4: {'offset': 0.003},  # RR Q
                5: {'offset': storage_params['iq_offset'][storage_params['storage_mode']][0]},  # storage I
                6: {'offset': storage_params['iq_offset'][storage_params['storage_mode']][1]},  # storage Q
            },
            'digital_outputs': {},
            'analog_inputs': {
                1: {'offset': (675)/2**12, 'gain_db': -3},
                2: {'offset': (674)/2**12, 'gain_db':-3}
            }
        }
    },

    'elements': {

        **{
            params[0]: {
                'mixInputs': {
                    'I': ('con1', 1),
                    'Q': ('con1', 2),
                    'lo_frequency': qubit_LO,
                    'mixer': 'mixer_qubit'
                },
                'intermediate_frequency': params[1],
                'operations': {
                    'CW': 'CW',
                    'saturation': 'saturation_pulse',
                    'gaussian': 'gaussian_pulse',
                    'gaussian_16': 'gaussian_16_pulse',
                    'pi': 'pi_pulse',
                    'pi2': 'pi2_pulse',
                    'minus_pi2': 'minus_pi2_pulse',
                    'res_pi': params[2],
                    'res_2pi': params[3],
                    'qoct': 'qoct_pulse',
                },
            }
            for params in [
                ('qubit_mode0', ge_IF[0], 'res_pi_pulse_mode0', 'res_2pi_pulse_mode0'),
                ('qubit_mode1', ge_IF[1], 'res_pi_pulse_mode1', 'res_2pi_pulse_mode1'),
                ('qubit_mode2', ge_IF[2], 'res_pi_pulse_mode2', 'res_2pi_pulse_mode2'),
                ('qubit_mode3', ge_IF[3], 'res_pi_pulse_mode3', 'res_2pi_pulse_mode3')
            ]
        },

        'rr': {
            'mixInputs': {
                'I': ('con1', 3),
                'Q': ('con1', 4),
                'lo_frequency': rr_LO,
                'mixer': 'mixer_RR'
            },
            'intermediate_frequency': rr_IF,
            'operations': {
                'CW': 'CW',
                'readout': 'readout_pulse',
                'clear': 'clear_pulse',
            },
            "outputs": {
                'out1': ('con1', 1),
                'out2': ('con1', 2),
            },
            'time_of_flight': 320, # ns should be a multiple of 4
            'smearing': 0,
            # 'digitalInputs': {
            #     'lo_readout': {
            #         'port': ('con1', 1),
            #         'delay': 0,
            #         'buffer': 0
            #     },
            # },
        },

        **{
            params[0]: {
                'mixInputs': {
                    'I': ('con1', 5),
                    'Q': ('con1', 6),
                    'lo_frequency': params[1],
                    'mixer': 'mixer_storage'
                },
                'intermediate_frequency': params[2],
                'operations': {
                    'CW': 'CW',
                    'saturation': 'saturation_pulse',
                    'gaussian': 'gaussian_pulse',
                    'soct': 'soct_pulse',
                },
            }
            for params in [
                ('storage_mode0', storage_LO[0], storage_IF[0]),
                ('storage_mode1', storage_LO[1], storage_IF[1]),
                ('storage_mode2', storage_LO[2], storage_IF[2]),
                ('storage_mode3', storage_LO[3], storage_IF[3])
            ]
        },
        # 'digitalInputs': {
        #     'lo_readout': {
        #         'port': ('con1', 3),
        #         'delay': 0,
        #         'buffer': 0
        #     },
        # },
    },

    "pulses": {

        "CW": {
            'operation': 'control',
            'length': 600000,  #ns,
            'waveforms': {
                'I': 'const_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON'
        },

        "saturation_pulse": {
            'operation': 'control',
            'length': 200000,  # several T1s
            'waveforms': {
                'I': 'saturation_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON'
        },

        "gaussian_pulse": {
            'operation': 'control',
            'length': gauss_len,
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON'
        },

        "gaussian_16_pulse": {
            'operation': 'control',
            'length': 16,
            'waveforms': {
                'I': 'gauss_16_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON'
        },

        'pi_pulse': {
            'operation': 'control',
            'length': pi_len,
            'waveforms': {
                'I': 'pi_wf',
                'Q': 'zero_wf'
            },
        },

        'pi2_pulse': {
            'operation': 'control',
            'length': half_pi_len,
            'waveforms': {
                'I': 'pi2_wf',
                'Q': 'zero_wf'
            },
        },

        'minus_pi2_pulse': {
            'operation': 'control',
            'length': half_pi_len,
            'waveforms': {
                'I': 'minus_pi2_wf',
                'Q': 'zero_wf'
            }
        },

        **{
            params[0]: {
                'operation': 'control',
                'length': params[2],
                'waveforms': {
                    'I': params[1],
                    'Q': 'zero_wf'
                },
            }
            for params in [
                ('res_pi_pulse_mode0', 'res_pi_wf_mode0', res_pi_len[0]),
                ('res_pi_pulse_mode1', 'res_pi_wf_mode1', res_pi_len[1]),
                ('res_pi_pulse_mode2', 'res_pi_wf_mode2', res_pi_len[2]),
                ('res_pi_pulse_mode3', 'res_pi_wf_mode3', res_pi_len[3])
            ]
        },

        **{
            params[0]: {
                'operation': 'control',
                'length': params[2],
                'waveforms': {
                    'I': params[1],
                    'Q': 'zero_wf'
                },
            }
            for params in [
                ('res_2pi_pulse_mode0', 'res_2pi_wf_mode0', res_pi_len[0]),
                ('res_2pi_pulse_mode1', 'res_2pi_wf_mode1', res_pi_len[1]),
                ('res_2pi_pulse_mode2', 'res_2pi_wf_mode2', res_pi_len[2]),
                ('res_2pi_pulse_mode3', 'res_2pi_wf_mode3', res_pi_len[3])
            ]
        },

        'readout_pulse': {
            'operation': 'measurement',
            'length': readout_len,
            'waveforms': {
                'I': 'readout_wf',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'integW1': 'integW1',
                'integW2': 'integW2',
                'integW3': 'integW3',
            },
            'digital_marker': 'ON'
        },

        'clear_pulse': {
            'operation': 'measurement',
            'length': opt_len,
            'waveforms': {
                'I': 'opt_wf',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'clear_integW1': 'clear_integW1',
                'clear_integW2': 'clear_integW2',
            },
            'digital_marker': 'ON'
        },

        'qoct_pulse': {
            'operation': 'control',
            'length': oct_len,  #ns,
            'waveforms': {
                'I': 'qoct_wf_i',
                'Q': 'qoct_wf_q'
            },
        },

        'soct_pulse': {
            'operation': 'control',
            'length': oct_len,  #ns,
            'waveforms': {
                'I': 'soct_wf_i',
                'Q': 'soct_wf_q'
            },
        },
    },

    'waveforms': {

        'const_wf': {
            'type': 'constant',
            'sample': 0.45
        },

        'zero_wf': {
            'type': 'constant',
            'sample': 0.0
        },

        'saturation_wf': {
            'type': 'constant',
            'sample': 0.45 #earlier set to 0.1
        },

        'gauss_wf': {
            'type': 'arbitrary',
            'samples': gauss(gauss_amp, 0.0, gauss_len//4, gauss_len)
        },

        'gauss_16_wf': {
            'type': 'arbitrary',
            'samples': gauss(gauss_amp, 0.0, 16//4, 16)
        },

        'pi_wf': {
            'type': 'arbitrary',
            'samples': gauss(gauss_amp * pi_amp, 0.0, pi_len//4, pi_len)
        },

        'pi2_wf': {
            'type': 'arbitrary',
            'samples': gauss(gauss_amp * half_pi_amp, 0.0, half_pi_len//4, half_pi_len)
        },

        'minus_pi2_wf': {
            'type': 'arbitrary',
            'samples': gauss(-gauss_amp * half_pi_amp, 0.0, half_pi_len//4, half_pi_len)
        },

        **{
            params[0]: {
                'type': 'arbitrary',
                'samples': gauss(params[1] * gauss_amp, 0.0, params[2]//4, params[2])
            }
            for params in [
                ('res_pi_wf_mode0', res_pi_amps[0], res_pi_len[0]),
                ('res_pi_wf_mode1', res_pi_amps[1], res_pi_len[1]),
                ('res_pi_wf_mode2', res_pi_amps[2], res_pi_len[2]),
                ('res_pi_wf_mode3', res_pi_amps[3], res_pi_len[3]),
            ]
        },

        **{
            params[0]: {
                'type': 'arbitrary',
                'samples': gauss(2*params[1] * gauss_amp, 0.0, params[2]//4, params[2])
            }
            for params in [
                ('res_2pi_wf_mode0', res_pi_amps[0], res_pi_len[0]),
                ('res_2pi_wf_mode1', res_pi_amps[1], res_pi_len[1]),
                ('res_2pi_wf_mode2', res_pi_amps[2], res_pi_len[2]),
                ('res_2pi_wf_mode3', res_pi_amps[3], res_pi_len[3]),
            ]
        },

        'pi_wf_ef': {
            'type': 'arbitrary',
            'samples': gauss(gauss_amp * pi_ef_amp, 0.0, pi_ef_len//4, pi_ef_len)
        },

        'pi2_wf_ef': {
            'type': 'arbitrary',
            'samples': gauss(gauss_amp * pi_2_ef_amp, 0.0, pi_ef_len//4, pi_ef_len)
        },

        'minus_pi2_wf_ef': {
            'type': 'arbitrary',
            'samples': gauss(-gauss_amp * pi_2_ef_amp, 0.0, pi_ef_len//4, pi_ef_len)
        },

        'readout_wf': {
            'type': 'constant',
            'sample': readout_params['sq_amp']
        },

        'opt_wf': {
            'type': 'arbitrary',
            'samples': 0.45 * opt_amp
        },

        'pump_wf': {
            'type': 'constant',
            'sample': 0.45 * readout_params['pump_amp']
        },

        'qoct_wf_i': {
            'type': 'arbitrary',
            'samples': gauss(gauss_amp, 0.0, oct_len//4, oct_len)
        },

        'qoct_wf_q': {
            'type': 'arbitrary',
            'samples':  gauss(gauss_amp, 0.0, oct_len//4, oct_len)
        },

        'soct_wf_i': {
            'type': 'arbitrary',
            'samples': gauss(gauss_amp, 0.0, oct_len//4, oct_len)
        },

        'soct_wf_q': {
            'type': 'arbitrary',
            'samples': gauss(gauss_amp, 0.0, oct_len//4, oct_len)
        },
    },

    'digital_waveforms': {
        'ON': {
            'samples': [(1, 0)]
        }
    },

    'integration_weights': {

        'integW1': {
            'cosine': [2.0] * int(readout_len / 4),
            'sine': [0.0] * int(readout_len / 4)
        },

        'integW2': {
            'cosine': [0.0] * int(readout_len / 4),
            'sine': [2.0] * int(readout_len / 4)
        },

        'integW3': {
            'cosine': [0.0] * int(readout_len / 4),
            'sine': [-2.0] * int(readout_len / 4)
        },
        'clear_integW1': {
            'cosine': [2.0] * int(opt_len / 4 ),
            'sine': [0.0] * int(opt_len / 4 )
        },

        'clear_integW2': {
            'cosine': [0.0] * int(opt_len / 4 ),
            'sine': [2.0] * int(opt_len / 4 )
        },

        'demod1_iw': {
            'cosine': [2.0] * int(readout_len / 4),
            'sine': [0.0] * int(readout_len / 4),
        },

        'demod2_iw': {
            'cosine': [0.0] * int(readout_len / 4),
            'sine': [2.0] * int(readout_len / 4),
        },

        'optW1': {
            'cosine': [2.0] * int(readout_len / 4),
            'sine': [0.0] * int(readout_len / 4)
        },

        'optW2': {
            'cosine': [0.0] * int(readout_len / 4),
            'sine': [2.0] * int(readout_len / 4)
        },
    },

    'mixers': {

        'mixer_qubit': [
            {'intermediate_frequency': ge_IF[0], 'lo_frequency': qubit_LO, 'correction': IQ_imbalance(-0.004, 0.01 * np.pi)},
            {'intermediate_frequency': ge_IF[1], 'lo_frequency': qubit_LO, 'correction': IQ_imbalance(-0.0, 0.0 * np.pi)},
            {'intermediate_frequency': ge_IF[2], 'lo_frequency': qubit_LO, 'correction': IQ_imbalance(-0.0, 0.0 * np.pi)},
            {'intermediate_frequency': ge_IF[3], 'lo_frequency': qubit_LO, 'correction': IQ_imbalance(-0.0, 0.0 * np.pi)},
        ],

        'mixer_RR': [
            {'intermediate_frequency': rr_IF, 'lo_frequency': rr_LO,
             'correction': IQ_imbalance(-0.0085, 0.013 * np.pi)}
        ],

        'mixer_storage': [
            {'intermediate_frequency': storage_IF[0], 'lo_frequency': storage_LO[0], 'correction': IQ_imbalance(-0.01, 0.018 * np.pi)},
            {'intermediate_frequency': storage_IF[1], 'lo_frequency': storage_LO[1], 'correction': IQ_imbalance(0.00, 0.010* np.pi)},
            {'intermediate_frequency': storage_IF[2], 'lo_frequency': storage_LO[2], 'correction': IQ_imbalance(-0.015, 0.017 * np.pi)},
            {'intermediate_frequency': storage_IF[3], 'lo_frequency': storage_LO[3], 'correction': IQ_imbalance(-0.018, 0.0125 * np.pi)},
        ],

    }

}

storage_cal_file = ['',
                    'C:\_Lib\python\slab\experiments\qm_opx_mm\drive_calibration/00000_2021_12_16_cavity_square_mode_2.h5',
                    '',
                    '']
