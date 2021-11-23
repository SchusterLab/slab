import numpy as np
import pandas as pd
from h5py import File
######################
# AUXILIARY FUNCTIONS:
######################
def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    return [float(x) for x in gauss_wave]

def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1-g**2)*(2*c**2-1))
    return [float(N * x) for x in [(1-g)*c, (1+g)*s, (1-g)*s, (1+g)*c]]

################
# CONFIGURATION:
################
long_redout_len = 3500
readout_len = 3000

qubit_freq = 4.746909500096027 *1e9
qubit_ef_freq = 4.6078190022032635*1e9
ge_IF = 100e6
ef_IF = -39.6814108981237e6
qubit_LO = qubit_freq - ge_IF

####---------------------####
rr_freq_g = 8.051843423081882e9
rr_freq_e = 8.051472688135474e9
rr_freq = 8.051666263034072e9 #for 4us
# rr_freq = rr_freq_e
rr_IF = 100e6
rr_LO = rr_freq + rr_IF
rr_amp = 1.0 *0.4

pump_LO = rr_LO
pump_IF = 1006

pump_amp = 1.0*0.01
pump_len = long_redout_len
####---------------------####
storage_freq = 6.01124448e9
storage_LO = 6.111e9
storage_IF = abs(storage_freq-storage_LO)
# storage_LO = storage_freq - storage_IF

sb_freq = 3.3434e9
sb_IF = 100e6
sb_LO = sb_freq + sb_IF

gauss_len = 32
gauss_amp = 0.45

pi_len = 32
pi_amp = 0.3507

half_pi_len = 16
half_pi_amp = pi_amp

pi_len_resolved = 3000
Pi_amp_resolved = 0.0036

pi_ef_len = 32
pi_ef_amp = 0.4222

opt_readout = "C:\\_Lib\\python\\slab\\experiments\\qm_opx\\pulses\\00014_readout_optimal_pulse.h5"
with File(opt_readout,'r') as a:
    opt_amp = np.array(a['I_wf'])
opt_len = len(opt_amp)

oct_len = 1000

config = {

    'version': 1,

    'controllers': {
        'con1': {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': 0.0162},  # qubit I
                2: {'offset':  -0.0706},  # qubit Q
                3: {'offset': -0.0025},  # RR I
                4: {'offset': 0.0038},  # RR Q
                5: {'offset': -0.021},  # sb I
                6: {'offset': -0.058},  # sb Q
                7: {'offset': -0.020},  # storage I
                8: {'offset': -0.0024},  # storage Q
                9: {'offset': -0.002},#0.0254},  # JPA Pump I
                10: {'offset': 0.005},#-0.0375},  # JPA Pump Q

            },
            'digital_outputs': {},
            'analog_inputs': {
                1: {'offset': 5.6/2**12, 'gain_db': 0},
                2: {'offset': 2.3/2**12, 'gain_db': 0}
            }
        }
    },

    'elements': {

        'qubit': {
            'mixInputs': {
                'I': ('con1', 1),
                'Q': ('con1', 2),
                'lo_frequency': qubit_LO,
                'mixer': 'mixer_qubit'
            },
            'intermediate_frequency': ge_IF,
            'operations': {
                'CW': 'CW',
                'saturation': 'saturation_pulse',
                'gaussian': 'gaussian_pulse',
                'gaussian_16': 'gaussian_16_pulse',
                'pi': 'pi_pulse',
                'pi2': 'pi2_pulse',
                'minus_pi2': 'minus_pi2_pulse',
                'res_pi': 'res_pi_pulse',
                # 'qoct': 'qoct_pulse',

            },
            'digitalInputs': {
                'lo_qubit': {
                    'port': ('con1', 2),
                    'delay': 0,
                    'buffer': 0
                },
            },
        },
        'qubit_ef': {
            'mixInputs': {
                'I': ('con1', 1),
                'Q': ('con1', 2),
                'lo_frequency': qubit_LO,
                'mixer': 'mixer_qubit'
            },
            'intermediate_frequency': ef_IF,
            'operations': {
                'CW': 'CW',
                'saturation': 'saturation_pulse',
                'gaussian': 'gaussian_pulse',
                'pi': 'pi_pulse_ef',
                'pi2': 'pi2_pulse_ef',
                'minus_pi2': 'minus_pi2_pulse_ef',
            },
            'digitalInputs': {
                'lo_qubit': {
                    'port': ('con1', 2),
                    'delay': 0,
                    'buffer': 0
                },
            },
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
                'long_readout': 'long_readout_pulse',
                'readout': 'readout_pulse',
                'clear': 'clear_pulse',
            },
            "outputs": {
                'out1': ('con1', 1),
                'out2': ('con1', 2),
            },
            'time_of_flight': 360+124-52, # ns should be a multiple of 4
            'smearing': 124-52,
            'digitalInputs': {
                'lo_readout': {
                    'port': ('con1', 1),
                    'delay': 0,
                    'buffer': 0
                },
            },
        },
        'storage': {
            'mixInputs': {
                'I': ('con1', 7),
                'Q': ('con1', 8),
                'lo_frequency': storage_LO,
                'mixer': 'mixer_storage'
            },
            'intermediate_frequency': storage_IF,
            'operations': {
                'CW': 'CW',
                'saturation': 'saturation_pulse',
                'gaussian': 'gaussian_pulse',
                'pi': 'pi_pulse',
                'pi2': 'pi2_pulse',
                'minus_pi2': 'minus_pi2_pulse',
                # 'soct': 'soct_pulse',

            },
            'digitalInputs': {
                'lo_storage': {
                    'port': ('con1', 3),
                    'delay': 128,
                    'buffer': 0
                },
            },
        },
        'sideband': {
            'mixInputs': {
                'I': ('con1', 5),
                'Q': ('con1', 6),
                'lo_frequency': sb_LO,
                'mixer': 'mixer_sb'
            },
            'intermediate_frequency': sb_IF,
            'operations': {
                'CW': 'CW',
                'saturation': 'saturation_pulse',
                'gaussian': 'gaussian_pulse',
                'pi': 'pi_pulse',
                'pi2': 'pi2_pulse',
                'minus_pi2': 'minus_pi2_pulse',
            },
            'digitalInputs': {
                'lo_storage': {
                    'port': ('con1', 6),
                    'delay': 128,
                    'buffer': 0
                },
            },
        },

        'jpa_pump': {
            'mixInputs': {
                'I': ('con1', 9),
                'Q': ('con1', 10),
                'lo_frequency': pump_LO,
                'mixer': 'mixer_jpa'
            },
            'intermediate_frequency': pump_IF,
            'operations': {
                'CW': 'CW',
                'saturation': 'saturation_pulse',
                'gaussian': 'gaussian_pulse',
                'pi': 'pi_pulse',
                'pi2': 'pi2_pulse',
                'minus_pi2': 'minus_pi2_pulse',
                'pump_square': 'pump_square',
            },
        },
    },

    "pulses": {

        "CW": {
            'operation': 'control',
            'length': 60000,  #ns,
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
            'digital_marker': 'ON'
        },

        'pi2_pulse': {
            'operation': 'control',
            'length': half_pi_len,
            'waveforms': {
                'I': 'pi2_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON'
        },

        'minus_pi2_pulse': {
            'operation': 'control',
            'length': half_pi_len,
            'waveforms': {
                'I': 'minus_pi2_wf',
                'Q': 'zero_wf'
            }
        },

        'res_pi_pulse': {
            'operation': 'control',
            'length': pi_len_resolved,
            'waveforms': {
                'I': 'res_pi_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON'
        },

        'pi_pulse_ef': {
            'operation': 'control',
            'length': pi_ef_len,
            'waveforms': {
                'I': 'pi_wf_ef',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON'
        },

        'pi2_pulse_ef': {
            'operation': 'control',
            'length': pi_ef_len//2,
            'waveforms': {
                'I': 'pi2_wf_ef',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON'
        },

        'minus_pi2_pulse_ef': {
            'operation': 'control',
            'length': pi_ef_len//2,
            'waveforms': {
                'I': 'minus_pi2_wf_ef',
                'Q': 'zero_wf'
            }
        },

        'long_readout_pulse': {
            'operation': 'measurement',
            'length': long_redout_len,
            'waveforms': {
                'I': 'long_readout_wf',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'long_integW1': 'long_integW1',
                'long_integW2': 'long_integW2',
            },
            'digital_marker': 'ON'
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
                'optW1': 'optW1',
                'optW2': 'optW2'
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

        'pump_square': {
            'operation': 'control',
            'length': pump_len,  #ns,
            'waveforms': {
                'I': 'pump_wf',
                'Q': 'zero_wf'
            },
        },

        # 'qoct_pulse': {
        #     'operation': 'control',
        #     'length': oct_len,  #ns,
        #     'waveforms': {
        #         'I': 'qoct_wf_i',
        #         'Q': 'qoct_wf_q'
        #     },
        # },
        #
        # 'soct_pulse': {
        #     'operation': 'control',
        #     'length': oct_len,  #ns,
        #     'waveforms': {
        #         'I': 'soct_wf_i',
        #         'Q': 'soct_wf_q'
        #     },
        # },

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

        'res_pi_wf': {
            'type': 'arbitrary',
            'samples': gauss(gauss_amp * Pi_amp_resolved, 0.0, pi_len_resolved//4, pi_len_resolved)
        },

        'pi_wf_ef': {
            'type': 'arbitrary',
            'samples': gauss(gauss_amp * pi_ef_amp, 0.0, pi_ef_len//4, pi_ef_len)
        },

        'pi2_wf_ef': {
            'type': 'arbitrary',
            'samples': gauss(gauss_amp * pi_ef_amp, 0.0, pi_ef_len//8, pi_ef_len//2)
        },

        'minus_pi2_wf_ef': {
            'type': 'arbitrary',
            'samples': gauss(-gauss_amp * pi_ef_amp, 0.0, pi_ef_len//8, pi_ef_len//2)
        },

        'long_readout_wf': {
            'type': 'constant',
            'sample': 0.45 * rr_amp
        },

        'readout_wf': {
            'type': 'constant',
            'sample': 0.45
        },
        'opt_wf': {
            'type': 'arbitrary',
            'samples': 0.45 * opt_amp
        },
        'pump_wf': {
            'type': 'constant',
            'sample': 0.45 * pump_amp
        },
        # 'qoct_wf_i': {
        #     'type': 'arbitrary',
        #     'samples': gauss(gauss_amp, 0.0, oct_len//4, oct_len)
        # },
        #
        # 'qoct_wf_q': {
        #     'type': 'arbitrary',
        #     'samples':  gauss(gauss_amp, 0.0, oct_len//4, oct_len)
        # },
        # 'soct_wf_i': {
        #     'type': 'arbitrary',
        #     'samples': gauss(gauss_amp, 0.0, oct_len//4, oct_len)
        # },
        #
        # 'soct_wf_q': {
        #     'type': 'arbitrary',
        #     'samples': gauss(gauss_amp, 0.0, oct_len//4, oct_len)
        # },

        'digital_waveforms': {
            'ON': {
                'samples': [(1, 0)]
            }
        },
    },

    'integration_weights': {

        'long_integW1': {
            'cosine': [2.0] * int(long_redout_len / 4),
            'sine': [0.0] * int(long_redout_len / 4)
        },

        'long_integW2': {
            'cosine': [0.0] * int(long_redout_len / 4),
            'sine': [2.0] * int(long_redout_len / 4)
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
            'cosine': [2.0] * int(long_redout_len / 4),
            'sine': [0.0] * int(long_redout_len / 4),
        },

        'demod2_iw': {
            'cosine': [0.0] * int(long_redout_len / 4),
            'sine': [2.0] * int(long_redout_len / 4),
        },

        'optW1': {
            'cosine': [1.0] * int(readout_len / 4),
            'sine': [0.0] * int(readout_len / 4)
        },

        'optW2': {
            'cosine': [0.0] * int(readout_len / 4),
            'sine': [1.0] * int(readout_len / 4)
        },
    },

    'mixers': {
        'mixer_qubit': [
            {'intermediate_frequency': ge_IF, 'lo_frequency': qubit_LO,
             'correction': IQ_imbalance(-0.015, 0.0175 * np.pi)},
            {'intermediate_frequency': ef_IF, 'lo_frequency': qubit_LO,
             'correction': IQ_imbalance(-0.015, 0.012 * np.pi)}
        ],

        'mixer_RR': [
            {'intermediate_frequency': rr_IF, 'lo_frequency': rr_LO,
             'correction': IQ_imbalance(-0.01, 0.013 * np.pi)}
        ],

        'mixer_storage': [
            {'intermediate_frequency': storage_IF, 'lo_frequency': storage_LO,
             'correction': IQ_imbalance(0.01, 0.044 * np.pi)}#IQ_imbalance(0.001, 0.015 * np.pi)}
        ],

        'mixer_sb': [
            {'intermediate_frequency': sb_IF, 'lo_frequency': sb_LO,
             'correction': IQ_imbalance(-0.010, 0.048 * np.pi)}#IQ_imbalance(0.001, 0.015 * np.pi)}
        ],

        'mixer_jpa': [
            {'intermediate_frequency': pump_IF, 'lo_frequency': pump_LO,
             'correction': IQ_imbalance(-0.005, 0.005 * np.pi)}
        ],

    }

}