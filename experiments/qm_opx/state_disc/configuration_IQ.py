import numpy as np
import pandas as pd
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
long_redout_len = 4000
readout_len = 3000

qubit_freq = 4.746938417620596 *1e9
qubit_ef_freq = 4.6078190022032635*1e9
ge_IF = 100e6
ef_IF = int(ge_IF - (qubit_freq-qubit_ef_freq))
qubit_LO = qubit_freq - ge_IF

####---------------------####
rr_freq_g = 8.05182319e9
rr_freq_e = 8.05145304e9

# rr_freq = 0.5*(rr_freq_g + rr_freq_e) #between g and e
rr_freq = 8.051658e9 #for 4us
# rr_freq = 8.051618e9 #for 3.6us

rr_IF = 100e6
rr_LO = rr_freq - rr_IF
# rr_amp = 0.5*0.7
rr_amp = 1.0*0.295

pump_LO = rr_LO
pump_IF = 100e6

pump_amp = 1.0
pump_len = long_redout_len
####---------------------####
storage_freq = 6.01124448e9
storage_LO = 5.911e9
storage_IF = storage_freq-storage_LO
# storage_LO = storage_freq - storage_IF

gauss_len = 32
gauss_amp = 0.45

pi_len = 32
pi_amp = 0.3508600798950383

half_pi_len = 16
half_pi_amp = pi_amp

pi_len_resolved = 3000
Pi_amp_resolved = 0.00884993938365933

pi_ef_len = 32
pi_ef_amp = 0.2843

data = pd.read_csv("S:\\Morgan\\qm_opx\\morgan\\pulse_shape\\pulse.csv")
amp = np.array(pd.DataFrame(data['amp']))
clear_amp = 1.0*0.35*(amp/np.max(amp))
clear_len = len(clear_amp)

config = {

    'version': 1,

    'controllers': {
        'con1': {
            'type': 'opx1',
            'analog_outputs': {
                3: {'offset': 0.0199},  # RR I
                4: {'offset': -0.04},  # RR Q
                1: {'offset': 0.0118},  # qubit I
                2: {'offset': -0.0660},  # qubit Q
                7: {'offset': 0.0158},  # storage I
                8: {'offset': -0.067},  # storage Q
                9: {'offset': 0.0259},  # JPA Pump I
                10: {'offset': -0.0366},  # JPA Pump Q

            },
            'digital_outputs': {},
            'analog_inputs': {
                1: {'offset': 0.006, 'gain_db': 2},
                2: {'offset': 0.0215, 'gain_db': 2}
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
            },
            'digitalInputs': {
                'lo_qubit': {
                    'port': ('con1', 2),
                    'delay': 72,
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
                    'delay': 72,
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
            'time_of_flight': 340, # ns should be a multiple of 4
            'smearing': 0,
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
            },
            'digitalInputs': {
                'lo_storage': {
                    'port': ('con1', 3),
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
            'length': clear_len,
            'waveforms': {
                'I': 'clear_wf',
                'Q': 'zero_wf'
            },
            'integration_weights': {
                'clear_integW1': 'clear_integW1',
                'clear_integW2': 'clear_integW2',
            },
            'digital_marker': 'ON'
        },

        "pump_square": {
            'operation': 'control',
            'length': pump_len,  #ns,
            'waveforms': {
                'I': 'pump_wf',
                'Q': 'zero_wf'
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
        'clear_wf': {
            'type': 'arbitrary',
            'samples': 0.45 * clear_amp
        },
        'pump_wf': {
            'type': 'constant',
            'sample': 0.45 * pump_amp
        },
    },

    'digital_waveforms': {
        'ON': {
            'samples': [(1, 0)]
        }
    },

    'integration_weights': {

        'long_integW1': {
            'cosine': [2.0] * int(long_redout_len / 4  + 60),
            'sine': [0.0] * int(long_redout_len / 4  + 60)
        },

        'long_integW2': {
            'cosine': [0.0] * int(long_redout_len / 4 + 60),
            'sine': [2.0] * int(long_redout_len / 4 + 60)
        },
        'clear_integW1': {
            'cosine': [2.0] * int(clear_len / 4 ),
            'sine': [0.0] * int(clear_len / 4 )
        },

        'clear_integW2': {
            'cosine': [0.0] * int(clear_len / 4 ),
            'sine': [2.0] * int(clear_len / 4 )
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
             'correction': IQ_imbalance(-0.010, 0.0175 * np.pi)},
            {'intermediate_frequency': ef_IF, 'lo_frequency': qubit_LO,
             'correction': IQ_imbalance(-0.015, 0.028 * np.pi)}
        ],

        'mixer_RR': [
            {'intermediate_frequency': rr_IF, 'lo_frequency': rr_LO,
             'correction': IQ_imbalance(0.007, 0.046 * np.pi)}
        ],

        'mixer_storage': [
            {'intermediate_frequency': storage_IF, 'lo_frequency': storage_LO,
             'correction': IQ_imbalance(0.001, 0.015 * np.pi)}
        ],
        'mixer_jpa': [
            {'intermediate_frequency': pump_IF, 'lo_frequency': pump_LO,
             'correction': IQ_imbalance(0.041, 0.018 * np.pi)}
        ],

    }

}