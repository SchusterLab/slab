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
long_redout_len = 3600
readout_len = 3000

qubit_freq = 4.748111765982531e9
qubit_ef_freq = 4.6078190022032635e9
ge_IF = 100e6
ef_IF = int(ge_IF - (qubit_freq-qubit_ef_freq))
qubit_LO = qubit_freq - ge_IF

# rr_freq = 0.5*(8.05174438 + 8.05140573)*1e9 #between g and e
rr_freq = 8.0517e9
rr_IF = 100e6
rr_LO = rr_freq - rr_IF
rr_amp = 1.0

gauss_len = 32
pi_len = 32
pi_amp = 0.41
pi_ef_len = 32
pi_ef_amp = 0.2843

data = pd.read_csv("C:\\_Lib\python\\slab\\experiments\\qm_opx\\data\\clear_pulse_3.csv")
amp = np.array(pd.DataFrame(data['amp']))
clear_amp = amp[1500:-490]/np.max(amp)/1.0
clear_len = len(clear_amp)

config = {

    'version': 1,

    'controllers': {
        'con1': {
            'type': 'opx1',
            'analog_outputs': {
                5: {'offset': 0.0002},#-0.0035},  # qubit I
                6: {'offset': -0.0365},#-0.0342},  # qubit Q
                3: {'offset': 0.012},  # RR I
                4: {'offset': -0.0325},  # RR Q
            },
            'digital_outputs': {},
            'analog_inputs': {
                1: {'offset': 0.0},
                2: {'offset': 0.0}
            }
        }
    },

    'elements': {

        'qubit': {
            'mixInputs': {
                'I': ('con1', 5),
                'Q': ('con1', 6),
                'lo_frequency': qubit_LO,
                'mixer': 'mixer_qubit'
            },
            'intermediate_frequency': ge_IF,
            'operations': {
                'CW': 'CW',
                'saturation': 'saturation_pulse',
                'gaussian': 'gaussian_pulse',
                'pi': 'pi_pulse',
                'pi2': 'pi2_pulse',
                'minus_pi2': 'minus_pi2_pulse',
            },
            'time_of_flight': 160, # ns should be a multiple of 4
            'smearing': 0,
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
                'I': ('con1', 5),
                'Q': ('con1', 6),
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
            'time_of_flight': 160,  # ns should be a multiple of 4
            'smearing': 0,
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
                'clear':'clear_pulse',
            },
            "outputs": {
                'out1': ('con1', 1),
                'out2': ('con1', 2),
            },
            'time_of_flight': 360, # ns should be a multiple of 4
            'smearing': 0,
            'digitalInputs': {
                'lo_readout': {
                    'port': ('con1', 1),
                    'delay': 0,
                    'buffer': 0
                },
            },
        },

    },

    "pulses": {

        "CW": {
            'operation': 'control',
            'length': 600,  #ns,
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
            'length': pi_len//2,
            'waveforms': {
                'I': 'pi2_wf',
                'Q': 'zero_wf'
            },
            'digital_marker': 'ON'
        },

        'minus_pi2_pulse': {
            'operation': 'control',
            'length': pi_len//2,
            'waveforms': {
                'I': 'minus_pi2_wf',
                'Q': 'zero_wf'
            }
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

    },

    'waveforms': {

        'const_wf': {
            'type': 'constant',
            'sample': 0.4
        },

        'zero_wf': {
            'type': 'constant',
            'sample': 0.0
        },

        'saturation_wf': {
            'type': 'constant',
            'sample': 0.1 #earlier set to 0.1
        },

        'gauss_wf': {
            'type': 'arbitrary',
            'samples': gauss(0.3, 0.0, gauss_len//4, gauss_len)
        },

        'pi_wf': {
            'type': 'arbitrary',
            'samples': gauss(0.3 * pi_amp, 0.0, pi_len//4, pi_len)
        },

        'pi2_wf': {
            'type': 'arbitrary',
            'samples': gauss(0.3 * pi_amp, 0.0, pi_len//8, pi_len//2)
        },

        'minus_pi2_wf': {
            'type': 'arbitrary',
            'samples': gauss(-0.3 * pi_amp,  0.0, pi_len//8, pi_len//2)
        },

        'pi_wf_ef': {
            'type': 'arbitrary',
            'samples': gauss(0.3 * pi_ef_amp, 0.0, pi_ef_len//4, pi_ef_len)
        },

        'pi2_wf_ef': {
            'type': 'arbitrary',
            'samples': gauss(0.3 * pi_ef_amp, 0.0, pi_ef_len//8, pi_ef_len//2)
        },

        'minus_pi2_wf_ef': {
            'type': 'arbitrary',
            'samples': gauss(-0.3 * pi_ef_amp, 0.0, pi_ef_len//8, pi_ef_len//2)
        },

        'long_readout_wf': {
            'type': 'constant',
            'sample': 0.40 * rr_amp
        },

        'readout_wf': {
            'type': 'constant',
            'sample': 0.40
        },
        'clear_wf': {
            'type': 'arbitrary',
            'samples': 0.40 * clear_amp
        },
    },

    'digital_waveforms': {
        'ON': {
            'samples': [(1, 0)]
        }
    },

    'integration_weights': {

        'long_integW1': {
            'cosine': [2.0] * int(long_redout_len / 4 - 250),
            'sine': [0.0] * int(long_redout_len / 4 - 250)
        },

        'long_integW2': {
            'cosine': [0.0] * int(long_redout_len / 4 - 250),
            'sine': [2.0] * int(long_redout_len / 4 - 250)
        },
        'clear_integW1': {
            'cosine': [2.0] * int(clear_len / 4 - 250),
            'sine': [0.0] * int(clear_len / 4 - 250)
        },

        'clear_integW2': {
            'cosine': [0.0] * int(clear_len / 4 - 250),
            'sine': [2.0] * int(clear_len / 4 - 250)
        },

        'integW1': {
            'cosine': [1.0] * int(readout_len / 4),
            'sine': [0.0] * int(readout_len / 4),
        },

        'integW2': {
            'cosine': [0.0] * int(readout_len / 4),
            'sine': [1.0] * int(readout_len / 4),
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
             'correction': IQ_imbalance(-0.015, 0.028*np.pi)},
            {'intermediate_frequency': ef_IF, 'lo_frequency': qubit_LO,
             'correction': IQ_imbalance(-0.015, 0.028 * np.pi)}
        ],

        'mixer_RR': [
            {'intermediate_frequency': rr_IF, 'lo_frequency': rr_LO,
             'correction': IQ_imbalance(-0.045, -0.0015 * np.pi)}
        ],
    }

}