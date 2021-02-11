import numpy as np

# 'gauss_wf': {
#    'type': 'arbitrary',
#    'samples': gauss(0.4, 0.0, 6.0, 60)
# },
def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1-g**2)*(2*c**2-1))
    return [float(N * x) for x in [(1-g)*c, (1+g)*s, (1-g)*s, (1+g)*c]]

qubit_IF = 100e6;  # 100e6
qubit_freq = 8.0517e9;
qubit_LO = qubit_freq - qubit_IF;  # ;
cavity_LO = 7.1e9;
cavity_IF = 114.57e6;
readout_len = 3600;  # 200;

def gauss(amplitude, mu, sigma, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    return [float(x) for x in gauss_wave]


config = {

    'version': 1,

    'controllers': {
        'con1': {
            'type': 'opx1',
            'analog_outputs': {
                9: {'offset': 0.0164},#0.0259},  # Qubit I
                10: {'offset': -0.037},#-0.0366},  # Qubit Q
                5: {'offset': 0.0},  # Cavity I
                6: {'offset': 0.0},  # Cavity Q
            },
            'digital_outputs': {},
            'analog_inputs': {
                1: {'offset': 0.0},
                2: {'offset': 0.0},
            }
        }
    },

    'elements': {

        'qubit': {
            'mixInputs': {
                'I': ('con1', 9),  ## Analog DAC 1 -> qubit I
                'Q': ('con1', 10),  ## Analog DAC 2 -> qubit Q
                'lo_frequency': qubit_LO,
                'mixer': 'mixer_qubit'
            },
            'intermediate_frequency': qubit_IF,
            'operations': {
                'my_control_op': 'my_control_pulse',
                'ps_op': 'ps_pulse',
                'gaussian': 'gaussian_pulse',
                'pi_op': 'pi_pulse',
            },
        },

        'cavity': {
            'mixInputs': {
                'I': ('con1', 5),
                'Q': ('con1', 6),
                'lo_frequency': cavity_LO,
                'mixer': 'mixer_cavity'
            },
            'intermediate_frequency': cavity_IF,
            'operations': {
                'my_control_op': 'my_control_pulse',
                'my_readout_op': 'my_readout_pulse',
            },
            "outputs": {
                'out1': ('con1', 1),
                'out2': ('con1', 2),
            },
            'time_of_flight': 192,  # 192 ns multiple of 4
            'smearing': 0
        },
    },

    "pulses": {

        "my_control_pulse": {
            'operation': 'control',
            'length': 60000,
            'waveforms': {
                'I': 'my_control_wf',
                # 'Q': 'my_control_wf',
                'Q': 'zero_wf',

            }
        },

        "ps_pulse": {
            'operation': 'control',
            'length': readout_len,
            'waveforms': {
                'I': 'my_control_wf',
                # 'Q': 'my_control_wf',
                'Q': 'zero_wf',

            }
        },

        "gaussian_pulse": {
            'operation': 'control',
            'length': 200,
            'waveforms': {
                'I': 'gauss_wf',
                'Q': 'zero_wf'
            }
        },

        "pi_pulse": {
            'operation': 'control',
            'length': 200,
            'waveforms': {
                'I': 'pi_wf',
                'Q': 'zero_wf'
            }
        },

        'my_readout_pulse': {

            'operation': 'measurement',
            'length': readout_len,
            'waveforms': {
                'I': 'my_readout_wf',
                'Q': 'zero_wf',
            },
            'integration_weights': {
                'integW1': 'my_integW1',
                'integW2': 'my_integW2',
            },
            'digital_marker': 'ON'
        },

    },

    'waveforms': {

        'my_control_wf': {
            'type': 'constant',
            'sample': 0.45
        },

        'my_readout_wf': {
            'type': 'constant',
            'sample': 0.2
        },

        'zero_wf': {
            'type': 'constant',
            'sample': 0.0
        },

        'gauss_wf': {
            'type': 'arbitrary',
            'samples': gauss(0.2, 0.0, 25.0, 200)
        },

        'pi_wf': {
            'type': 'arbitrary',
            'samples': gauss(0.0375, 0.0, 20.0, 200)
        },

    },

    'digital_waveforms': {
        'ON': {
            'samples': [(1, 0)]
        }
    },

    'integration_weights': {

        'my_integW1': {
            'cosine': [2.0] * int(readout_len / 4),
            'sine': [0.0] * int(readout_len / 4)
        },

        'my_integW2': {
            'cosine': [0.0] * int(readout_len / 4),
            'sine': [2.0] * int(readout_len / 4)
        },
    },

    'mixers': {
        'mixer_qubit': [
            {'intermediate_frequency': qubit_IF, 'lo_frequency': qubit_LO, 'correction': IQ_imbalance(0.0,0.0 * np.pi)}
        ],
        'mixer_cavity': [
            {'intermediate_frequency': cavity_IF, 'lo_frequency': cavity_LO, 'correction': IQ_imbalance(0.041,0.018 * np.pi)}
        ],
    }
}
