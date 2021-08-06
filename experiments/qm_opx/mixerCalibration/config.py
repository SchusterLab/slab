import numpy as np

# 'gauss_wf': {
#    'type': 'arbitrary',
#    'samples': gauss(0.4, 0.0, 6.0, 60)
# },

qubit_IF = 100e6;  # 100e6
qubit_LO = 3.645e9;  # ;
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
                1: {'offset': 0.0},  # Qubit I 
                2: {'offset': 0.0},  # Qubit Q
                5: {'offset': 0.0},  # Cavity I
                6: {'offset': 0.0},  # Cavity Q
            },
            'digital_outputs': {},
            'analog_inputs': {
                1: {'offset': 0.0},  # 0.02563476562; -0.00109863281}, #1: {'offset': 0.01220703125}
                2: {'offset': 0.0}, # 0.04125976562 + 0.00561523437 - 0.00024414062 + 0.02
            }
        }
    },

    'elements': {

        'qubit': {
            'mixInputs': {
                'I': ('con1', 1),  ## Analog DAC 1 -> qubit I
                'Q': ('con1', 2),  ## Analog DAC 2 -> qubit Q
                'lo_frequency': qubit_LO,
                'mixer': 'mixer_q'
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
            'sample': 0.2
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
        'mixer_q': [
            {'intermediate_frequency': qubit_IF, 'lo_frequency': qubit_LO, 'correction': [1.0, 0.0, 0.0, 1.0]}
        ],
        'mixer_cavity': [
            {'intermediate_frequency': cavity_IF, 'lo_frequency': cavity_LO, 'correction': [1.0, 0.0, 0.0, 1.0]}
        ],
    }
}
