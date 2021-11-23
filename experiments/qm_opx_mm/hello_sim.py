from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from qm import LoopbackInterface
import pprint
import matplotlib.pyplot as plt

config = {

    'version': 1,

    'controllers': {
        'con1': {
            'type': 'opx1',
            'analog_outputs': {
                1: {'offset': 0.0},  # element1
                2: {'offset': 0.0},  # element2
            },
            'digital_outputs': {},
            'analog_inputs': {
                1: {'offset': 0.0}
            }
        }
    },

    'elements': {

        'my_element1': {
            'singleInput': {
                'port': ('con1', 1),
            },
            'intermediate_frequency': 50e6,
            'operations': {
                'my_control_op': 'my_control_pulse',
            }
        },

        'my_element2': {
            'singleInput': {
                'port': ('con1', 2)
            },
            'intermediate_frequency': 25e6,
            'operations': {
                'my_readout_op': 'my_readout_pulse',
            },
            "outputs": {
                'out1': ('con1', 1)
            },
            'time_of_flight': 32,
            'smearing': 0
        },

    },

    "pulses": {

        "my_control_pulse": {
            'operation': 'control',
            'length': 100,
            'waveforms': {
                'single': 'my_control_wf',
            }
        },


        'my_readout_pulse': {
            'operation': 'measurement',
            'length': 220,
            'waveforms': {
                'single': 'my_readout_wf',
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
            'sample': 0.4
        },

        'my_readout_wf': {
            'type': 'constant',
            'sample': 0.25
        },

    },

    'digital_waveforms': {
        'ON': {
            'samples': [(1, 0)]
        }
    },

    'integration_weights': {

        'my_integW1': {
            'cosine': [1.0] * int(240 / 4),
            'sine': [0.0] * int(240 / 4)
        },

        'my_integW2': {
            'cosine': [0.0] * int(240 / 4),
            'sine': [1.0] * int(240 / 4)
        },
    },
}

qmm = QuantumMachinesManager()

qm = qmm.open_qm(config)

with program() as my_prog:

    play("my_control_op", "my_element1")
    # update_frequency("my_element1", 100e6)
    play("my_control_op"*amp(0.5), "my_element1")

    # i = declare(int)
    # a = declare(fixed)
    # I = declare(fixed)

    # with for_(i, 0, i < 3, i+1):
    #     with for_(a, 0.1, a < 1.0, a + 0.2):
    #         play("my_control_op"*amp(a), "my_element1")
    #         align("my_element1", "my_element2")
    #         measure("my_readout_op"*amp(1.0-a), "my_element2", "adc", demod.full("integW1", I))
    #         save(I, "I")

res = qm.execute(my_prog)
samps = res.get_simulated_samples()
samps.con1.plot()
plt.show()

