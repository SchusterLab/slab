from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
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

        'element1': {
            'digitalInputs': {
                "switch1": {
                    "port": ("con1", 1),    # corresponding to DO2 of the OPX
                    "delay": 0,
                    "buffer": 0,
                },
            },
            'operations': {
                'my_trig_op': 'my_trig_pulse',
                'arb_dig_op': 'arb_dig_pulse'
            },
        },

        'element2': {
            'digitalInputs': {
                "switch1": {
                    "port": ("con1", 2),    # corresponding to DO2 of the OPX
                    "delay": 100,    # will be delayed by 100ns wrt the AO
                    "buffer": 12,     # will be smeared symmetrically to each direction by 12ns
                },
            },
            'singleInput': {
                'port': ('con1', 1)      # corresponding to AO1 of the OPX
            },
            'intermediate_frequency': 25e6,
            'operations': {
                'my_readout_op': 'my_readout_pulse',
            },
            "outputs": {
                'out1': ('con1', 1)      # corresponding to AI1 of the OPX
            },
            'time_of_flight': 32,
            'smearing': 0
        },

    },

    "pulses": {

        "my_trig_pulse": {
            'operation': 'control',
            'length': 100,
            'digital_marker': 'ON'
        },

        "arb_dig_pulse": {
            'operation': 'control',
            'length': 200,
            'digital_marker': 'arb_dig_wf'
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

        },
        'arb_dig_wf': {
            'samples': [(1, 3), (0, 3), (1, 3), (0, 3), (1, 6), (0, 6), (1, 12), (0, 12), (1,0)]
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

qmm = QuantumMachinesManager(host="18.156.117.64")

qm = qmm.open_qm(config)

with program() as my_prog:

    I = declare(fixed)

    play("my_trig_op", "element1")
    align("element1", "element2")
    measure("my_readout_op", "element2", None, demod.full("integW1", I))
    align("element1", "element2")
    wait(100, "element1")  # units of clock cycles (4ns)
    play("arb_dig_op", "element1")
    save(I, "I")

res = qm.simulate(my_prog, SimulationConfig(400))
samps = res.get_simulated_samples()
samps.con1.plot()
plt.show()