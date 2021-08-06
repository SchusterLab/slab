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

        'my_element3': {
            'singleInput': {
                'port': ('con1', 1),
            },
            'intermediate_frequency': 100e6,
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
            'digitalInputs': {
                'lo': {
                    'port': ('con1', 1),
                    'buffer': 20,
                    'delay': 100
                },
                'switch': {
                    'port': ('con1', 2),
                    'buffer': 20,
                    'delay': 200
                },
            },
            'time_of_flight': 180+32-24, # ns should be a multiple of 4
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
            'samples': [(1,0)]
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

    adc_st = declare_stream(adc_trace=True)
    n = declare(int)

    with for_(n, 0, n<50, n+1):
        measure('my_readout_op'*amp(0.01), "my_element2", adc_st)


    with stream_processing():

        adc_st.input1().average().save("some_name")


res = qm.execute(my_prog)
result_handle = res.result_handles
some_name_handle = result_handle.get('some_name')
some_name_handle.wait_for_all_values()
y = some_name_handle.fetch_all()
plt.plot(y/2**12 + 0.04)
plt.show()


