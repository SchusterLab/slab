from qpu_resolver import resolve as _resolve

_num_qubits = 10
_num_waveguide_in_lines = 2
_num_waveguide_out_lines = 2

qpu_base_data = {
    _resolve.q(1): {'t1': 3.4, 'f_intermediate': 20e6, 'north_flux': 0.13, 'drag_alpha': 1.0, 'anharmonicity': 0.3,
                    'half_pulse_correction': 0.50, 'f_lo': 5e9
                    },
    _resolve.q(2): {'t1': 3.4, 'f_intermediate': 20e6, 'north_flux': 0.13, 'drag_alpha': 1.0, 'anharmonicity': 0.3,
                    'half_pulse_correction': 0.50, 'f_lo': 5e9
                    },
    _resolve.q(3): {'t1': 3.4, 'f_intermediate': 20e6, 'north_flux': 0.13, 'drag_alpha': 1.0, 'anharmonicity': 0.3,
                    'half_pulse_correction': 0.50, 'f_lo': 5e9
                    },
    _resolve.q(4): {'t1': 3.4, 'f_intermediate': 20e6, 'north_flux': 0.13, 'drag_alpha': 1.0, 'anharmonicity': 0.3,
                    'half_pulse_correction': 0.50, 'f_lo': 5e9
                    },
    _resolve.q(5): {'t1': 3.4, 'f_intermediate': 20e6, 'north_flux': 0.13, 'drag_alpha': 1.0, 'anharmonicity': 0.3,
                    'half_pulse_correction': 0.50, 'f_lo': 5e9
                    },
    _resolve.q(6): {'t1': 3.4, 'f_intermediate': 20e6, 'north_flux': 0.13, 'drag_alpha': 1.0, 'anharmonicity': 0.3,
                    'half_pulse_correction': 0.50, 'f_lo': 5e9
                    },
    _resolve.q(7): {'t1': 3.4, 'f_intermediate': 20e6, 'north_flux': 0.13, 'drag_alpha': 1.0, 'anharmonicity': 0.3,
                    'half_pulse_correction': 0.50, 'f_lo': 5e9
                    },
    _resolve.q(8): {'t1': 3.4, 'f_intermediate': 20e6, 'north_flux': 0.13, 'drag_alpha': 1.0, 'anharmonicity': 0.3,
                    'half_pulse_correction': 0.50, 'f_lo': 5e9
                    },
    _resolve.q(9): {'t1': 3.4, 'f_intermediate': 20e6, 'north_flux': 0.13, 'drag_alpha': 1.0, 'anharmonicity': 0.3,
                    'half_pulse_correction': 0.50, 'f_lo': 5e9
                    },
    _resolve.q(10): {'t1': 3.4, 'f_intermediate': 20e6, 'north_flux': 0.13, 'drag_alpha': 1.0, 'anharmonicity': 0.3,
                    'half_pulse_correction': 0.50, 'f_lo': 5e9
                    },

    **{
        _resolve.res(i+1): {
            'threshold': -0.1,
            'f_intermediate': i * 20e6 + 25e6,
            'time_of_flight': 180,
            'f_lo': 7e9,
            'readout_amp': 0.3,
            'readout_pulse_len': 200
        }
        for i in range(_num_qubits)
    },

    'system':
        {'single_pulse_duration': 20, 'num_qubits': 10},

    'con1': {'adc1_offset': 0.0, 'adc2_offset': 0.0},
    'con2': {'adc1_offset': 0.0, 'adc2_offset': 0.0},
    'con3': {'adc1_offset': 0.0, 'adc2_offset': 0.0},
    'con4': {'adc1_offset': 0.0, 'adc2_offset': 0.0}



}

