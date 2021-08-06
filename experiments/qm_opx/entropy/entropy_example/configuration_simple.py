import numpy as np
from scipy.signal.windows import gaussian as gaussian



# todo: for testing
def simulate_pulse(chi, k, Ts, Td, power):
    I = [0]
    Q = [0]
    # solve numerically a simplified version of the readout resonator
    for t in range(Ts):
        I.append(I[-1] + (power / 2 - k * I[-1] + Q[-1] * chi))
        Q.append(Q[-1] + (power / 2 - k * Q[-1] - I[-1] * chi))

    for t in range(Td - 1):
        I.append(I[-1] + (-k * I[-1] + Q[-1] * chi))
        Q.append(Q[-1] + (-k * Q[-1] - I[-1] * chi))

    I = np.array(I)
    Q = np.array(Q)

    return I, Q


time_of_flight = [284, 172, 180, 188, 180, 164, 160, 156, 164, 160]
# time_of_flight = [160]*9+[180]
# time_of_flight = [184, 184, 184, 184, 184, 184, 184, 184, 184, 184]

# todo: end testing

rr_len_clk = 120

############################
# generate waveforms #
############################
# todo: for testing purposes
rr_num = 10
readout_len = rr_len_clk * 4
Ts = readout_len - 200
Td = 200
power = 0.2
k = 0.04
chi = 0.023 * np.array([-1, 1, 3.5])
num_of_levels = 3
I_, Q_ = [[0]] * num_of_levels, [[0]] * num_of_levels
for i in range(num_of_levels):
    I_[i], Q_[i] = simulate_pulse(chi[i], k, Ts, Td, power)

divide_signal_factor = 60  # scale to get the signal within -0.5,0.5 range
# todo: end testing
#############################

qb_LO = [1e9] * 10
qb_IF = [1e6] * 10
rl_LO = [1e9] * 10
# rr_IF = [1e6] * 10
rr_IF = [51.223e6, 68.635e6, 77.11e6, 86.784e6, 95.533e6, 104.109e6, 110.662e6, 124.235e6, 139.1e6, 151e6]

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0},  # RR_0-4_I
                2: {"offset": 0.0},  # RR_0-4_Q
                3: {"offset": 0.0},  # qubit0_xy_I
                4: {"offset": 0.0},  # qubit0_xy_Q
                5: {"offset": 0.0},  # qubit1_xy_I
                6: {"offset": 0.0},  # qubit1_xy_Q
                7: {"offset": 0.0},  # flux0
                8: {"offset": 0.0},  # flux1
            },
            "digital_outputs": {},
            "analog_inputs": {
                1: {"offset": 0.0},  # RO_I
                2: {"offset": 0.0},  # RO_Q
            },
        },
        "con2": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0},  # RR_5-9_I
                2: {"offset": 0.0},  # RR_5-9_Q
                3: {"offset": 0.0},  # qubit2_xy_I
                4: {"offset": 0.0},  # qubit2_xy_Q
                5: {"offset": 0.0},  # qubit3_xy_I
                6: {"offset": 0.0},  # qubit3_xy_Q
                7: {"offset": 0.0},  # flux2
                8: {"offset": 0.0},  # flux3
            },
            "digital_outputs": {},
            "analog_inputs": {
                1: {"offset": 0.0},  # RO_I
                2: {"offset": 0.0},  # RO_Q
            },
        },
        "con3": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0},  # qubit4_xy_I
                2: {"offset": 0.0},  # qubit4_xy_Q
                3: {"offset": 0.0},  # qubit5_xy_I
                4: {"offset": 0.0},  # qubit5_xy_Q
                5: {"offset": 0.0},  # qubit6_xy_I
                6: {"offset": 0.0},  # qubit6_xy_Q
                7: {"offset": 0.0},  # flux4
                8: {"offset": 0.0},  # flux5
                9: {"offset": 0.0},  # flux6
            },
            "digital_outputs": {},
            "analog_inputs": {

            },
        },
        "con4": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0},  # qubit7_xy_I
                2: {"offset": 0.0},  # qubit7_xy_Q
                3: {"offset": 0.0},  # qubit8_xy_I
                4: {"offset": 0.0},  # qubit8_xy_Q
                5: {"offset": 0.0},  # qubit9_xy_I
                6: {"offset": 0.0},  # qubit9_xy_Q
                7: {"offset": 0.0},  # flux7
                8: {"offset": 0.0},  # flux8
                9: {"offset": 0.0},  # flux9
            },
            "digital_outputs": {},
            "analog_inputs": {

            },
        },
    },
    "elements": {
        **{
            params[0]: {
                "mixInputs": {
                    "I": (params[1], params[2]),
                    "Q": (params[1], params[3]),
                    "lo_frequency": params[4],
                    "mixer": params[5],
                },
                "intermediate_frequency": params[6],
                "digitalInputs": {},
                "operations": {
                    "const": "const_pulse_IQ",
                    "gaussian": "gaussian_pulse_IQ",
                    "pi": params[7],
                    "pi2": params[8],
                },
            }
            for params in [
                ("q0_xy", "con1", 3, 4, qb_LO[0], "mixer_q0", qb_IF[0], "pi_pulse0", "pi2_pulse0"),
                ("q1_xy", "con1", 5, 6, qb_LO[1], "mixer_q1", qb_IF[1], "pi_pulse1", "pi2_pulse1"),
                ("q2_xy", "con2", 3, 4, qb_LO[2], "mixer_q2", qb_IF[2], "pi_pulse2", "pi2_pulse2"),
                ("q3_xy", "con2", 5, 6, qb_LO[3], "mixer_q3", qb_IF[3], "pi_pulse3", "pi2_pulse3"),
                ("q4_xy", "con3", 1, 2, qb_LO[4], "mixer_q4", qb_IF[4], "pi_pulse4", "pi2_pulse4"),
                ("q5_xy", "con3", 3, 4, qb_LO[5], "mixer_q5", qb_IF[5], "pi_pulse5", "pi2_pulse5"),
                ("q6_xy", "con3", 5, 6, qb_LO[6], "mixer_q6", qb_IF[6], "pi_pulse6", "pi2_pulse6"),
                ("q7_xy", "con4", 1, 2, qb_LO[7], "mixer_q7", qb_IF[7], "pi_pulse7", "pi2_pulse7"),
                ("q8_xy", "con4", 3, 4, qb_LO[8], "mixer_q8", qb_IF[8], "pi_pulse8", "pi2_pulse8"),
                ("q9_xy", "con4", 5, 6, qb_LO[9], "mixer_q9", qb_IF[9], "pi_pulse9", "pi2_pulse9")
            ]
        },
        **{
            params[0]: {
                "mixInputs": {
                    "I": (params[1], params[2]),
                    "Q": (params[1], params[3]),
                    "lo_frequency": params[4],
                    "mixer": params[5],
                },
                "intermediate_frequency": params[6],
                "outputs": {
                    "out1": (params[1], 1),
                    "out2": (params[1], 2),
                },
                "time_of_flight": params[7],
                "smearing": params[8],
                "operations": {
                    "const": "const_pulse_IQ",
                    "readout": params[9],
                    # todo: for testing !!!!
                    'test_readout_pulse_0': 'test_readout_pulse_0',
                    'test_readout_pulse_1': 'test_readout_pulse_1',
                    'test_readout_pulse_2': 'test_readout_pulse_2',
                },
            }
            for params in [
                ("rr_0", "con1", 1, 2, rl_LO[0], "mixer_rl0", rr_IF[0], time_of_flight[0], 0, "ro_pulse0"),
                ("rr_1", "con1", 1, 2, rl_LO[1], "mixer_rl0", rr_IF[1], time_of_flight[1], 0, "ro_pulse1"),
                ("rr_2", "con1", 1, 2, rl_LO[2], "mixer_rl0", rr_IF[2], time_of_flight[2], 0, "ro_pulse2"),
                ("rr_3", "con1", 1, 2, rl_LO[3], "mixer_rl0", rr_IF[3], time_of_flight[3], 0, "ro_pulse3"),
                ("rr_4", "con1", 1, 2, rl_LO[4], "mixer_rl0", rr_IF[4], time_of_flight[4], 0, "ro_pulse4"),
                ("rr_5", "con2", 1, 2, rl_LO[5], "mixer_rl1", rr_IF[5], time_of_flight[5], 0, "ro_pulse5"),
                ("rr_6", "con2", 1, 2, rl_LO[6], "mixer_rl1", rr_IF[6], time_of_flight[6], 0, "ro_pulse6"),
                ("rr_7", "con2", 1, 2, rl_LO[7], "mixer_rl1", rr_IF[7], time_of_flight[7], 0, "ro_pulse7"),
                ("rr_8", "con2", 1, 2, rl_LO[8], "mixer_rl1", rr_IF[8], time_of_flight[8], 0, "ro_pulse8"),
                ("rr_9", "con2", 1, 2, rl_LO[9], "mixer_rl1", rr_IF[9], time_of_flight[9], 0, "ro_pulse9"),
            ]
        },
        **{
            params[0]: {
                "singleInput": {
                    "port": (params[1], params[2]),
                },
                "digitalInputs": {},
                # 'hold_offset': {'duration': 100},
                "operations": {
                    "flux": params[3],
                },
            }
            for params in [
                ("flux0", "con1", 7, "flux0"),
                ("flux1", "con1", 8, "flux1"),
                ("flux2", "con2", 7, "flux2"),
                ("flux3", "con2", 8, "flux3"),
                ("flux4", "con3", 7, "flux4"),
                ("flux5", "con3", 8, "flux5"),
                ("flux6", "con3", 9, "flux6"),
                ("flux7", "con4", 7, "flux7"),
                ("flux8", "con4", 8, "flux8"),
                ("flux9", "con4", 9, "flux9"),

            ]
        }
    },
    "pulses": {
        **{
            params[0]: {
                "operation": "control",
                "length": params[1],
                "waveforms": {
                    "I": params[2],
                    "Q": params[3],
                },
            }
            for params in [
                ("const_pulse_IQ", 100, "const_wf", "zero_wf"),
                ("gaussian_pulse_IQ", 40, "gauss_wf", "zero_wf"),
                ("pi_pulse0", 40, "pi_wf_q0", "zero_wf"),
                ("pi_pulse1", 40, "pi_wf_q1", "zero_wf"),
                ("pi_pulse2", 40, "pi_wf_q2", "zero_wf"),
                ("pi_pulse3", 40, "pi_wf_q3", "zero_wf"),
                ("pi_pulse4", 40, "pi_wf_q4", "zero_wf"),
                ("pi_pulse5", 40, "pi_wf_q5", "zero_wf"),
                ("pi_pulse6", 40, "pi_wf_q6", "zero_wf"),
                ("pi_pulse7", 40, "pi_wf_q7", "zero_wf"),
                ("pi_pulse8", 40, "pi_wf_q8", "zero_wf"),
                ("pi_pulse9", 40, "pi_wf_q9", "zero_wf"),

                ("pi2_pulse0", 40, "pi2_wf_q0", "zero_wf"),
                ("pi2_pulse1", 40, "pi2_wf_q1", "zero_wf"),
                ("pi2_pulse2", 40, "pi2_wf_q2", "zero_wf"),
                ("pi2_pulse3", 40, "pi2_wf_q3", "zero_wf"),
                ("pi2_pulse4", 40, "pi2_wf_q4", "zero_wf"),
                ("pi2_pulse5", 40, "pi2_wf_q5", "zero_wf"),
                ("pi2_pulse6", 40, "pi2_wf_q6", "zero_wf"),
                ("pi2_pulse7", 40, "pi2_wf_q7", "zero_wf"),
                ("pi2_pulse8", 40, "pi2_wf_q8", "zero_wf"),
                ("pi2_pulse9", 40, "pi2_wf_q9", "zero_wf"),

            ]
        },
        **{
            params[0]: {
                "operation": "measurement",
                "length": params[1],
                "waveforms": {"I": params[2], "Q": params[3]},
                "integration_weights": {
                    "integW_cos": params[4],
                    "integW_sin": params[5],
                },
                "digital_marker": "ON",
            }
            for params in [
                ("ro_pulse0", rr_len_clk * 4, "ro_wf0", "zero_wf", "integW0_cos", "integW0_sin",),
                ("ro_pulse1", rr_len_clk * 4, "ro_wf1", "zero_wf", "integW1_cos", "integW1_sin",),
                ("ro_pulse2", rr_len_clk * 4, "ro_wf2", "zero_wf", "integW2_cos", "integW2_sin",),
                ("ro_pulse3", rr_len_clk * 4, "ro_wf3", "zero_wf", "integW3_cos", "integW3_sin",),
                ("ro_pulse4", rr_len_clk * 4, "ro_wf4", "zero_wf", "integW4_cos", "integW4_sin",),
                ("ro_pulse5", rr_len_clk * 4, "ro_wf5", "zero_wf", "integW5_cos", "integW5_sin",),
                ("ro_pulse6", rr_len_clk * 4, "ro_wf6", "zero_wf", "integW6_cos", "integW6_sin",),
                ("ro_pulse7", rr_len_clk * 4, "ro_wf7", "zero_wf", "integW7_cos", "integW7_sin",),
                ("ro_pulse8", rr_len_clk * 4, "ro_wf8", "zero_wf", "integW8_cos", "integW8_sin",),
                ("ro_pulse9", rr_len_clk * 4, "ro_wf9", "zero_wf", "integW9_cos", "integW9_sin",),
            ]
        },
        **{  # todo: for testing !!!!!!!!!!!!!!
            params[0]: {
                "operation": "measurement",
                "length": params[1],
                "waveforms": {"I": params[2], "Q": params[3]},
                "integration_weights": {
                },
                "digital_marker": "ON"
            }
            for params in [
                ("test_readout_pulse_0", rr_len_clk * 4, "I_wf_0", "Q_wf_0"),
                ("test_readout_pulse_1", rr_len_clk * 4, "I_wf_1", "Q_wf_1"),
                ("test_readout_pulse_2", rr_len_clk * 4, "I_wf_2", "Q_wf_2"),
            ]
        },
        **{
            params[0]: {
                "operation": "control",
                "length": params[1],
                "waveforms": {
                    "single": params[2],
                },
            }
            for params in [
                ("flux0", 100, "flux_wf0"),
                ("flux1", 100, "flux_wf1"),
                ("flux2", 100, "flux_wf2"),
                ("flux3", 100, "flux_wf3"),
                ("flux4", 100, "flux_wf4"),
                ("flux5", 100, "flux_wf5"),
                ("flux6", 100, "flux_wf6"),
                ("flux7", 100, "flux_wf7"),
                ("flux8", 100, "flux_wf8"),
                ("flux9", 100, "flux_wf9")
            ]
        },
    },
    "waveforms": {
        **{
            params[0]: {
                "type": "constant",
                "sample": params[1],
            }
            for params in [
                ("zero_wf", 0.1),
                ("const_wf", 0.1),
                ("ro_wf0", 0.1),
                ("ro_wf1", 0.1),
                ("ro_wf2", 0.1),
                ("ro_wf3", 0.1),
                ("ro_wf4", 0.1),
                ("ro_wf5", 0.1),
                ("ro_wf6", 0.1),
                ("ro_wf7", 0.1),
                ("ro_wf8", 0.1),
                ("ro_wf9", 0.1),
                ("flux_wf0", 0.1),
                ("flux_wf1", 0.1),
                ("flux_wf2", 0.1),
                ("flux_wf3", 0.1),
                ("flux_wf4", 0.1),
                ("flux_wf5", 0.1),
                ("flux_wf6", 0.1),
                ("flux_wf7", 0.1),
                ("flux_wf8", 0.1),
                ("flux_wf9", 0.1),
            ]
        },
        **{
            params[0]: {
                "type": "arbitrary",
                "samples": (
                        params[1] * gaussian(params[2], params[3])
                ).tolist(),
            }
            for params in [
                ("gauss_wf", 0.3, 40, 8),
                ("pi_wf_q0", 0.3, 40, 8),
                ("pi_wf_q1", 0.3, 40, 8),
                ("pi_wf_q2", 0.3, 40, 8),
                ("pi_wf_q3", 0.3, 40, 8),
                ("pi_wf_q4", 0.3, 40, 8),
                ("pi_wf_q5", 0.3, 40, 8),
                ("pi_wf_q6", 0.3, 40, 8),
                ("pi_wf_q7", 0.3, 40, 8),
                ("pi_wf_q8", 0.3, 40, 8),
                ("pi_wf_q9", 0.3, 40, 8),
                ("pi2_wf_q0", 0.15, 40, 8),
                ("pi2_wf_q1", 0.15, 40, 8),
                ("pi2_wf_q2", 0.15, 40, 8),
                ("pi2_wf_q3", 0.15, 40, 8),
                ("pi2_wf_q4", 0.15, 40, 8),
                ("pi2_wf_q5", 0.15, 40, 8),
                ("pi2_wf_q6", 0.15, 40, 8),
                ("pi2_wf_q7", 0.15, 40, 8),
                ("pi2_wf_q8", 0.15, 40, 8),
                ("pi2_wf_q9", 0.15, 40, 8),

            ]
        },
        **{  # todo: for testing !!!!!!!!!!!!!!!
            params[0]: {
                "type": "arbitrary",
                "samples": params[1],
            }
            for params in [
                ("I_wf_0", [float(arg / divide_signal_factor) for arg in I_[0]]),
                ("Q_wf_0", [float(arg / divide_signal_factor) for arg in Q_[0]]),
                ("I_wf_1", [float(arg / divide_signal_factor) for arg in I_[1]]),
                ("Q_wf_1", [float(arg / divide_signal_factor) for arg in Q_[1]]),
                ("I_wf_2", [float(arg / divide_signal_factor) for arg in I_[2]]),
                ("Q_wf_2", [float(arg / divide_signal_factor) for arg in Q_[2]]),
            ]
        }
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        **{
            params[0]: {
                "cosine": params[1],
                "sine": params[2],
            }
            for params in [
                ("integW0_cos", [1.0] * rr_len_clk, [0.0] * rr_len_clk),
                ("integW0_sin", [0.0] * rr_len_clk, [1.0] * rr_len_clk),
                ("integW1_cos", [1.0] * rr_len_clk, [0.0] * rr_len_clk),
                ("integW1_sin", [0.0] * rr_len_clk, [1.0] * rr_len_clk),
                ("integW2_cos", [1.0] * rr_len_clk, [0.0] * rr_len_clk),
                ("integW2_sin", [0.0] * rr_len_clk, [1.0] * rr_len_clk),
                ("integW3_cos", [1.0] * rr_len_clk, [0.0] * rr_len_clk),
                ("integW3_sin", [0.0] * rr_len_clk, [1.0] * rr_len_clk),
                ("integW4_cos", [1.0] * rr_len_clk, [0.0] * rr_len_clk),
                ("integW4_sin", [0.0] * rr_len_clk, [1.0] * rr_len_clk),
                ("integW5_cos", [1.0] * rr_len_clk, [0.0] * rr_len_clk),
                ("integW5_sin", [0.0] * rr_len_clk, [1.0] * rr_len_clk),
                ("integW6_cos", [1.0] * rr_len_clk, [0.0] * rr_len_clk),
                ("integW6_sin", [0.0] * rr_len_clk, [1.0] * rr_len_clk),
                ("integW7_cos", [1.0] * rr_len_clk, [0.0] * rr_len_clk),
                ("integW7_sin", [0.0] * rr_len_clk, [1.0] * rr_len_clk),
                ("integW8_cos", [1.0] * rr_len_clk, [0.0] * rr_len_clk),
                ("integW8_sin", [0.0] * rr_len_clk, [1.0] * rr_len_clk),
                ("integW9_cos", [1.0] * rr_len_clk, [0.0] * rr_len_clk),
                ("integW9_sin", [0.0] * rr_len_clk, [1.0] * rr_len_clk),

            ]
        }
    },
    "mixers": {
        **{
            params[0]: [
                {
                    "intermediate_frequency": params[1],
                    "lo_frequency": params[2],
                    "correction": params[3],
                }
            ]
            for params in [
                ("mixer_q0", qb_IF[0], qb_LO[0], [1, 0, 0, 1]),
                ("mixer_q1", qb_IF[1], qb_LO[1], [1, 0, 0, 1]),
                ("mixer_q2", qb_IF[2], qb_LO[2], [1, 0, 0, 1]),
                ("mixer_q3", qb_IF[3], qb_LO[3], [1, 0, 0, 1]),
                ("mixer_q4", qb_IF[4], qb_LO[4], [1, 0, 0, 1]),
                ("mixer_q5", qb_IF[5], qb_LO[5], [1, 0, 0, 1]),
                ("mixer_q6", qb_IF[6], qb_LO[6], [1, 0, 0, 1]),
                ("mixer_q7", qb_IF[7], qb_LO[7], [1, 0, 0, 1]),
                ("mixer_q8", qb_IF[8], qb_LO[8], [1, 0, 0, 1]),
                ("mixer_q9", qb_IF[9], qb_LO[9], [1, 0, 0, 1]),
            ]
        },
        **{
            params[0]: [
                {
                    "intermediate_frequency": params[2],
                    "lo_frequency": params[1],
                    "correction": params[3],
                },
                {
                    "intermediate_frequency": params[4],
                    "lo_frequency": params[1],
                    "correction": params[5],
                },
                {
                    "intermediate_frequency": params[6],
                    "lo_frequency": params[1],
                    "correction": params[7],
                },
                {
                    "intermediate_frequency": params[8],
                    "lo_frequency": params[1],
                    "correction": params[9],
                },
                {
                    "intermediate_frequency": params[10],
                    "lo_frequency": params[1],
                    "correction": params[11],
                },
            ]
            for params in [
                ("mixer_rl0", rl_LO[0], rr_IF[0], [1, 0, 0, 1], rr_IF[1], [1, 0, 0, 1], rr_IF[2], [1, 0, 0, 1],
                 rr_IF[3], [1, 0, 0, 1], rr_IF[4], [1, 0, 0, 1]),
                ("mixer_rl1", rl_LO[1], rr_IF[5], [1, 0, 0, 1], rr_IF[6], [1, 0, 0, 1], rr_IF[7], [1, 0, 0, 1],
                 rr_IF[8], [1, 0, 0, 1], rr_IF[9], [1, 0, 0, 1]),
            ]
        },
    },
}
