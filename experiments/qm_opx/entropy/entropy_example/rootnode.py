from quaentropy.api.execution import EntropyContext
from quaentropy.graph_experiment import pynode
from entropyext_cal import *
from scipy.signal.windows import gaussian as gaussian
import numpy as np
from qpu_resolver import resolve
from configuration_10q import config

def drag_pulse(N, alpha, anharomonicity):
    t = np.linspace(0, N - 1, N)
    sig_i = (0.5 - 2 ** -16) * 0.5 * (1 - np.cos(2 * np.pi * t / (N - 1)))
    sig_q = -(0.5 - 2 ** -16) * 0.5 * - (alpha / (anharomonicity * (N - 1))) * np.sin(2 * np.pi * t / (N - 1))
    return sig_i, sig_q


@pynode("root", output_vars={'config'})
def root(context: EntropyContext):
    qpu_db = context.get_resource('qpu_db1')
    
    for con in config["controllers"].keys():
        qpu_db.set(con, 'adc1_offset', config["controllers"][con]["analog_inputs"][1]["offset"])
        qpu_db.set(con, 'adc2_offset', config["controllers"][con]["analog_inputs"][2]["offset"])
    for i in range(10):
        rr = resolve.res(i + 1)
        qpu_db.set(rr, 'time_of_flight', config["elements"][rr]["time_of_flight"])
        qpu_db.set(rr, 'f_lo', config["elements"][rr]["mixInputs"]['lo_frequency'])

    qpu_db.commit()
    return {'config': QuaConfig(config)}
    
    # qpu_db = context.get_resource('qpu_db')
    # qb_IF = [qpu_db.q(i + 1).f_intermediate.value for i in range(qpu_db.num_qubits)]
    # rr_IF = [qpu_db.res(i + 1).f_intermediate.value for i in range(qpu_db.num_qubits)]
    # qb_LO = [qpu_db.q(i + 1).f_lo.value for i in range(qpu_db.num_qubits)]
    # rl_LO = [qpu_db.res(i + 1).f_lo.value for i in range(qpu_db.num_qubits)]
    # time_of_flight = [qpu_db.res(i + 1).time_of_flight.value for i in range(qpu_db.num_qubits)]
    # rr_len_clk = 40
    # return { "config": QuaConfig({
    #                 "version": 1,
    #                 "controllers": {
    #                     "con1": {
    #                         "type": "opx1",
    #                         "analog_outputs": {
    #                             1: {"offset": 0.0},  # q1_I
    #                             2: {"offset": 0.0},  # q1_Q
    #                             3: {"offset": 0.0},  # q2_I
    #                             4: {"offset": 0.0},  # q2_Q
    #                             5: {"offset": 0.0},  # q3_I
    #                             6: {"offset": 0.0},  # q3_Q
    #                             7: {"offset": 0.0},  # q4_I
    #                             8: {"offset": 0.0},  # q4_Q
    #                             9: {"offset": 0.0},  # RR14_I
    #                             10: {"offset": 0.0},  # RR14_Q
    #                         },
    #                         "digital_outputs": {},
    #                         "analog_inputs": {
    #                             1: {"offset": 0.0},  # RL14_I
    #                             2: {"offset": 0.0},  # RL14_Q
    #                         },
    #                     },
    #                     "con2": {
    #                         "type": "opx1",
    #                         "analog_outputs": {
    #                             1: {"offset": 0.0},
    #                             2: {"offset": 0.0},
    #                             3: {"offset": 0.0},
    #                             4: {"offset": 0.0},
    #                             5: {"offset": 0.0},
    #                             6: {"offset": 0.0},
    #                             7: {"offset": 0.0},
    #                             8: {"offset": 0.0},
    #                             9: {"offset": 0.0},
    #                             10: {"offset": 0.0},
    #                         },
    #                         "digital_outputs": {},
    #                         "analog_inputs": {
    #                             1: {"offset": 0.0},
    #                             2: {"offset": 0.0},
    #                         },
    #                     },
    #                     "con3": {
    #                         "type": "opx1",
    #                         "analog_outputs": {
    #                             1: {"offset": 0.0},
    #                             2: {"offset": 0.0},
    #                             3: {"offset": 0.0},
    #                             4: {"offset": 0.0},
    #                             5: {"offset": 0.0},
    #                             6: {"offset": 0.0},
    #                             7: {"offset": 0.0},
    #                             8: {"offset": 0.0},
    #                         },
    #                         "digital_outputs": {},
    #                         "analog_inputs": {},
    #                     },
    #                     "con4": {
    #                         "type": "opx1",
    #                         "analog_outputs": {
    #                             1: {"offset": 0.0},
    #                             2: {"offset": 0.0},
    #                             3: {"offset": 0.0},
    #                             4: {"offset": 0.0},
    #                             5: {"offset": 0.0},
    #                             6: {"offset": 0.0},
    #                             7: {"offset": 0.0},
    #                             8: {"offset": 0.0},
    #                             9: {"offset": 0.0},
    #                             10: {"offset": 0.0},
    #                         },
    #                         "digital_outputs": {},
    #                         "analog_inputs": {},
    #                     },
    #                 },
    #                 "elements": {
    #                     **{
    #                         params[0]: {
    #                             "mixInputs": {
    #                                 "I": (params[1], params[2]),
    #                                 "Q": (params[1], params[3]),
    #                                 "lo_frequency": params[4],
    #                                 "mixer": params[5],
    #                             },
    #                             "intermediate_frequency": params[6],
    #                             "digitalInputs": {},
    #                             "operations": {
    #                                 "const": "const_pulse_IQ",
    #                                 "gaussian": "gaussian_pulse_IQ",
    #                                 "pi": params[7],
    #                                 "pi2": params[8],
    #                             },
    #                         }
    #                         for params in [
    #                             (
    #                                 resolve.q(1, "xy"),
    #                                 "con1",
    #                                 1,
    #                                 2,
    #                                 qb_LO[0],
    #                                 "mixer_q1",
    #                                 qb_IF[0],
    #                                 "pi_pulse1",
    #                                 "pi2_pulse1",
    #                             ),
    #                             (
    #                                 resolve.q(2, "xy"),
    #                                 "con1",
    #                                 3,
    #                                 4,
    #                                 qb_LO[1],
    #                                 "mixer_q2",
    #                                 qb_IF[1],
    #                                 "pi_pulse2",
    #                                 "pi2_pulse2",
    #                             ),
    #                             (
    #                                 resolve.q(3, "xy"),
    #                                 "con1",
    #                                 5,
    #                                 6,
    #                                 qb_LO[2],
    #                                 "mixer_q3",
    #                                 qb_IF[2],
    #                                 "pi_pulse3",
    #                                 "pi2_pulse3",
    #                             ),
    #                             (
    #                                 resolve.q(4, "xy"),
    #                                 "con1",
    #                                 7,
    #                                 8,
    #                                 qb_LO[3],
    #                                 "mixer_q4",
    #                                 qb_IF[3],
    #                                 "pi_pulse4",
    #                                 "pi2_pulse4",
    #                             ),
    #                             (
    #                                 resolve.q(5, "xy"),
    #                                 "con2",
    #                                 1,
    #                                 2,
    #                                 qb_LO[4],
    #                                 "mixer_q5",
    #                                 qb_IF[4],
    #                                 "pi_pulse5",
    #                                 "pi2_pulse5",
    #                             ),
    #                             (
    #                                 resolve.q(6, "xy"),
    #                                 "con2",
    #                                 3,
    #                                 4,
    #                                 qb_LO[5],
    #                                 "mixer_q6",
    #                                 qb_IF[5],
    #                                 "pi_pulse6",
    #                                 "pi2_pulse6",
    #                             ),
    #                             (
    #                                 resolve.q(7, "xy"),
    #                                 "con2",
    #                                 5,
    #                                 6,
    #                                 qb_LO[6],
    #                                 "mixer_q7",
    #                                 qb_IF[6],
    #                                 "pi_pulse7",
    #                                 "pi2_pulse7",
    #                             ),
    #                             (
    #                                 resolve.q(8, "xy"),
    #                                 "con2",
    #                                 7,
    #                                 8,
    #                                 qb_LO[7],
    #                                 "mixer_q8",
    #                                 qb_IF[7],
    #                                 "pi_pulse8",
    #                                 "pi2_pulse8",
    #                             ),
    #                         ]
    #                     },
    #                     **{
    #                         params[0]: {
    #                             "mixInputs": {
    #                                 "I": (params[1], params[2]),
    #                                 "Q": (params[1], params[3]),
    #                                 "lo_frequency": params[4],
    #                                 "mixer": params[5],
    #                             },
    #                             "intermediate_frequency": params[6],
    #                             "outputs": {
    #                                 "out1": (params[1], 1),
    #                                 "out2": (params[1], 2),
    #                             },
    #                             "time_of_flight": params[7],
    #                             "smearing": params[8],
    #                             "operations": {
    #                                 "const": "const_pulse_IQ",
    #                                 "readout": params[9],
    #                             },
    #                         }
    #                         for params in [
    #                             (
    #                                 resolve.res(1),
    #                                 "con1",
    #                                 9,
    #                                 10,
    #                                 rl_LO[0],
    #                                 "mixer_rl1",
    #                                 rr_IF[0],
    #                                 time_of_flight[0],
    #                                 0,
    #                                 "ro_pulse1",
    #                             ),
    #                             (
    #                                 resolve.res(2),
    #                                 "con1",
    #                                 9,
    #                                 10,
    #                                 rl_LO[1],
    #                                 "mixer_rl1",
    #                                 rr_IF[1],
    #                                 time_of_flight[1],
    #                                 0,
    #                                 "ro_pulse2",
    #                             ),
    #                             (
    #                                 resolve.res(3),
    #                                 "con1",
    #                                 9,
    #                                 10,
    #                                 rl_LO[2],
    #                                 "mixer_rl1",
    #                                 rr_IF[2],
    #                                 time_of_flight[2],
    #                                 0,
    #                                 "ro_pulse3",
    #                             ),
    #                             (
    #                                 resolve.res(4),
    #                                 "con1",
    #                                 9,
    #                                 10,
    #                                 rl_LO[3],
    #                                 "mixer_rl1",
    #                                 rr_IF[3],
    #                                 time_of_flight[3],
    #                                 0,
    #                                 "ro_pulse4",
    #                             ),
    #                             (
    #                                 resolve.res(5),
    #                                 "con2",
    #                                 9,
    #                                 10,
    #                                 rl_LO[4],
    #                                 "mixer_rl2",
    #                                 rr_IF[4],
    #                                 time_of_flight[4],
    #                                 0,
    #                                 "ro_pulse5",
    #                             ),
    #                             (
    #                                 resolve.res(6),
    #                                 "con2",
    #                                 9,
    #                                 10,
    #                                 rl_LO[5],
    #                                 "mixer_rl2",
    #                                 rr_IF[5],
    #                                 time_of_flight[5],
    #                                 0,
    #                                 "ro_pulse6",
    #                             ),
    #                             (
    #                                 resolve.res(7),
    #                                 "con2",
    #                                 9,
    #                                 10,
    #                                 rl_LO[6],
    #                                 "mixer_rl2",
    #                                 rr_IF[6],
    #                                 time_of_flight[6],
    #                                 0,
    #                                 "ro_pulse7",
    #                             ),
    #                             (
    #                                 resolve.res(8),
    #                                 "con2",
    #                                 9,
    #                                 10,
    #                                 rl_LO[7],
    #                                 "mixer_rl2",
    #                                 rr_IF[7],
    #                                 time_of_flight[7],
    #                                 0,
    #                                 "ro_pulse8",
    #                             ),
    #                         ]
    #                     },
    #                     **{
    #                         params[0]: {
    #                             "singleInput": {
    #                                 "port": (params[1], params[2]),
    #                             },
    #                             "digitalInputs": {},
    #                             # 'hold_offset': {'duration': 100},
    #                             "operations": {
    #                                 "flux": params[3],
    #                             },
    #                         }
    #                         for params in [
    #                             (resolve.q(1, "z"), "con3", 1, "flux1"),
    #                             (resolve.q(2, "z"), "con3", 2, "flux2"),
    #                             (resolve.q(3, "z"), "con3", 3, "flux3"),
    #                             (resolve.q(4, "z"), "con3", 4, "flux4"),
    #                             (resolve.q(5, "z"), "con3", 5, "flux5"),
    #                             (resolve.q(6, "z"), "con3", 6, "flux6"),
    #                             (resolve.q(7, "z"), "con3", 7, "flux7"),
    #                             (resolve.q(8, "z"), "con3", 8, "flux8"),
    #                         ]
    #                     },
    #                     **{
    #                         params[0]: {
    #                             "singleInput": {
    #                                 "port": (params[1], params[2]),
    #                             },
    #                             "digitalInputs": {},
    #                             # 'hold_offset': {'duration': 100},
    #                             "operations": {
    #                                 "flux": params[3],
    #                             },
    #                         }
    #                         for params in [
    #                             (resolve.coupler(1, 2), "con4", 1, "flux12"),
    #                             (resolve.coupler(2, 3), "con4", 2, "flux23"),
    #                             (resolve.coupler(3, 4), "con4", 3, "flux34"),
    #                             (resolve.coupler(1, 5), "con4", 4, "flux15"),
    #                             (resolve.coupler(5, 6), "con4", 5, "flux56"),
    #                             (resolve.coupler(2, 6), "con4", 6, "flux26"),
    #                             (resolve.coupler(6, 7), "con4", 7, "flux67"),
    #                             (resolve.coupler(3, 7), "con4", 8, "flux37"),
    #                             (resolve.coupler(7, 8), "con4", 9, "flux78"),
    #                             (resolve.coupler(4, 8), "con4", 10, "flux48"),
    #                         ]
    #                     },
    #                 },
    #                 "pulses": {
    #                     **{
    #                         params[0]: {
    #                             "operation": "control",
    #                             "length": params[1],
    #                             "waveforms": {
    #                                 "I": params[2],
    #                                 "Q": params[3],
    #                             },
    #                         }
    #                         for params in [
    #                             ("const_pulse_IQ", 100, "const_wf", "zero_wf"),
    #                             ("gaussian_pulse_IQ", 40, "gauss_wf", "zero_wf"),
    #                             ("pi_pulse1", 40, "pi_wf_q1", "zero_wf"),
    #                             ("pi_pulse2", 40, "pi_wf_q2", "zero_wf"),
    #                             ("pi_pulse3", 40, "pi_wf_q3", "zero_wf"),
    #                             ("pi_pulse4", 40, "pi_wf_q4", "zero_wf"),
    #                             ("pi_pulse5", 40, "pi_wf_q5", "zero_wf"),
    #                             ("pi_pulse6", 40, "pi_wf_q6", "zero_wf"),
    #                             ("pi_pulse7", 40, "pi_wf_q7", "zero_wf"),
    #                             ("pi_pulse8", 40, "pi_wf_q8", "zero_wf"),
    #                             ("pi2_pulse1", 40, "pi2_wf_q1", "zero_wf"),
    #                             ("pi2_pulse2", 40, "pi2_wf_q2", "zero_wf"),
    #                             ("pi2_pulse3", 40, "pi2_wf_q3", "zero_wf"),
    #                             ("pi2_pulse4", 40, "pi2_wf_q4", "zero_wf"),
    #                             ("pi2_pulse5", 40, "pi2_wf_q5", "zero_wf"),
    #                             ("pi2_pulse6", 40, "pi2_wf_q6", "zero_wf"),
    #                             ("pi2_pulse7", 40, "pi2_wf_q7", "zero_wf"),
    #                             ("pi2_pulse8", 40, "pi2_wf_q8", "zero_wf"),
    #                         ]
    #                     },
    #                     **{
    #                         params[0]: {
    #                             "operation": "measurement",
    #                             "length": params[1],
    #                             "waveforms": {"I": params[2], "Q": params[3]},
    #                             "integration_weights": {
    #                                 "integW_cos": params[4],
    #                                 "integW_sin": params[5],
    #                             },
    #                             "digital_marker": "ON",
    #                         }
    #                         for params in [
    #                             (
    #                                 "ro_pulse1",
    #                                 rr_len_clk * 4,
    #                                 "ro_wf1",
    #                                 "zero_wf",
    #                                 "integW1_cos",
    #                                 "integW1_sin",
    #                             ),
    #                             (
    #                                 "ro_pulse2",
    #                                 rr_len_clk * 4,
    #                                 "ro_wf2",
    #                                 "zero_wf",
    #                                 "integW2_cos",
    #                                 "integW2_sin",
    #                             ),
    #                             (
    #                                 "ro_pulse3",
    #                                 rr_len_clk * 4,
    #                                 "ro_wf3",
    #                                 "zero_wf",
    #                                 "integW3_cos",
    #                                 "integW3_sin",
    #                             ),
    #                             (
    #                                 "ro_pulse4",
    #                                 rr_len_clk * 4,
    #                                 "ro_wf4",
    #                                 "zero_wf",
    #                                 "integW4_cos",
    #                                 "integW4_sin",
    #                             ),
    #                             (
    #                                 "ro_pulse5",
    #                                 rr_len_clk * 4,
    #                                 "ro_wf5",
    #                                 "zero_wf",
    #                                 "integW5_cos",
    #                                 "integW5_sin",
    #                             ),
    #                             (
    #                                 "ro_pulse6",
    #                                 rr_len_clk * 4,
    #                                 "ro_wf6",
    #                                 "zero_wf",
    #                                 "integW6_cos",
    #                                 "integW6_sin",
    #                             ),
    #                             (
    #                                 "ro_pulse7",
    #                                 rr_len_clk * 4,
    #                                 "ro_wf7",
    #                                 "zero_wf",
    #                                 "integW7_cos",
    #                                 "integW7_sin",
    #                             ),
    #                             (
    #                                 "ro_pulse8",
    #                                 rr_len_clk * 4,
    #                                 "ro_wf8",
    #                                 "zero_wf",
    #                                 "integW8_cos",
    #                                 "integW8_sin",
    #                             ),
    #                         ]
    #                     },
    #                     **{
    #                         params[0]: {
    #                             "operation": "control",
    #                             "length": params[1],
    #                             "waveforms": {
    #                                 "single": params[2],
    #                             },
    #                         }
    #                         for params in [
    #                             ("flux1", 100, "flux_wf1"),
    #                             ("flux2", 100, "flux_wf2"),
    #                             ("flux3", 100, "flux_wf3"),
    #                             ("flux4", 100, "flux_wf4"),
    #                             ("flux5", 100, "flux_wf5"),
    #                             ("flux6", 100, "flux_wf6"),
    #                             ("flux7", 100, "flux_wf7"),
    #                             ("flux8", 100, "flux_wf8"),
    #                         ]
    #                     },
    #                     **{
    #                         params[0]: {
    #                             "operation": "control",
    #                             "length": params[1],
    #                             "waveforms": {
    #                                 "single": params[2],
    #                             },
    #                         }
    #                         for params in [
    #                             ("flux12", 100, "flux_wf12"),
    #                             ("flux23", 100, "flux_wf23"),
    #                             ("flux34", 100, "flux_wf34"),
    #                             ("flux15", 100, "flux_wf15"),
    #                             ("flux56", 100, "flux_wf56"),
    #                             ("flux26", 100, "flux_wf26"),
    #                             ("flux67", 100, "flux_wf67"),
    #                             ("flux37", 100, "flux_wf37"),
    #                             ("flux78", 100, "flux_wf78"),
    #                             ("flux48", 100, "flux_wf48"),
    #                         ]
    #                     },
    #                 },
    #                 "waveforms": {
    #                     **{
    #                         params[0]: {
    #                             "type": "constant",
    #                             "sample": params[1],
    #                         }
    #                         for params in [
    #                             ("zero_wf", 0.1),
    #                             ("const_wf", 0.1),
    #                             ("ro_wf1", 0.1),
    #                             ("ro_wf2", 0.1),
    #                             ("ro_wf3", 0.1),
    #                             ("ro_wf4", 0.1),
    #                             ("ro_wf5", 0.1),
    #                             ("ro_wf6", 0.1),
    #                             ("ro_wf7", 0.1),
    #                             ("ro_wf8", 0.1),
    #                             ("flux_wf1", 0.1),
    #                             ("flux_wf2", 0.1),
    #                             ("flux_wf3", 0.1),
    #                             ("flux_wf4", 0.1),
    #                             ("flux_wf5", 0.1),
    #                             ("flux_wf6", 0.1),
    #                             ("flux_wf7", 0.1),
    #                             ("flux_wf8", 0.1),
    #                             ("flux_wf12", 0.1),
    #                             ("flux_wf23", 0.1),
    #                             ("flux_wf34", 0.1),
    #                             ("flux_wf15", 0.1),
    #                             ("flux_wf56", 0.1),
    #                             ("flux_wf26", 0.1),
    #                             ("flux_wf67", 0.1),
    #                             ("flux_wf37", 0.1),
    #                             ("flux_wf78", 0.1),
    #                             ("flux_wf48", 0.1),
    #                         ]
    #                     },
    #                     **{
    #                         params[0]: {
    #                             "type": "arbitrary",
    #                             "samples": (
    #                                     params[1] * gaussian(params[2], params[3])
    #                             ).tolist(),
    #                         }
    #                         for params in [
    #                             ("gauss_wf", 0.3, 40, 8),
    #                             ("pi_wf_q1", 0.3, 40, 8),
    #                             ("pi_wf_q2", 0.3, 40, 8),
    #                             ("pi_wf_q3", 0.3, 40, 8),
    #                             ("pi_wf_q4", 0.3, 40, 8),
    #                             ("pi_wf_q5", 0.3, 40, 8),
    #                             ("pi_wf_q6", 0.3, 40, 8),
    #                             ("pi_wf_q7", 0.3, 40, 8),
    #                             ("pi_wf_q8", 0.3, 40, 8),
    #                             ("pi2_wf_q1", 0.15, 40, 8),
    #                             ("pi2_wf_q2", 0.15, 40, 8),
    #                             ("pi2_wf_q3", 0.15, 40, 8),
    #                             ("pi2_wf_q4", 0.15, 40, 8),
    #                             ("pi2_wf_q5", 0.15, 40, 8),
    #                             ("pi2_wf_q6", 0.15, 40, 8),
    #                             ("pi2_wf_q7", 0.15, 40, 8),
    #                             ("pi2_wf_q8", 0.15, 40, 8),
    #                         ]
    #                     },
    #                 },
    #                 "digital_waveforms": {
    #                     "ON": {"samples": [(1, 0)]},
    #                 },
    #                 "integration_weights": {
    #                     **{
    #                         params[0]: {
    #                             "cosine": params[1],
    #                             "sine": params[2],
    #                         }
    #                         for params in [
    #                             ("integW1_cos", [1.0] * rr_len_clk, [0.0] * rr_len_clk),
    #                             ("integW1_sin", [0.0] * rr_len_clk, [1.0] * rr_len_clk),
    #                             ("integW2_cos", [1.0] * rr_len_clk, [0.0] * rr_len_clk),
    #                             ("integW2_sin", [0.0] * rr_len_clk, [1.0] * rr_len_clk),
    #                             ("integW3_cos", [1.0] * rr_len_clk, [0.0] * rr_len_clk),
    #                             ("integW3_sin", [0.0] * rr_len_clk, [1.0] * rr_len_clk),
    #                             ("integW4_cos", [1.0] * rr_len_clk, [0.0] * rr_len_clk),
    #                             ("integW4_sin", [0.0] * rr_len_clk, [1.0] * rr_len_clk),
    #                             ("integW5_cos", [1.0] * rr_len_clk, [0.0] * rr_len_clk),
    #                             ("integW5_sin", [0.0] * rr_len_clk, [1.0] * rr_len_clk),
    #                             ("integW6_cos", [1.0] * rr_len_clk, [0.0] * rr_len_clk),
    #                             ("integW6_sin", [0.0] * rr_len_clk, [1.0] * rr_len_clk),
    #                             ("integW7_cos", [1.0] * rr_len_clk, [0.0] * rr_len_clk),
    #                             ("integW7_sin", [0.0] * rr_len_clk, [1.0] * rr_len_clk),
    #                             ("integW8_cos", [1.0] * rr_len_clk, [0.0] * rr_len_clk),
    #                             ("integW8_sin", [0.0] * rr_len_clk, [1.0] * rr_len_clk),
    #                         ]
    #                     }
    #                 },
    #                 "mixers": {
    #                     **{
    #                         params[0]: [
    #                             {
    #                                 "intermediate_frequency": params[1],
    #                                 "lo_frequency": params[2],
    #                                 "correction": params[3],
    #                             }
    #                         ]
    #                         for params in [
    #                             ("mixer_q1", qb_IF[0], qb_LO[0], [1, 0, 0, 1]),
    #                             ("mixer_q2", qb_IF[1], qb_LO[1], [1, 0, 0, 1]),
    #                             ("mixer_q3", qb_IF[2], qb_LO[2], [1, 0, 0, 1]),
    #                             ("mixer_q4", qb_IF[3], qb_LO[3], [1, 0, 0, 1]),
    #                             ("mixer_q5", qb_IF[4], qb_LO[4], [1, 0, 0, 1]),
    #                             ("mixer_q6", qb_IF[5], qb_LO[5], [1, 0, 0, 1]),
    #                             ("mixer_q7", qb_IF[6], qb_LO[6], [1, 0, 0, 1]),
    #                             ("mixer_q8", qb_IF[7], qb_LO[7], [1, 0, 0, 1]),
    #                         ]
    #                     },
    #                     **{
    #                         params[0]: [
    #                             {
    #                                 "intermediate_frequency": params[1],
    #                                 "lo_frequency": params[2],
    #                                 "correction": params[3],
    #                             },
    #                             {
    #                                 "intermediate_frequency": params[4],
    #                                 "lo_frequency": params[2],
    #                                 "correction": params[5],
    #                             },
    #                             {
    #                                 "intermediate_frequency": params[6],
    #                                 "lo_frequency": params[2],
    #                                 "correction": params[7],
    #                             },
    #                             {
    #                                 "intermediate_frequency": params[8],
    #                                 "lo_frequency": params[2],
    #                                 "correction": params[9],
    #                             },
    #                         ]
    #                         for params in [
    #                             (
    #                                 "mixer_rl1",
    #                                 rr_IF[0],
    #                                 rl_LO[0],
    #                                 [1, 0, 0, 1],
    #                                 rr_IF[1],
    #                                 [1, 0, 0, 1],
    #                                 rr_IF[2],
    #                                 [1, 0, 0, 1],
    #                                 rr_IF[3],
    #                                 [1, 0, 0, 1],
    #                             ),
    #                             (
    #                                 "mixer_rl2",
    #                                 rr_IF[4],
    #                                 rl_LO[1],
    #                                 [1, 0, 0, 1],
    #                                 rr_IF[5],
    #                                 [1, 0, 0, 1],
    #                                 rr_IF[6],
    #                                 [1, 0, 0, 1],
    #                                 rr_IF[7],
    #                                 [1, 0, 0, 1],
    #                             ),
    #                         ]
    #                     },
    #                 },
    #             })}
