from configuration_IQ import config, ge_IF, ef_IF, qubit_LO
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np
from slab import*
from slab.instruments import instrumentmanager
im = InstrumentManager()
LO = im['RF5']
spec = im['SA']

nu_q = 4.748488058822229e9
nu_IF = 100e6
nu_LO = nu_q - 100e6
LO.set_frequency(nu_LO)
LO.set_power(16)
LO.set_output(True)
LO.set_ext_pulse(mod=False)

#############
# Functions #
#############

def get_amp():
    time.sleep(5)
    tr = spec.take_one()
    freq, amp = tr[0], tr[1]
    max_signal_power = max(amp)
    return max_signal_power

with program() as mixer_cal:

    with infinite_loop_():
        play("CW"*amp(1.0), "qubit_ef")

def leakageMap():
    i_min = 0.00
    i_max = 0.02
    di = 0.002

    q_min = -0.10
    q_max = -0.04
    dq = 0.006

    offI = np.arange(i_min, i_max + di/2, di)
    offQ = np.arange(q_min, q_max + dq/2, dq)
    total_pts = len(offI)*len(offQ)
    amps = []
    count = 0
    x = 0
    for i in offI[x:]:
        for j in offQ[x:]:
            qm.set_dc_offset_by_qe("qubit", "I", i)
            qm.set_dc_offset_by_qe("qubit", "Q", j)
            amp_ = get_amp()
            amps.append(amp_)
            count += 1
            print(" %.f out of %.f"%(count, total_pts))
            print("amp = {}, offI = {}, offQ = {}".format(amp_, i, j))

    return offI[x:], offQ[x:], amps


def gradLeakage(offI_, offQ_):

    eps = 0.0002
    qm.set_dc_offset_by_qe("qubit", "I", offI_+eps)
    qm.set_dc_offset_by_qe("qubit", "Q", offQ_)
    a1 = get_amp()
    qm.set_dc_offset_by_qe("qubit", "I", offI_-eps)
    qm.set_dc_offset_by_qe("qubit", "Q", offQ_)
    a2 = get_amp()
    qm.set_dc_offset_by_qe("qubit", "I", offI_)
    qm.set_dc_offset_by_qe("qubit", "Q", offQ_+eps)
    a3 = get_amp()
    qm.set_dc_offset_by_qe("qubit", "I", offI_)
    qm.set_dc_offset_by_qe("qubit", "Q", offQ_-eps)
    a4 = get_amp()

    gradx = (a2 - a1) / (2 * eps)
    grady = (a4 - a3) / (2 * eps)

    return gradx, grady


def gamma(prev_x, prev_y, curr_x, curr_y, prev_grad_x, prev_grad_y, curr_grad_x, curr_grad_y):

    df = np.array([curr_x, curr_y]) - np.array([prev_x, prev_y])
    d_grad = np.array([curr_grad_x, curr_grad_y]) - np.array([prev_grad_x, prev_grad_y])
    norm = np.sum(d_grad * d_grad)
    g = np.abs(np.sum(df*d_grad) / norm)

    return g


def IQ_imbalance_corr(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g ** 2) * (2 * c ** 2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s,
                                   (1 - g) * s, (1 + g) * c]]


def imbalancesMap():

    dphi = 0.001
    dg = 0.001
    phi = np.arange(0.02, 0.04 + dphi/2, dphi)
    phi = np.pi*(phi)
    g = np.arange(-0.015, -0.005 + dg/2, dg)

    amps = []

    for i in phi:

        for j in g:

            qm.set_mixer_correction("mixer_qubit_ef", int(ef_IF), int(qubit_LO), IQ_imbalance_corr(j, i))
            amp_ = get_amp()
            amps.append(amp_)
            print("amp = {}, phase = {}, gain = {}".format(amp_, i, j))

    return phi, g, amps


def gradImbalances(phase_, gain_):

    eps = 0.0002
    qm.set_mixer_correction("mixer_qubit", int(ge_IF), int(qubit_LO), IQ_imbalance_corr(gain_, phase_ + eps))
    a1 = get_amp()
    qm.set_mixer_correction("mixer_qubit", int(ge_IF), int(qubit_LO), IQ_imbalance_corr(gain_, phase_ - eps))
    a2 = get_amp()
    qm.set_mixer_correction("mixer_qubit", int(ge_IF), int(qubit_LO), IQ_imbalance_corr(gain_ + eps, phase_))
    a3 = get_amp()
    qm.set_mixer_correction("mixer_qubit", int(ge_IF), int(qubit_LO), IQ_imbalance_corr(gain_ - eps, phase_))
    a4 = get_amp()

    gradx = (a2 - a1) / (2 * eps)
    grady = (a4 - a3) / (2 * eps)

    return gradx, grady


def grad_descent(x0, y0, peak):

    """
    :param x0: starting point in the x axis
    :param y0: starting point in the y axis
    :param peak: "LO" or "LSB"
    :return:
    """

    # initializations:
    x_track = [x0]
    y_track = [y0]
    curr_x = x0
    curr_y = y0

    if peak == "LO":
        curr_grad_x, curr_grad_y = gradLeakage(x0, y0)
    elif peak == "LSB":
        curr_grad_x, curr_grad_y = gradImbalances(x0, y0)
    else:
        raise ValueError("Unexpected peak value")

    g = 1e-7
    precision = 0.0000003
    prev_step_size_x = precision + 1
    prev_step_size_y = precision + 1

    B = False

    while prev_step_size_x > precision or prev_step_size_y > precision or B == False:

        B = True
        prev_x = curr_x
        prev_y = curr_y

        prev_grad_x = curr_grad_x
        prev_grad_y = curr_grad_y

        curr_x += -g * curr_grad_x
        curr_y += -g * curr_grad_y

        if peak == "LO":
            curr_grad_x, curr_grad_y = gradLeakage(curr_x, curr_y)
        elif peak == "LSB":
            curr_grad_x, curr_grad_y = gradImbalances(curr_x, curr_y)

        g = gamma(prev_x, prev_y, curr_x, curr_y, prev_grad_x, prev_grad_y, curr_grad_x, curr_grad_y)

        prev_step_size_x = abs(curr_x - prev_x)
        prev_step_size_y = abs(curr_y - prev_y)

        x_track.append(curr_x)
        y_track.append(curr_y)

    xf = curr_x
    yf = curr_y

    return xf, yf, x_track, y_track


##########
# Execute:
##########

# Execute the mixer_cal program:
qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
job = qm.execute(mixer_cal)

# Configure the SA :
# delta_F = 10e6
# spec.set_center_frequency(qubit_LO)
# spec.set_span(delta_F)
# spec.set_resbw(100e3)
# time.sleep(2)

# LO leakage 2D map
# offI, offQ, amps = leakageMap()
# plt.figure()
# offI_grid, offQ_grid = np.meshgrid(offI, offQ)
# amps_grid = np.transpose(np.reshape(amps, [len(offI), len(offQ)]))
# plt.pcolormesh(offI_grid, offQ_grid, amps_grid)
# plt.colorbar()
# plt.xlabel("I offset")
# plt.ylabel("Q offset")
# plt.tight_layout()
# plt.show()
#
# # Gradient descent for the LO peak
# indices = np.where(amps_grid == np.min(amps_grid))
# offI0 = offI_grid[indices[0][0]][indices[1][0]]
# offQ0 = offQ_grid[indices[0][0]][indices[1][0]]
# print("OffI_min = {}, offQ_min = {}".format(offI0, offQ0))
# offIf, offQf, offI_track, offQ_track = grad_descent(offI0, offQ0, "LO")
#
# Configure the SA:
delta_F = 10e6
spec.set_center_frequency(qubit_LO-ef_IF)
spec.set_span(delta_F)
spec.set_resbw(100e3)
time.sleep(2)
# #
# IQ imbalances 2D map
phase, gain, amps = imbalancesMap()
plt.figure()
phase_grid, gain_grid = np.meshgrid(phase, gain)
amps_grid = np.transpose(np.reshape(amps, [len(phase), len(gain)]))
plt.pcolormesh(phase_grid, gain_grid, amps_grid)
plt.colorbar()
plt.xlabel("phase")
plt.ylabel("gain")
# #
# Gradient descent for the SSB peak
indices = np.where(amps_grid == np.min(amps_grid))
phase0 = phase_grid[indices[0][0]][indices[1][0]]
gain0 = gain_grid[indices[0][0]][indices[1][0]]
# print("phase_min = {}, gain_min = {}".format(phase0, gain0))
# phasef, gainf, phase_track, ogain_track = grad_descent(phase0, gain0, "LSB")


