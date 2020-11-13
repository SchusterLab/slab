from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import time
import matplotlib.pyplot as plt
from lab_automation import automation
from configuration import config, qubit_IF, qubit_LO
import numpy as np
import sys

if "c:\\git\\opx\\lab\\automation" not in sys.path:
    sys.path.append("c:\\git\\opx\\lab\\automation")


#############
# Functions #
#############

def get_amp():
    time.sleep(0.05)
    data = spec.get_trace()
    max_signal_power = max(data[:, 1])
    return max_signal_power

with program() as mixer_cal:

    with infinite_loop_():
        play("CW", "qubit")

def leakageMap():

    offI = np.arange(-0.29, 0.3, 0.04)
    offQ = np.arange(-0.29, 0.3, 0.04)

    amps = []

    for i in offI:

        for j in offQ:

            qm.set_dc_offset_by_qe("qubit", "I", i)
            qm.set_dc_offset_by_qe("qubit", "Q", j)
            amp_ = get_amp()
            amps.append(amp_)
            print("amp = {}, offI = {}, offQ = {}".format(amp_, i, j))

    return offI, offQ, amps


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

    phi = np.arange(-np.pi, np.pi, np.pi/10)
    g = np.arange(-0.2, 0.2, 0.02)

    amps = []

    for i in phi:

        for j in g:

            qm.set_mixer_correction("mixer_qubit", int(qubit_IF), int(qubit_LO), IQ_imbalance_corr(j, i))
            amp_ = get_amp()
            amps.append(amp_)
            print("amp = {}, phase = {}, gain = {}".format(amp_, i, j))

    return phi, g, amps


def gradImbalances(phase_, gain_):

    eps = 0.0002
    qm.set_mixer_correction("mixer_qubit", int(qubit_IF), int(qubit_LO), IQ_imbalance_corr(gain_, phase_ + eps))
    a1 = get_amp()
    qm.set_mixer_correction("mixer_qubit", int(qubit_IF), int(qubit_LO), IQ_imbalance_corr(gain_, phase_ - eps))
    a2 = get_amp()
    qm.set_mixer_correction("mixer_qubit", int(qubit_IF), int(qubit_LO), IQ_imbalance_corr(gain_ + eps, phase_))
    a3 = get_amp()
    qm.set_mixer_correction("mixer_qubit", int(qubit_IF), int(qubit_LO), IQ_imbalance_corr(gain_ - eps, phase_))
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
spectrum_ip = "192.168.116.159"
spec = automation.Spectrum(spectrum_ip)
LO = 3e9
delta_F = 20e6
spec.frequency(LO, LO - delta_F, LO + delta_F)
time.sleep(2)

# LO leakage 2D map
offI, offQ, amps = leakageMap()
plt.figure()
offI_grid, offQ_grid = np.meshgrid(offI, offQ)
amps_grid = np.transpose(np.reshape(amps, [len(offI),len(offQ)]))
plt.pcolormesh(offI_grid, offQ_grid, amps_grid)
plt.colorbar()
plt.xlabel("I offset")
plt.ylabel("Q offset")

# Gradient descent for the LO peak
indices = np.where(amps_grid == np.min(amps_grid))
offI0 = offI_grid[indices[0][0]][indices[1][0]]
offQ0 = offQ_grid[indices[0][0]][indices[1][0]]
print("OffI_min = {}, offQ_min = {}".format(offI0, offQ0))
offIf, offQf, offI_track, offQ_track = grad_descent(offI0, offQ0, "LO")

# Configure the SA:
LO = 3e9
IF_freq = qubit_IF
LSB = LO-IF_freq
delta_F = 20e6
spec.frequency(LSB, LSB-delta_F, LSB+delta_F)
time.sleep(2)

# IQ imbalances 2D map
phase, gain, amps = imbalancesMap()
plt.figure()
phase_grid, gain_grid = np.meshgrid(phase, gain)
amps_grid = np.transpose(np.reshape(amps, [len(phase), len(gain)]))
plt.pcolormesh(phase_grid, gain_grid, amps_grid)
plt.colorbar()
plt.xlabel("phase")
plt.ylabel("gain")

# Gradient descent for the SSB peak
indices = np.where(amps_grid == np.min(amps_grid))
phase0 = phase_grid[indices[0][0]][indices[1][0]]
gain0 = gain_grid[indices[0][0]][indices[1][0]]
print("phase_min = {}, gain_min = {}".format(phase0, gain0))
phasef, gainf, phase_track, ogain_track = grad_descent(phase0, gain0, "LSB")


