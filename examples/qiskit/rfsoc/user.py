"""
user.py
"""

import time

import h5py
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from slab import generate_file_path
from slab.instruments.qiskit import SLabProviderInterface
from qiskit import pulse
from qiskit.providers.jobstatus import JOB_FINAL_STATES
from qiskit.qobj.utils import MeasLevel, MeasReturnType

DPI = 300

# unit conversion factors -> all backend properties returned in SI (Hz, sec, etc)
GHz = 1.0e9 #Hz
MHz = 1.0e6 #Hz
us = 1.0e-6 #s
ns = 1.0e-9 #s

def get_closest_multiple_of_16(num):
    return int(num + 8 ) - (int(num + 8 ) % 16)
#ENDDEF

def rs(plot=True):
    provider = SLabProviderInterface()
    backend = provider.get_backend("RFSoC2")
    backend_config = backend.configuration()
    backend_defaults = backend.defaults()
    dt = backend_config.dt #s

    qubit_chan = pulse.DriveChannel(0)
    cavity_chan = pulse.DriveChannel(1)
    meas_chan = pulse.MeasureChannel(0)
    acq_chan = pulse.AcquireChannel(0)
    
    meas_amp = 0.125
    meas_duration = get_closest_multiple_of_16(32000) #dts
    meas_pulse = pulse.library.Constant(meas_duration, meas_amp)
    
    meas_freq_count = 100
    meas_freq_start = 90 * MHz #Hz
    meas_freq_stop = 110 * MHz #Hz
    meas_freqs = np.linspace(meas_freq_start, meas_freq_stop, meas_freq_count)
    
    schedule = pulse.Schedule(name="Frequency sweep")
    schedule += pulse.Play(meas_pulse, meas_chan)
    schedule += pulse.Acquire(meas_duration, acq_chan, pulse.MemorySlot(0))
    
    # fig = schedule.draw()
    # plot_file_path = generate_file_path(".", "rs_sched", "png")
    # plt.savefig(plot_file_path)
    # print(f"plotted schedule to {plot_file_path}")

    num_shots = 1000
    rep_delay = 10 * us #s
    schedule_los = list()
    for meas_freq in meas_freqs:
        schedule_los.append({
            meas_chan: meas_freq,
        })
    #ENDFOR
    
    job = backend.run(
        schedule, meas_level=MeasLevel.KERNELED,
        meas_return=MeasReturnType.AVERAGE, shots=num_shots,
        schedule_los=schedule_los,
        rep_delay=rep_delay,
        shots_per_set=10,
    )

    job.wait_for_final_state()
    result = job.result()

    # parse data
    data = np.zeros(meas_freq_count, dtype=np.complex128)
    amp_data = np.zeros(meas_freq_count, dtype=np.complex128)
    for i in range(meas_freq_count):
        data[i] = result.get_memory(i)[0]
        amp_data[i] = abs(data)
    #ENDFOR

    if plot:
        meas_freqs_Hz = meas_freqs / MHz
        fig = plt.figure()
        # plt.plot(meas_freqs_Hz, data.real, label="I")
        # plt.plot(meas_freqs_Hz, data.imag, label="Q")
        plt.plot(meas_freqs_Hz, amp_data)
        plt.scatter(meas_freqs_Hz, amp_data, label="A")
        plt.xlabel("DDS Frequency (MHz)")
        plt.ylabel("V (a.u.)")
        plt.legend()
        plot_file_path = generate_file_path(".", "t1_rs", "png")
        plt.savefig(plot_file_path, dpi=DPI)
        plt.close()
        print("plotted to {}".format(plot_file_path))
    #ENDIF
    
    return result
#ENDDEF

def main():
    rs()
#ENDDEF

if __name__ == "__main__":
    main()
#ENDIF
