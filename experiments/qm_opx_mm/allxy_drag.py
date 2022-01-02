from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration_IQ import config, disc_file_opt, ge_IF, disc_file_opt_drag
import numpy as np
from matplotlib import pyplot as plt
from qm.qua import *
from TwoStateDiscriminator_2103 import TwoStateDiscriminator

qmm = QuantumMachinesManager()
discriminator = TwoStateDiscriminator(qmm, config, True, 'rr', disc_file_opt_drag, lsb=True)

points = 50e3
pulse_len_cycle = 40//4

with program() as allxy:

    n = declare(int)
    r = Random()
    r_ = declare(int)
    st = [declare_stream() for i in range(21)]
    res = declare(bool)

    update_frequency('qubit_mode0', int(ge_IF[0]))

    with for_(n, 0, n < points, n+1):

        assign(r_, r.rand_int(21))

        wait(int((120*5)*1e3)//4, 'qubit_mode0')

        align('qubit_mode0', 'rr')

        with switch_(r_):
            with case_(0):
                wait(pulse_len_cycle, 'qubit_mode0')
                wait(pulse_len_cycle, 'qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[0])
            with case_(1):
                play('pi_drag', 'qubit_mode0')
                play('pi_drag', 'qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[1])
            with case_(2):
                frame_rotation(np.pi/2, 'qubit_mode0'); play('pi_drag', 'qubit_mode0'); reset_frame('qubit_mode0')
                frame_rotation(np.pi/2, 'qubit_mode0'); play('pi_drag', 'qubit_mode0'); reset_frame('qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[2])
            with case_(3):
                play('pi_drag', 'qubit_mode0')
                frame_rotation(np.pi/2, 'qubit_mode0'); play('pi_drag', 'qubit_mode0'); reset_frame('qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[3])
            with case_(4):
                frame_rotation(np.pi/2, 'qubit_mode0'); play('pi_drag', 'qubit_mode0'); reset_frame('qubit_mode0')
                play('pi_drag', 'qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[4])
            with case_(5):
                play('pi_2_drag', 'qubit_mode0')
                wait(pulse_len_cycle, 'qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[5])
            with case_(6):
                frame_rotation(np.pi/2, 'qubit_mode0'); play('pi_2_drag', 'qubit_mode0'); reset_frame('qubit_mode0')
                wait(pulse_len_cycle, 'qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[6])
            with case_(7):
                play('pi_2_drag', 'qubit_mode0')
                frame_rotation(np.pi/2, 'qubit_mode0'); play('pi_2_drag', 'qubit_mode0'); reset_frame('qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[7])
            with case_(8):
                frame_rotation(np.pi/2, 'qubit_mode0'); play('pi_2_drag', 'qubit_mode0'); reset_frame('qubit_mode0')
                play('pi_2_drag', 'qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[8])
            with case_(9):
                play('pi_2_drag', 'qubit_mode0')
                frame_rotation(np.pi/2, 'qubit_mode0'); play('pi_drag', 'qubit_mode0'); reset_frame('qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[9])
            with case_(10):
                frame_rotation(np.pi/2, 'qubit_mode0'); play('pi_2_drag', 'qubit_mode0'); reset_frame('qubit_mode0')
                play('pi_drag', 'qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[10])
            with case_(11):
                play('pi_drag', 'qubit_mode0')
                frame_rotation(np.pi/2, 'qubit_mode0'); play('pi_2_drag', 'qubit_mode0'); reset_frame('qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[11])
            with case_(12):
                frame_rotation(np.pi/2, 'qubit_mode0'); play('pi_drag', 'qubit_mode0'); reset_frame('qubit_mode0')
                play('pi_2_drag', 'qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[12])
            with case_(13):
                play('pi_2_drag', 'qubit_mode0')
                play('pi_drag', 'qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[13])
            with case_(14):
                play('pi_drag', 'qubit_mode0')
                play('pi_2_drag', 'qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[14])
            with case_(15):
                frame_rotation(np.pi/2, 'qubit_mode0'); play('pi_2_drag', 'qubit_mode0'); reset_frame('qubit_mode0')
                frame_rotation(np.pi/2, 'qubit_mode0'); play('pi_drag', 'qubit_mode0'); reset_frame('qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[15])
            with case_(16):
                frame_rotation(np.pi/2, 'qubit_mode0'); play('pi_drag', 'qubit_mode0'); reset_frame('qubit_mode0')
                frame_rotation(np.pi/2, 'qubit_mode0'); play('pi_2_drag', 'qubit_mode0'); reset_frame('qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[16])
            with case_(17):
                play('pi_drag', 'qubit_mode0')
                wait(pulse_len_cycle, 'qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[17])
            with case_(18):
                frame_rotation(np.pi/2, 'qubit_mode0'); play('pi_drag', 'qubit_mode0'); reset_frame('qubit_mode0')
                wait(pulse_len_cycle, 'qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[18])
            with case_(19):
                play('pi_2_drag', 'qubit_mode0')
                play('pi_2_drag', 'qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[19])
            with case_(20):
                frame_rotation(np.pi/2, 'qubit_mode0'); play('pi_2_drag', 'qubit_mode0'); reset_frame('qubit_mode0')
                frame_rotation(np.pi/2, 'qubit_mode0'); play('pi_2_drag', 'qubit_mode0'); reset_frame('qubit_mode0')
                align('qubit_mode0', 'rr')
                wait(20, 'rr');discriminator.measure_state("clear", "out1", "out2", res)
                save(res, st[20])

    with stream_processing():
        for i in range(21):
            st[i].boolean_to_int().average().save('res{}'.format(i))

qm = qmm.open_qm(config)
job = qm.execute(allxy, duration_limit=0, data_limit=0)
job.result_handles.wait_for_all_values()
job.halt()
res = []
for x in range(21):
    res.append(job.result_handles.get("res{}".format(x)).fetch_all())


sequence = [  # based on https://rsl.yale.edu/sites/default/files/physreva.82.pdf-optimized_driving_0.pdf
    ('I', 'I'),
    ('X', 'X'),
    ('Y', 'Y'),
    ('X', 'Y'),
    ('Y', 'X'),
    ('X/2', 'I'),
    ('Y/2', 'I'),
    ('X/2', 'y'),
    ('Y/2', 'X/2'),
    ('X/2', 'Y'),
    ('Y/2', 'X'),
    ('X', 'Y/2'),
    ('Y', 'X/2'),
    ('X/2', 'X'),
    ('X', 'X/2'),
    ('Y/2', 'Y'),
    ('Y', 'Y/2'),
    ('X', 'I'),
    ('Y', 'I'),
    ('X/2', 'X/2'),
    ('Y/2', 'Y/2'),
]
plt.figure()
plt.xticks(ticks=range(21), labels=[str(el) for el in sequence], rotation=90)
res=np.array(res)
values=res*(-2)+1
plt.plot(values, 'bD')

# plt.plot(res, '.')
plt.show()
# np.savez('all_xy_values', values)



