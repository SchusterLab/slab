import numpy as np
import visdom
from slab.datamanagement import SlabFile
from slab.dataanalysis import get_next_filename
import os
path = os.getcwd()

try:
    from .pulse_classes import Gauss, Idle, Ones, Square
except:
    from pulse_classes import Gauss, Idle, Ones, Square


class Sequencer:

    def __init__(self, channels, channels_awg, awg_info, channels_delay):
        self.channels = channels
        self.channels_awg = channels_awg
        self.awg_info = awg_info
        self.channels_delay = channels_delay

        channels_awg_info = {}

        for channel in channels:
            channels_awg_info[channel] = awg_info[channels_awg[channel]]

        self.channels_awg_info = channels_awg_info

        self.pulse_array_list = {}
        self.multiple_sequences = []

    def new_sequence(self, sequences):
        self.pulse_array_list = {}
        for channel in self.channels:
            idle = Idle(time=100, dt=self.channels_awg_info[channel]['dt'])
            idle.generate_pulse_array()
            self.pulse_array_list[channel] = [idle.pulse_array]

        # M8195a trig
        self.append('m8195a_trig', Ones(time=sequences.hardware_cfg['trig_pulse_len']['m8195a']))

        # sideband cooling
        for qubit_id in sequences.expt_cfg.get('on_qubits', ['1','2']):
            if sequences.sideband_cooling[qubit_id]['cool']:
                self.append('flux%s'%qubit_id,
                             Square(max_amp=sequences.multimodes[qubit_id]['pi_amp'][sequences.sideband_cooling[qubit_id]['mode_id']], flat_len=sequences.multimodes[qubit_id]['pi_len'][sequences.sideband_cooling[qubit_id]['mode_id']],
                                    ramp_sigma_len=sequences.quantum_device_cfg['flux_pulse_info'][qubit_id]['ramp_sigma_len'], cutoff_sigma=2, freq=sequences.multimodes[qubit_id]['freq'][sequences.sideband_cooling[qubit_id]['mode_id']], phase=0,
                                    plot=False))

                self.append('flux%s'%qubit_id,
                             Idle(time=sequences.multimodes[qubit_id]['pi_len'][sequences.sideband_cooling[qubit_id]['mode_id']]))

        self.sync_channels_time(sequences.channels)



    def append(self, channel, pulse):
        pulse.generate_pulse_array(t0=self.get_time(channel), dt=self.channels_awg_info[channel]['dt'])
        pulse_array = pulse.pulse_array
        self.pulse_array_list[channel].append(pulse_array)

    def get_time(self, channel):
        return self.channels_awg_info[channel]['dt'] * len(np.concatenate(self.pulse_array_list[channel])) + \
               self.channels_awg_info[channel]['time_delay']

    def append_idle_to_time(self, channel, time):
        current_time = self.get_time(channel)
        extra_time = time - current_time - 0.0001*self.channels_awg_info[channel]['dt']
        # the "- delta*self.channels_awg_info[channel]['dt']" is to add numerical stability
        # when expect time - current_time ==  I (an integer) multiple of clock rate, i.e. expect numpy array with lenth I
        # if time - current_time is slightly larger than I * dt,
        # np.arange(0, I * dt + eps, self.dt) will result a numpy array with length I+1

        if extra_time > 0:
            self.append(channel, Idle(time=extra_time, dt=self.channels_awg_info[channel]['dt']))

    def sync_channels_time(self, channels):

        max_time = 0

        for channel in channels:
            channel_time = self.get_time(channel)
            if channel_time > max_time:
                max_time = channel_time

        buffer_time = 5 #add 5ns buffer time everytime sync
        max_time += buffer_time

        for channel in channels:
            self.append_idle_to_time(channel, max_time)

    def rectify_pulse_len(self, sequence):
        for channel in self.channels:
            channel_len = len(sequence[channel])

            min_samples = self.channels_awg_info[channel]['min_samples']

            if channel_len < min_samples:
                sequence[channel] = np.pad(sequence[channel], (0, min_samples - channel_len), 'constant',
                                           constant_values=(0, 0))

            channel_len = len(sequence[channel])
            min_increment = self.channels_awg_info[channel]['min_increment']

            if channel_len % min_increment != 0:
                sequence[channel] = np.pad(sequence[channel], (0, min_increment - (channel_len % min_increment)),
                                           'constant', constant_values=(0, 0))

        return sequence

    def equalize_sequences(self):
        awg_max_len = {}
        awg_list = set(self.channels_awg.values())
        for awg in awg_list:
            awg_max_len[awg] = 0

        for sequence in self.multiple_sequences:
            for channel in self.channels:
                channel_len = len(sequence[channel])
                if channel_len > awg_max_len[self.channels_awg[channel]]:
                    awg_max_len[self.channels_awg[channel]] = channel_len

        for sequence in self.multiple_sequences:
            for channel in self.channels:
                channel_len = len(sequence[channel])
                if channel_len < awg_max_len[self.channels_awg[channel]]:
                    sequence[channel] = np.pad(sequence[channel],
                                               (0, awg_max_len[self.channels_awg[channel]] - channel_len), 'constant',
                                               constant_values=(0, 0))

    def delay_channels(self, channels_delay):
        for sequence in self.multiple_sequences:
            for channel, delay in channels_delay.items():
                delay_num = int(delay / self.channels_awg_info[channel]['dt'])
                sequence[channel] = np.roll(sequence[channel], delay_num)

    def get_sequence(self):
        sequence = {}
        for channel in self.channels:
            sequence[channel] = np.concatenate(self.pulse_array_list[channel])

        return sequence

    def end_sequence(self):
        for channel in self.channels:
            self.append(channel,Idle(time=100))
        sequence = self.get_sequence()
        sequence = self.rectify_pulse_len(sequence)

        self.multiple_sequences.append(sequence)

    def complete(self, sequences, plot=True):
        if sequences.expt_cfg.get('9_calibration', False):
            qubit_state = ['g','e','f']

            for qubit_1_state in qubit_state:
                for qubit_2_state in qubit_state:
                    self.new_sequence(sequences)
                    qubit_id = "1"
                    if qubit_1_state == 'e':
                        self.append('charge%s' %qubit_id, sequences.qubit_pi[qubit_id])
                    if qubit_1_state == 'f':
                        self.append('charge%s' %qubit_id, sequences.qubit_pi[qubit_id])
                        self.append('charge%s' %qubit_id, sequences.qubit_ef_pi[qubit_id])

                    qubit_id = "2"
                    if qubit_2_state == 'e':
                        self.append('charge%s' %qubit_id, sequences.qubit_pi[qubit_id])
                    if qubit_2_state == 'f':
                        self.append('charge%s' %qubit_id, sequences.qubit_pi[qubit_id])
                        self.append('charge%s' %qubit_id, sequences.qubit_ef_pi[qubit_id])

                    sequences.readout(self, sequences.expt_cfg.get('on_qubits',["1", "2"]))
                    self.end_sequence()

        elif sequences.expt_cfg.get('4_calibration', False):
            self.new_sequence(sequences)
            sequences.readout(self, sequences.expt_cfg.get('on_qubits',["1", "2"]))
            self.end_sequence()

            self.new_sequence(sequences)
            qubit_id = "2"
            self.append('charge%s' %qubit_id, sequences.qubit_pi[qubit_id])
            sequences.readout(self, sequences.expt_cfg.get('on_qubits',["1", "2"]))
            self.end_sequence()

            self.new_sequence(sequences)
            qubit_id = "1"
            self.append('charge%s' %qubit_id, sequences.qubit_pi[qubit_id])
            sequences.readout(self, sequences.expt_cfg.get('on_qubits',["1", "2"]))
            self.end_sequence()

            self.new_sequence(sequences)
            for qubit_id in sequences.expt_cfg.get('on_qubits',["1","2"]):
                self.append('charge%s' %qubit_id, sequences.qubit_pi[qubit_id])
            sequences.readout(self, sequences.expt_cfg.get('on_qubits',["1", "2"]))
            self.end_sequence()

        elif sequences.expt_cfg.get('4_calibration_single_cavity', False):
            self.new_sequence(sequences)
            sequences.readout(self, "1")
            self.end_sequence()

            self.new_sequence(sequences)
            charge_port = sequences.charge_port
            qubit_id = "2"
            self.append('charge%s' %charge_port[qubit_id], sequences.qubit_pi[qubit_id])
            sequences.readout(self, "1")
            self.end_sequence()

            self.new_sequence(sequences)
            charge_port = sequences.charge_port
            qubit_id = "1"
            self.append('charge%s' %charge_port[qubit_id], sequences.qubit_pi[qubit_id])
            sequences.readout(self, "1")
            self.end_sequence()

            self.new_sequence(sequences)
            charge_port = sequences.charge_port
            qubit_id = "2"
            self.append('charge%s' %charge_port[qubit_id], sequences.qubit_pi[qubit_id])
            self.sync_channels_time(sequences.channels)
            self.append('charge2', sequences.qubit_ee_pi["1"])
            sequences.readout(self, "1")
            self.end_sequence()

        elif sequences.expt_cfg.get('pi_calibration', False):

            self.new_sequence(sequences)
            sequences.readout(self, sequences.expt_cfg.get('calibration_qubit',["1","2"]))
            self.end_sequence()


            self.new_sequence(sequences)
            charge_port = sequences.charge_port
            for qubit_id in sequences.expt_cfg.get('calibration_qubit',["1","2"]):

                if sequences.expt_cfg.get('flux_pi_calibration', False):
                    self.append('flux1', sequences.qubit_pi[qubit_id])
                else:
                    self.append('charge%s' %charge_port[qubit_id], sequences.qubit_pi[qubit_id])

            sequences.readout(self, sequences.expt_cfg.get('calibration_qubit',["1","2"]))
            self.end_sequence()

        self.equalize_sequences()

        self.delay_channels(self.channels_delay)

        if plot:
            self.plot_sequences()

        return self.multiple_sequences

    def plot_sequences_backup(self):

        vis = visdom.Visdom()
        vis.close()

        sequence_id = 0

        for sequence in self.multiple_sequences[::20]:

            sequence_id += 1

            vis = visdom.Visdom()
            win = vis.line(
                X=np.arange(0, 1),
                Y=np.arange(0, 1),
                opts=dict(
                    legend=[self.channels[0]], title='seq %d' % sequence_id, xlabel='Time (ns)'))

            kk = 0
            for channel in self.channels:
                sequence_array = sequence[channel]
                if sequence_id ==1:
                    print(sequence_array)
                    print(channel)
                    data_path = os.path.join(path, 'data/')
                    data_file = os.path.join(data_path, get_next_filename(data_path, 'sequence', suffix='.h5'))
                    self.slab_file = SlabFile(data_file)
                    with self.slab_file as f:
                        f.add(channel, sequence_array)
                        f.close()
                vis.updateTrace(
                    X=np.arange(0, len(sequence_array)) * self.channels_awg_info[channel]['dt'] +
                      self.channels_awg_info[channel]['time_delay'],
                    Y=sequence_array + 2 * (len(self.channels) - kk),
                    win=win, name=channel, append=False)

                kk += 1

    def plot_sequences(self):

        vis = visdom.Visdom()
        vis.close()

        sequence_id = 0

        for sequence in self.multiple_sequences[::20]:

            sequence_id += 1

            vis = visdom.Visdom()
            win = vis.line(
                X=np.arange(0, 1),
                Y=np.arange(0, 1),
                opts=dict(
                    legend=[self.channels[0]], title='seq %d' % sequence_id, xlabel='Time (ns)'))

            kk = 0
            if sequence_id >=1: # Set to ==1 if wanna see only the first sequence
                print("=========\nseq_id: %s" %sequence_id)
                data_path = os.path.join(path, 'data/')
                data_file = os.path.join(data_path, get_next_filename(data_path, 'sequence', suffix='.h5'))
                self.slab_file = SlabFile(data_file)
                with self.slab_file as f:
                    for channel in self.channels:
                        sequence_array = sequence[channel]

                        # print(sequence_array)
                        print(channel)

                        f.add(channel, sequence_array)

                        vis.updateTrace(
                            X=np.arange(0, len(sequence_array)) * self.channels_awg_info[channel]['dt'] +
                              self.channels_awg_info[channel]['time_delay'],
                            Y=sequence_array + 2 * (len(self.channels) - kk),
                            win=win, name=channel, append=False)

                        kk += 1
                    f.close()

def testing_function():
    vis = visdom.Visdom()
    vis.close()

    channels = ['charge1', 'flux1', 'charge2', 'flux2',
                'hetero1_I', 'hetero1_Q', 'hetero2_I', 'hetero2_Q',
                'm8195a_trig', 'readout1_trig', 'readout2_trig', 'alazar_trig']

    channels_awg = {'charge1': 'm8195a', 'flux1': 'm8195a', 'charge2': 'm8195a', 'flux2': 'm8195a',
                    'hetero1_I': 'tek5014a', 'hetero1_Q': 'tek5014a', 'hetero2_I': 'tek5014a', 'hetero2_Q': 'tek5014a',
                    'm8195a_trig': 'tek5014a', 'readout1_trig': 'tek5014a', 'readout2_trig': 'tek5014a',
                    'alazar_trig': 'tek5014a'}

    awg_info = {'m8195a': {'dt': 1. / 16., 'min_increment': 16, 'min_samples': 128, 'time_delay': 110},
                'tek5014a': {'dt': 1. / 1.2, 'min_increment': 16, 'min_samples': 128, 'time_delay': 0}}

    channels_delay = {'readout1_trig': -20, 'readout2_trig': -20, 'alazar_trig': -50}

    sequencer = Sequencer(channels, channels_awg, awg_info, channels_delay)

    # sequence 1
    sequencer.new_sequence()

    sequencer.append('charge1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=1, phase=0, plot=False))
    sequencer.append('charge1', Idle(time=10))
    sequencer.append('charge1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2, phase=np.pi))

    sequencer.append('flux1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2, phase=np.pi))

    sequencer.append('charge2', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2, phase=np.pi))

    sequencer.append('flux2', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2, phase=np.pi))

    sequencer.end_sequence()

    # sequence 2
    sequencer.new_sequence()

    sequencer.append('charge1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=1, phase=0, plot=False))
    sequencer.append('charge1', Idle(time=10))
    sequencer.append('charge1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2, phase=np.pi))

    sequencer.append('flux1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2, phase=np.pi))

    sequencer.append('charge2', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2, phase=np.pi))

    sequencer.append('flux2', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2, phase=np.pi))

    sequencer.end_sequence()

    # sequence 3
    sequencer.new_sequence()

    sequencer.append('m8195a_trig', Ones(time=100))

    sequencer.append('charge1', Gauss(max_amp=0.5, sigma_len=10, cutoff_sigma=2, freq=4.524, phase=0, plot=False))
    sequencer.append('charge1', Idle(time=10))
    sequencer.append('charge1', Gauss(max_amp=0.5, sigma_len=10, cutoff_sigma=2, freq=4.524, phase=np.pi))

    sequencer.append('flux1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2.1, phase=np.pi))

    sequencer.append('charge2', Gauss(max_amp=0.5, sigma_len=30.2, cutoff_sigma=2, freq=4.524, phase=0))

    sequencer.append('flux2', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2.1, phase=np.pi))

    sequencer.sync_channels_time(['charge1', 'flux1', 'charge2'])

    sequencer.append('charge1', Gauss(max_amp=0.5, sigma_len=10, cutoff_sigma=2, freq=4.5, phase=0, plot=False))
    sequencer.append('charge1', Idle(time=10))
    sequencer.append('charge1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=4.524, phase=np.pi))

    sequencer.append('flux1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2.1, phase=np.pi))

    sequencer.append('charge2', Gauss(max_amp=0.5, sigma_len=20.3, cutoff_sigma=2, freq=4.524, phase=0))

    sequencer.sync_channels_time(channels)

    sequencer.append('flux1', Gauss(max_amp=0.5, sigma_len=15, cutoff_sigma=2, freq=2.1, phase=np.pi))
    sequencer.append('flux2', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2.1, phase=np.pi))
    sequencer.append('flux2', Gauss(max_amp=0.5, sigma_len=10, cutoff_sigma=2, freq=2.1, phase=np.pi))

    sequencer.sync_channels_time(['charge1', 'charge2', 'flux2'])
    sequencer.append('charge1', Gauss(max_amp=0.5, sigma_len=15, cutoff_sigma=2, freq=4.524, phase=np.pi))
    sequencer.append('charge2', Gauss(max_amp=0.5, sigma_len=30, cutoff_sigma=2, freq=4.524, phase=np.pi))
    sequencer.append('flux2', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2.1, phase=np.pi))

    sequencer.sync_channels_time(['flux1', 'flux2'])

    sequencer.append('flux1', Gauss(max_amp=0.5, sigma_len=20, cutoff_sigma=2, freq=2.1, phase=np.pi))

    sequencer.sync_channels_time(channels)

    sequencer.append('hetero1_I', Square(max_amp=0.5, flat_len= 200, ramp_sigma_len=20, cutoff_sigma=2, freq=0.2, phase=0, plot=False))
    sequencer.append('hetero1_Q', Square(max_amp=0.5, flat_len= 200, ramp_sigma_len=20, cutoff_sigma=2, freq=0.2, phase=np.pi / 2))

    sequencer.append('hetero2_I', Square(max_amp=0.5, flat_len= 200, ramp_sigma_len=20, cutoff_sigma=2, freq=0.2, phase=0, plot=False))
    sequencer.append('hetero2_Q', Square(max_amp=0.5, flat_len= 200, ramp_sigma_len=20, cutoff_sigma=2, freq=0.2, phase=np.pi / 2))

    sequencer.append('alazar_trig', Ones(time=100))
    sequencer.append('readout1_trig', Ones(time=250))
    sequencer.append('readout2_trig', Ones(time=250))

    sequencer.end_sequence()

    sequencer.complete(plot=True)


if __name__ == "__main__":
    testing_function()




