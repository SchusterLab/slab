__author__ = 'dave'
import numpy as np

from slab.instruments.awg import write_Tek5014_file, write_Tek70001_waveform_file
from slab.instruments import InstrumentManager


class PulseSequence:
    def __init__(self, name, awg_info, waveform_length, sequence_length):
        self.name = name
        self.awg_info = awg_info
        self.waveform_length = waveform_length
        self.sequence_length = sequence_length
        self.waveforms = {}
        self.waveform_info = {}
        self.markers = {}
        self.marker_info = {}

        for awg in awg_info:
            for waveform in awg['waveforms']:
                self.waveforms[waveform['name']] = np.zeros((self.sequence_length, self.waveform_length))
                self.waveform_info[waveform['name']] = waveform.copy()
                self.waveform_info[waveform['name']]['tpts'] = np.linspace(0.,
                                                                           self.waveform_length / float(
                                                                               awg['clock_speed']),
                                                                           self.waveform_length)
            for marker in awg['markers']:
                self.markers[marker['name']] = np.zeros((self.sequence_length, self.waveform_length))
                self.marker_info[marker['name']] = marker.copy()
                self.marker_info[marker['name']]['tpts'] = np.linspace(0.,
                                                                       self.waveform_length / float(
                                                                           awg['clock_speed']),
                                                                       self.waveform_length)

    def get_waveform_times(self, name):
        return self.waveform_info[name]['tpts']

    def get_marker_times(self, name):
        return self.marker_info[name]['tpts']

    def write_sequence(self, file_prefix, upload=False):
        write_function = {'Tek5014': self.write_Tek5014_sequence, 'Tek70001': self.write_Tek70001_sequence}
        for awg in self.awg_info:
            write_function[awg['type']](awg, file_prefix, upload)

    def write_Tek5014_sequence(self, awg, file_prefix, upload=False):
        waveforms=[self.waveforms[waveform['name']] for waveform in awg['waveforms']]
        markers=[self.markers[marker['name']] for marker in awg['markers']]
        write_Tek5014_file(waveforms,markers,file_prefix+'.awg',self.name)

        if upload:
            im = InstrumentManager()
            im[awg['name']].pre_load()
            im[awg['name']].load_sequence_file(file_prefix+'.awg',force_reload=True)
            im[awg['name']].prep_experiment()

    def write_Tek70001_sequence(self, awg, file_prefix, upload=False):
        pass

    def build_sequence(self):
        """Abstract method to be implemented by specific sequences, fills out waveforms and markers"""
        pass

    def reshape_data(self, data):
        """Abstract method which reshapes data taken from the acquisition card"""
        pass


class PulseSequenceArray:
    def __init__(self, sequences):
        self.sequences = sequences

    def write_sequences(self):
        pass

    def reshape_data(self, data):
        pass


