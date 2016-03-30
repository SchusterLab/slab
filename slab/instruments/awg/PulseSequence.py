__author__ = 'dave'
import numpy as np

from slab.instruments.awg import write_Tek5014_file, write_Tek70001_sequence
from slab.instruments import InstrumentManager
import os


def round_samples(x, min_samples=0, increment=1):
    return max(min_samples, int(increment * np.ceil(float(x) / float(increment))))


class PulseSequence:
    def __init__(self, name, awg_info, sequence_length):
        self.name = name
        self.awg_info = awg_info
        self.sequence_length = sequence_length
        self.waveforms = {}
        self.waveform_info = {}
        self.markers = {}
        self.marker_info = {}

        for awg in awg_info:
            for waveform in awg['waveforms']:
                self.waveform_info[waveform['name']] = waveform.copy()

            for marker in awg['markers']:
                self.marker_info[marker['name']] = marker.copy()


    def init_waveforms_markers(self):
        for awg in self.awg_info:
            for waveform in awg['waveforms']:
                waveform_length=self.waveform_info[waveform['name']]['length']
                waveform_clk_length = round_samples( waveform_length* awg['clock_speed'],awg['min_samples'],awg['min_increment'])
                self.waveforms[waveform['name']] = np.zeros((self.sequence_length, waveform_clk_length))
                self.waveform_info[waveform['name']]['tpts'] = np.linspace(0., (waveform_clk_length-1)/awg['clock_speed'],waveform_clk_length)

            for marker in awg['markers']:
                marker_length=self.marker_info[marker['name']]['length']
                marker_clk_length = round_samples( marker_length* awg['clock_speed'],awg['min_samples'],awg['min_increment'])
                self.markers[marker['name']] = np.zeros((self.sequence_length, marker_clk_length))
                self.marker_info[marker['name']]['tpts'] = np.linspace(0., (marker_clk_length-1)/awg['clock_speed'],marker_clk_length)

    def set_all_lengths(self, length):
        for name in self.marker_info.keys():
            self.set_marker_length(name, length)

        for name in self.waveform_info.keys():
            self.set_waveform_length(name, length)

    def set_waveform_length(self, name, length):
        self.waveform_info[name]['length'] = length

    def set_marker_length(self, name, length):
        self.marker_info[name]['length'] = length

    def get_waveform_times(self, name):
        return self.waveform_info[name]['tpts']

    def get_marker_times(self, name):
        return self.marker_info[name]['tpts']

    def write_sequence(self, path, file_prefix, upload=False):
        write_function = {'Tek5014': self.write_Tek5014_sequence, 'Tek70001': self.write_Tek70001_sequence}
        for awg in self.awg_info:
            write_function[awg['type']](awg, path, file_prefix, awg['upload'])

    def write_Tek5014_sequence(self, awg, path, file_prefix, upload=False):
        waveforms = [self.waveforms[waveform['name']] for waveform in awg['waveforms']]
        markers = [self.markers[marker['name']] for marker in awg['markers']]
        write_Tek5014_file(waveforms, markers, os.path.join(path, file_prefix + '.awg'), self.name)

        if upload:
            im = InstrumentManager()
            im[awg['name']].pre_load()
            #print "Sequence preloaded"
            im[awg['name']].load_sequence_file(os.path.join(path, file_prefix + '.awg'), force_reload=True)
            print "Sequence file uploaded"
            im[awg['name']].prep_experiment()

    def write_Tek70001_sequence(self, awg, path, file_prefix, upload=False):
        waveforms = [self.waveforms[waveform['name']] for waveform in awg['waveforms']]
        # markers=[self.markers[marker['name']] for marker in awg['markers']]

        if upload:
            tek7 = InstrumentManager()[awg['name']]
            for waveform in waveforms:
                write_Tek70001_sequence(waveform, path, file_prefix, awg=tek7)
            tek7.prep_experiment()
            tek7.run()
        else:
            tek7 = None



    def build_sequence(self):
        """Abstract method to be implemented by specific sequences, fills out waveforms and markers"""
        self.init_waveforms_markers()

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


