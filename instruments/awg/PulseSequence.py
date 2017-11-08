__author__ = 'dave'
import numpy as np

from slab.instruments.awg import write_Tek5014_file, write_Tek70001_sequence, write_PXDAC4800_file, M8195A, upload_M8195A_sequence
from slab.instruments.awg.PXDAC4800 import PXDAC4800
from slab.instruments import InstrumentManager, LocalInstruments
import os
import time


def round_samples(x, min_samples=0, increment=1):
    ## non-integer round samples
    return max(min_samples, increment * int(np.ceil(float(x) / float(increment))))

class PulseSequence:
    def __init__(self, name, awg_info, sequence_length):
        self.name = name.lower()
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
                # round_samples ensure waveform_length is int multiple of 1/clk_speed
                waveform_length = round_samples(self.waveform_info[waveform['name']]['length'], increment=1.0/awg['clock_speed'])
                waveform_clk_length = round_samples(waveform_length * awg['clock_speed'], awg['min_samples'],
                                                    awg['min_increment'])
                self.waveforms[waveform['name']] = np.zeros((self.sequence_length, waveform_clk_length))
                self.waveform_info[waveform['name']]['tpts'] = np.linspace(0., (waveform_clk_length - 1) / awg[
                    'clock_speed'], waveform_clk_length)

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
        write_function = {'Tek5014': self.write_Tek5014_sequence, 'Tek70001': self.write_Tek70001_sequence,
                          'PXDAC4800_1': self.write_PXDAC4800_1_sequence,
                          'PXDAC4800_2': self.write_PXDAC4800_2_sequence,
                          'PXDAC4800_3': self.write_PXDAC4800_3_sequence,
                          'M8195A': self.write_M8195A_sequence}
        for awg in self.awg_info:
            # try:
            print awg['type']
            if not awg['type'] == "NONE":
                write_function[awg['type']](awg, path, file_prefix, awg['upload'])
            # except KeyError:
            #     print "Error in writing pulse to awg named: " + str(awg['type'])

    def write_M8195A_sequence(self, awg, path, file_prefix, upload=False):

        start_time = time.time()
        print '\nStart writing M8195A sequences...(PulseSequence.py)'
        print 'fast awg waveforms (ch1-4):', [waveform['name'] for waveform in awg['waveforms']]

        # todo: this is where RAM blows up..
        waveform_matrix = [self.waveforms[waveform['name']] for waveform in awg['waveforms']]

        # im = InstrumentManager()
        # print type(im[awg['name']])
        m8195a = M8195A(address='192.168.14.244:5025')
        upload_M8195A_sequence(m8195a,waveform_matrix, awg)

        end_time = time.time()
        print 'Finished writing M8195A sequences in', end_time - start_time, 'seconds.\n'

    def write_Tek5014_sequence(self, awg, path, file_prefix, upload=False):
        waveforms = [self.waveforms[waveform['name']] for waveform in awg['waveforms']]
        markers = [self.markers[marker['name']] for marker in awg['markers']]
        write_Tek5014_file(waveforms, markers, os.path.join(path, file_prefix + '.awg'), self.name)

        if upload:
            im = InstrumentManager()
            im[awg['name']].pre_load()
            # print "Sequence preloaded"
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

    def write_PXDAC4800_1_sequence(self, awg, path, file_prefix, upload=False):
        self.write_PXDAC4800_sequence(awg, path, file_prefix, upload, 1)

    def write_PXDAC4800_2_sequence(self, awg, path, file_prefix, upload=False):
        self.write_PXDAC4800_sequence(awg, path, file_prefix, upload, 2)

    def write_PXDAC4800_3_sequence(self, awg, path, file_prefix, upload=False):
        self.write_PXDAC4800_sequence(awg, path, file_prefix, upload, 3)

    def write_PXDAC4800_sequence(self, awg, path, file_prefix, upload=False, brdNum=0):
        waveforms = [self.waveforms[waveform['name']] for waveform in awg['waveforms']]
        offset_bytes_list = write_PXDAC4800_file(waveforms, os.path.join(path, file_prefix + '_%d.rd16' % brdNum),
                                                 self.name,
                                                 awg['iq_offsets_bytes'], awg['sample_size'])
        # TODO : code would not work if upload is false
        if upload:
            pxdac4800 = LocalInstruments().inst_dict['pxdac4800_%d' % brdNum]
            pxdac4800.load_sequence_file(os.path.join(path, file_prefix + '_%d.rd16' % brdNum), awg)
            print "Sequence file uploaded"
            print "Waveform length: " + str(len(waveforms[0][0]))
            pxdac4800.waveform_length = len(waveforms[0][0])
            print "PXDAC4800 waveform length: " + str(pxdac4800.waveform_length)
            pxdac4800.run_experiment()

            return pxdac4800

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
