__author__ = 'dave'


class QubitControl:
    def __init__(self, **kwargs):
        self.params = {}
        self.params.update(kwargs)
        self.IQ_ch = kwargs['IQ_ch']
        self.flux_ch = kwargs['flux_ch']


class IQSequence:
    """
    Object to specify IQ channel sequence
    """

    def __init__(self, seq=[]):
        """
        :param seq: initial sequence [] by default
        :return: None
        """
        self.seq = seq


class PulseSpecAWG:
    # Pulse Spectrometer Interface
    def setup(self, **kwargs):
        """
        This method is intended to configure the AWG for use
        within a PulseSpectrometer class

        :param kwargs: a dictionary of parameters specific to the
        individual AWG model that has the necessary parameters for initializing the AWG for use
        :return: None
        """
        pass

    def compile(self, upload=True):
        """
        Compile the IQ+direct channels/markers
        :param upload: if True it will automatically upload after compiling
        :return:
        """
        pass

    def upload(self):
        """
        Upload the compiled sequences to the AWG
        :return:
        """
        pass

    def clear_channels(self):
        """
        Delete all direct/IQ channels
        :return:
        """
        pass

    def get_IQchannel(self, Ich, Qch):
        """
        Get virtual IQChannel
        :param Ich: channel # for I
        :param Qch: channel # for Q
        :return: virtual IQChannel object composed of (Ich,Qch) physical channels
        """
        pass

    def get_direct_channel(self, ch):
        """
        Get direct AWG channel
        :param ch: Physical channel #
        :return: virtual direct AWG channel
        """
        pass

    def get_marker_channel(self, mch):
        """
        Get marker channel
        :param mch: Physical marker channel #
        :return: virtual marker channel
        """
        pass

    # standard AWG features

    def get_id(self):
        pass

    def set_ch_amp(self, ch, amp):
        pass

    def get_ch_amp(self, ch):
        pass

    def set_ch_offset(self, ch, offset):
        pass

    def get_ch_offset(self, ch):
        pass

    def set_output(self):
        pass

    def get_output(self):
        pass

    def run(self):
        pass

    def stop(self):
        pass


class DataSetInfo:
    def __init__(self, name, domain=None, labels=[],reduce_function=None,save_raw=True):
        self.name = name
        self.domain = domain
        self.labels = labels
        self.reduce_function = reduce_function
        self.save_raw=save_raw


class DataTag:
    def __init__(self, name, loc, sample_range=None):
        self.name = name
        self.loc = loc
        self.sample_range = sample_range



class MeasurementPlan:
    """
    MeasurementPlan contains:
    dataset definitions
    and the
    datamap
    """

    def __init__(self, rounds=1):
        self.dataset_infos = {}
        self.datamap = []
        self.rounds = rounds

    def add_infos(self, info):
        self.datasets[info.name] = info

    def add_trigger(self, data_tags, repeats=1):
        self.datamap.append(data_tags, repeats)


class ADC:
    """
    Acquisition card abstract class
    """

    def compile(self, upload=True):
        pass

    def process_data(self):
        pass

    def run(self):
        pass

    def stop(self):
        pass


class IQChannel:
    def __init__(self, rf, awg, Ich, Qch):
        self.rf = rf
        self.awg = awg
        self.Ich = Ich
        self.Qch = Qch


# How to address awgs/adcs/rfs, by name, by ref, ...
# how to pass them in, as list, as dict, via IM
class PulseSpectrometer:
    def __init__(self, awgs=[], adcs=[], rfs=[], **kwargs):
        self.awgs = awgs
        self.adcs = adcs
        self.rfs = rfs

        self.defaults = kwargs

    def get_IQChannel(self, rf, awg, Ich, Qch):
        return IQChannel(rf, awg, Ich, Qch)


    def compile(self, upload=True):
        for awg in self.awgs:
            awg.compile(upload)
        for adc in self.adcs:
            adc.compile(upload)

    def run(self):
        for adc in self.adcs:
            adc.run()

        for awg in self.awgs:
            awg.stop()
            awg.run()  # maybe need to check for master to do things in correct order

    def stop(self):
        for adc in self.adcs:
            adc.stop()

        for awg in self.awgs:
            awg.stop()

