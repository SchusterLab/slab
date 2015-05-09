__author__ = 'Nitrogen'

from liveplot import LivePlotClient
# from dataserver import dataserver_client
import os.path
import json

from slab import SlabFile, InstrumentManager, get_next_filename, AttrDict


class Experiment:
    """Base class for all experiments"""

    def __init__(self, expt_path='', prefix='data', config_file=None, **kwargs):
        """ Initializes experiment class
            @param expt_path - directory where data will be stored
            @param prefix - prefix to use when creating data files
            @param config_file - parameters for config file specified are loaded into the class dict
                                 (name relative to expt_directory if no leading /)
                                 Default = None looks for expt_directory/prefix.cfg

            @param **kwargs - by default kwargs are updated to class dict

            also loads InstrumentManager, LivePlotter, and other helpers
        """

        self.__dict__.update(kwargs)
        self.expt_path=expt_path
        self.prefix=prefix
        if config_file is None:
            self.config_file=None
        self.im = InstrumentManager()
        self.plotter = LivePlotClient()
        #self.dataserver= dataserver_client()
        self.fname = os.path.join(expt_path, get_next_filename(expt_path, prefix, suffix='.h5'))

        self.load_config()


    def load_config(self):
        if self.config_file is None:
            self.config_file = os.path.join(self.expt_path, self.prefix + ".json")

        try:
            self.cfg = AttrDict(json.load(open(self.config_file, 'r')))
        except:
            self.cfg = None

        if self.cfg is not None:
            for alias, inst in self.cfg['aliases'].iter_items():
                setattr(self, alias, expt.im['inst'])

    def datafile(self, group = None, remote=False):
        """returns a SlabFile instance
           proxy functionality not implemented yet"""
        f= SlabFile(self.fname)
        if group is None:
            return f
        else:
            return f.require_group(group)


