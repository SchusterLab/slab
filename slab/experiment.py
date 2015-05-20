__author__ = 'Nitrogen'

from liveplot import LivePlotClient
# from dataserver import dataserver_client
import os.path
import json

from slab import SlabFile, InstrumentManager, get_next_filename


class Experiment:
    """Base class for all experiments"""

    def __init__(self, path='', prefix='data', config_file=None, **kwargs):
        """ Initializes experiment class
            @param path - directory where data will be stored
            @param prefix - prefix to use when creating data files
            @param config_file - parameters for config file specified are loaded into the class dict
                                 (name relative to expt_directory if no leading /)
                                 Default = None looks for path/prefix.json

            @param **kwargs - by default kwargs are updated to class dict

            also loads InstrumentManager, LivePlotter, and other helpers
        """

        self.__dict__.update(kwargs)
        self.path = path
        self.prefix = prefix
        if config_file is not None:
            self.config_file = os.path.join(path,config_file)
        else:
            self.config_file = None
        self.im = InstrumentManager()
        self.plotter = LivePlotClient()
        # self.dataserver= dataserver_client()
        self.fname = os.path.join(path, get_next_filename(path, prefix, suffix='.h5'))

        self.load_config()

    def load_config(self):
        if self.config_file is None:
            self.config_file = os.path.join(self.path, self.prefix + ".json")

        if self.config_file[:-3] == '.h5':
            with SlabFile(self.config_file) as f: cfg_str=f['config']
        else:
            with open(self.config_file, 'r') as fid: cfg_str = fid.read()

        self.cfg=json.loads(cfg_str)

        if self.cfg is not None:
            for alias, inst in self.cfg['aliases'].iteritems():
                setattr(self, alias, self.im[inst])

    def save_config(self):
        if self.config_file[:-3] != '.h5':
            with open(self.config_file, 'w') as fid:
                json.dump(self.cfg, fid)

    def datafile(self, group=None, remote=False):
        """returns a SlabFile instance
           proxy functionality not implemented yet"""
        f = SlabFile(self.fname)
        if group is not None:
            f = f.require_group(group)
        if 'config' not in f.keys():
            f['config'] = json.dumps(self.cfg)

    def go(self):
        pass


