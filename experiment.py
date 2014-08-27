__author__ = 'Nitrogen'

from liveplot import LivePlotClient
#from dataserver import dataserver_client
import os.path

from slab import SlabFile,InstrumentManager,get_next_filename

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

        #config file not implemented yet
        self.__dict__.update(kwargs)

        self.im = InstrumentManager()
        self.plotter = LivePlotClient()
        #self.dataserver= dataserver_client()
        self.fname = os.path.join(expt_path,get_next_filename(expt_path,prefix,suffix='.h5'))


    def datafile(self, remote=False):
        """returns a SlabFile instance
           proxy functionality not implemented yet"""
        return SlabFile(self.fname)
