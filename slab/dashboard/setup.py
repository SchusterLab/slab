from distutils.core import setup
import py2exe
import os
import zmq

os.environ["PATH"] =  \
    os.environ["PATH"] + \
    os.path.pathsep + os.path.split(zmq.__file__)[0]

windows = [{'script':'dashboard.py', 'icon_resources':[(0, 'plot.ico')]}]
#setup(console=['launcher.py'], options={'py2exe':{'includes':['sip']}})
setup(windows=windows, options={'py2exe':
                                    {'includes':
                                              ['sip', 'zmq.utils',
                                               'zmq.utils.jsonapi',
                                               'zmq.utils.strtypes',
                                               'scipy.sparse.csgraph._validation',
                                               'h5py.defs',
                                               'h5py.utils']}})