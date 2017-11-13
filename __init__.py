from .instruments import InstrumentManager, LocalInstruments
from .datamanagement import SlabFile, h5File, AttrDict
#
# try:
#     from plotting import ScriptPlotter
# except:
#     "Warning could not import ScriptPlotter"

from .dsfit import set_fit_plotting, argselectdomain, selectdomain, zipsort, fitgeneral, fitexp, fitgauss, fithanger, \
    fithangertilt, fitlor, fitdecaysin, fithanger_new, hangerfunc_new, fit_SNT
from .dataanalysis import *
from .matplotlib_text_wrapper import *
from .experiment import Experiment
