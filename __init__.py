from instruments import InstrumentManager
from datamanagement import SlabFile, h5File
try:
    from plotting import ScriptPlotter
except:
    "Warning could not import ScriptPlotter"
from dsfit import set_fit_plotting,argselectdomain,selectdomain,zipsort,fitgeneral, fitexp, fitgauss, fithanger, fithangertilt, fitlor, fitdecaysin,fithanger_new,hangerfunc_new
from dataanalysis import *
from matplotlib_text_wrapper import *
from experiment import Experiment
