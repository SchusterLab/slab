from instrumenttypes import Instrument,VisaInstrument,TelnetInstrument,SocketInstrument,SerialInstrument
from nwa import E5071
from rfgenerators import E8257D
from labbrick.labbrick import LMS_get_device_info,LMS103,LabbrickWindow
from cryostat import Triton
from instrumentmanager import InstrumentManager
from awg import *
from spec_analyzer.spectrum_analyzer import *
from ipsmagnet import *

try: from relaybox.relaybox import RelayBox
except: print "Could not load relaybox"
try: from relaybox.heliummanifold import HeliumManifold
except: print "Could not load heliummanifold"
try: from relaybox.rfswitchcontroller import RFSwitch
except: print "Could not load heliummanifold"
try: from bkpowersupply import BKPowerSupply
except: print "Could not load BKPowerSupply"
try: from KEPCOPowerSupply import KEPCOPowerSupply
except: print "Could not load KEPCOPowerSupply"
try: from voltsource import SRS900
except: print "Could not load SRS900"
try: from Alazar import Alazar, AlazarConfig, AlazarConstants
except: print "Could not load Alazar card"