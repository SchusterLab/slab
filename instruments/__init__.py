from instrumenttypes import Instrument, VisaInstrument, TelnetInstrument, SocketInstrument, SerialInstrument, WebInstrument
from instrumentmanager import InstrumentManager
try: from InstrumentManagerWindow import InstrumentManagerWindow
except: print "Could not load InstrumentManagerWindow"

from spectrumanalyzer import E4440
from nwa import E5071
from rfgenerators import E8257D,BNC845
from cryostat import Triton
from awg import *
from spec_analyzer.spectrum_analyzer import *
from ipsmagnet import *
from TestInstruments import EchoInstrument,RandomInstrument

try: from labbrick.labbrick import LMS_get_device_info,LMS103,LPS802,LabbrickWindow,LDA602
except: print "Could not load labbrick"
try: from relaybox.relaybox import RelayBox
except: print "Could not load relaybox"
try: from relaybox.heliummanifold import HeliumManifold
except: print "Could not load heliummanifold"
try: from relaybox.RFSwitch import RFSwitch
except: print "Could not load heliummanifold"
try: from bkpowersupply import BKPowerSupply
except: print "Could not load BKPowerSupply"
try: from KEPCOPowerSupply import KEPCOPowerSupply
except: print "Could not load KEPCOPowerSupply"
try: from voltsource import SRS900
except: print "Could not load SRS900"
try: from voltsource import YokogawaGS200
except: print "Could not load YokogawaGS200"
try: from Alazar import Alazar, AlazarConfig, AlazarConstants
except: print "Could not load Alazar card"
try: from function_generator import BiasDriver,FilamentDriver,BNCAWG
except: print "Could not load BNC AWG classes"
try: from Keithley199 import Keithley199
except: print "Could not load Keithley199 voltmeter classes"
try: from spectrumanalyzer import E4440
except: print "Could not load E4440 Spectrum Analyzer"
try: from cryocon import Cryocon
except: print "Could not load Cryocon instrument driver"
try: from DigitalAttenuator import DigitalAttenuator
except: print "Could not load Digital Attenuator driver"
try: from HeaterSwitch import HeaterSwitch
except: print "Could not load Heater Switch Driver"
try: from Omega16i import Omega16i
except: print "Could not load Omega 16i driver"
try: from lockin import SR844
except: print "Could not load SR844 driver"
