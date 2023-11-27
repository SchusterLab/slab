
from .instrumentmanager import InstrumentManager
from .instrumenttypes import Instrument
try: from .qick_instrument import QickInstrument
except: print("Could not import QickInstrument (will only work if running on rfsoc). If running on rfsoc, try install/update qick package")

from .localinstruments import LocalInstruments
from .instrumenttypes import VisaInstrument, TelnetInstrument, SocketInstrument, SerialInstrument, WebInstrument
try: from .InstrumentManagerWindow import InstrumentManagerWindow
except: print("Could not load InstrumentManagerWindow")
try: from .spectrumanalyzer import E4440
except: print("Could not load E4440")
try: from .nwa import E5071
except: print("Could not load E5071")
try: from .PNAX import N5242A
except: print("Could not load N5242A")
try: from .RhodeSchwarz import RhodeSchwarz
except: print("Could not load RhodeSchwarz")
try: from .rfgenerators import N5183B,E8257D,BNC845
except: print("Could not load N5183B,E8257D,BNC845")
try: from .cryostat import Triton_old
except: print("Could not load Triton_old")
try: from .awg import *
except Exception as e: print(e)
try: from .spec_analyzer.spectrum_analyzer import *
except Exception as e: print(e)
try: from .ipsmagnet import *
except Exception as e: print(e)
try: from .TestInstruments import EchoInstrument,RandomInstrument
except Exception as e: print(e)
try: from .TDS7104 import TekTDS7104
except: print("Could not load TekTDS7104")
try: from .RCA18 import MCRCA18
except: print("Could not load MCRCA18")
try: from .multimeter import Keithley199
except: print("Could not load Keithley199")
from .minicircuits_rfswitch import MiniCircuitsSwitch
try: from .AD5780PARDAC.DACInterface import AD5780_serial
except: print("Could not load AD5780 dac ")
try: from.SignalCore import SignalCore
except: print("Could not load SignalCore")
from .spectrumanalyzer import E4440
try: from .nwa import E5071, E5071_USB
except: print("Could not load E5071")
try: from .AD5780DAC.AD5780 import AD5780
except: print("Could not load AD5780 dac")
try: from .ArduinoADC import ArduinoADC
except: print("Could not load ArduinoADC")
try: from .labbrick.labbrick import LMS_get_device_info,LMS103,LPS802,LDA602
except: print("Could not load labbrick")
try: from .relaybox.relaybox import RelayBox
except: print("Could not load relaybox")
try: from .relaybox.heliummanifold import HeliumManifold
except: print("Could not load heliummanifold")
try: from .relaybox.RFSwitch import RFSwitch
except: print("Could not load heliummanifold")
try: from .bkpowersupply import BK9130A, BK9130B
except: print("Could not load BKPowerSupply")
try: from .KEPCOPowerSupply import KEPCOPowerSupply
except: print("Could not load KEPCOPowerSupply")
try: from .voltsource import SRS900
except: print("Could not load SRS900")
try: from .voltsource import YokogawaGS200
except: print("Could not load YokogawaGS200")
from .Alazar import Alazar, AlazarConfig, AlazarConstants
try: from .Alazar import Alazar, AlazarConfig, AlazarConstants
except: print("Could not load Alazar card")
try: from .function_generator import BiasDriver,FilamentDriver,BNCAWG
except: print("Could not load BNC AWG classes")
try: from .multimeter import Keithley199, HP34401A
except: print("Could not load Keithley199/HP34401A multimeter classes")
try: from .spectrumanalyzer import E4440
except: print("Could not load E4440 Spectrum Analyzer")
try: from .cryocon import Cryocon
except: print("Could not load Cryocon instrument driver")
try: from .DigitalAttenuator import DigitalAttenuator
except: print("Could not load Digital Attenuator driver")
try: from .HeaterSwitch import HeaterSwitch
except: print("Could not load Heater Switch Driver")
try: from .Omega16i import Omega16i
except: print("Could not load Omega 16i driver")
try: from .lockin import SR844
except: print("Could not load SR844 driver")
try: from .PressureGauge import PressureGauge
except: print("Could not load PressureGauge driver")
try: from .RGA100 import RGA100
except: print('Could not load SRS RGA100 driver')
try: from .AG850 import AG850
except: print("Could not load AG850 Driver")
try: from .Autonics import TM4
except: print("Could not load Autonics TM4 Driver")
try: from .Oerlikon import Center_Three
except: print("Could not load Oerlikon Center Three Driver")
try: from .TempScanner import HP34970A
except: print("Could not load HP34970A Driver")
try: from .PLC import FurnacePLC
except: print("Could not load Furnace PLC")
try: from .Triton import Triton
except: print("Could not load Oxford Trition driver")
try: from.OmniYigController import OmniYigController
except: print("Could not load OmniYig controller")
try: from .Keithley import *
except: print("Could not load Keithley")