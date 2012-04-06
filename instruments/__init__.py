from instrumenttypes import Instrument,VisaInstrument,TelnetInstrument,SocketInstrument,SerialInstrument
from nwa import E5071
from rfgenerators import E8257D
from labbrick.labbrick import LMS_get_device_info,LMS103,LabbrickWindow
from cryostat import Triton
from instrumentmanager import InstrumentManager
from awg import *
from spec_analyzer.spectrum_analyzer import *
from ipsmagnet import *
from relaybox.relaybox import RelayBox
from relaybox.heliummanifold import HeliumManifold
from bkpowersupply import BKPowerSupply
from KEPCOPowerSupply import KEPCOPowerSupply

from Alazar import Alazar, AlazarConfig, AlazarConstants

instrument_dict = { "Network Analyzer" : E5071,
                    "Radio Frequency Generator" : E8257D,
                    "Lab Brick" : LMS103,
                    "Cryostat" : Triton,
                    "Arbitrary Waveform Generator" : AWG81180A,
                    "Spectrum Analyzer" : SpectrumAnalyzer,
                    "IPS Magnet" : IPSMagnet,
                    "BK": BKPowerSupply,
                    "KEPCO": KEPCOPowerSupply}
