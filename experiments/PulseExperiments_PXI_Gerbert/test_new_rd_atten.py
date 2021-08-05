# from slab import InstrumentManager
# im = InstrumentManager()
# atten = im['readoutattengrb']
# atten.set_attenuator[20]

from slab.instruments import SerialInstrument, Instrument
from slab.instruments import DigitalAttenuator



a = DigitalAttenuator(name='readoutattengrb', address='COM4')
print("hi")
a.set_attenuator(1)

