from slab import *
from slab.instruments import *
im = InstrumentManager()

atten = 'driveattengrbB'
d = im[atten]
d.set_attenuator(0)


