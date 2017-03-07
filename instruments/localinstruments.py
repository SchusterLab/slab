from slab.instruments.awg.PXDAC4800 import PXDAC4800

class LocalInstruments():
    inst_dict = {}
    inst_dict['pxdac4800_1'] = PXDAC4800(1)
    inst_dict['pxdac4800_2'] = PXDAC4800(2)

    def __init__(self):
        pass
