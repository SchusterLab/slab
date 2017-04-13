from slab.instruments.awg.PXDAC4800 import PXDAC4800

class LocalInstruments():
    inst_dict = {}
    inst_dict['pxdac4800'] = PXDAC4800()

    def __init__(self):
        pass