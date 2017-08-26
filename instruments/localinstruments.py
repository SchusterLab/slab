from slab.instruments.awg.PXDAC4800 import PXDAC4800

class LocalInstruments():

    # need a classmethod to generate these class variables inside init?

    inst_dict = {}
    # inst_dict['pxdac4800_1'] = PXDAC4800(1)
    #inst_dict['pxdac4800_2'] = PXDAC4800(2)
    #inst_dict['pxdac4800_3'] = PXDAC4800(3)

    def __init__(self):
        #self.test()
        pass

    # @classmethod
    # def test(cls):
    #     cls.inst_dict['pxdac4800_2'] = PXDAC4800(2)
    #     cls.inst_dict['pxdac4800_3'] = PXDAC4800(3)

    # note:
    # class method works for initialization, but errors out in
    # prep_function when start/stopping pxdacs

