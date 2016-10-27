# -*- coding: utf-8 -*-
"""
Created on Wed Aug 01 09:44:59 2012

@author: slab
"""

from slab.circuits import *

class TestTwoLayer(Chip):
    def __init__(self):
        Chip.__init__(self, name="A", two_layer=True, solid=True)
        d = {'pinw':10, 'gapw':10, 'radius':25}
        s = Structure(self, start=self.left_midpt, defaults=d)
        Launcher(s),
        CPWStraight(s, 400)
        cap = sapphire_capacitor_by_Q(10, 1000)
        cap.draw(s)
        CPWBend(s, 180, radius=200)
        cap.draw(s)
        #CPWFingerCap(6, 80, 5, 5, taper_length=40).draw(s)
        CPWStraight(s, 400) 
        Launcher(s, flipped=True)

if __name__ == "__main__":
    m = WaferMask("TL")
    c = TestTwoLayer()
    m.add_chip(c, 5)
    m.save()