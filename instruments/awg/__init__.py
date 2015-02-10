# -*- coding: utf-8 -*-

try:
    from AWG81180 import AWG81180A
except: print "Warning failed to import AWG81180A"
try:
    from Tek5014 import Tek5014, Tek5014Sequence
except: print "Warning failed to import Tek5014"
try:
    from Tek70001 import Tek70001
except: print "Warning failed to import Tek70001"

import awgpulses