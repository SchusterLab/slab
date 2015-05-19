# -*- coding: utf-8 -*-

try:
    from AWG81180 import AWG81180A
except: print "Warning failed to import AWG81180A"
try:
    from Tek5014 import Tek5014, Tek5014Sequence, write_Tek5014_file
except: print "Warning failed to import Tek5014"
try:
    from Tek70001 import Tek70001, write_Tek70001_waveform_file
except: print "Warning failed to import Tek70001"

import awgpulses,awgpulses2

from PulseSequence import PulseSequence,PulseSequenceArray

import StandardPulseSequences