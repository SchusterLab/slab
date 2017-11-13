__author__ = 'Dave'

import ctypes as C
import numpy as np
import sys  # sys.exit(1), sys.stdout.write

CHASE_VERSION = "32bit"

try:
    if CHASE_VERSION == "32bit":

        chase_dll_path = r'C:\_Lib\python\slab\instruments\awg\chase\dax22000_lib_DLL32.dll'
    elif CHASE_VERSION == "64bit":
        chase_dll_path = r'C:\_Lib\python\slab\instruments\awg\chase\dax22000_lib_DLL64.dll'
    chaseDLL = C.CDLL(chase_dll_path)
except:
    print("Warning could not load Chase AWG dll, check that dll located at '%s'" % chase_dll_path)

addr = 1

# ------------------------------------------
# Get Number of Cards (exit if 0)
# ------------------------------------------
x = chaseDLL.DAx22000_GetNumCards()
if x == 0:
    print('No boards found.')
    sys.exit(1);
else:
    print('GetNumCards() = ' + str(x))

# raw_input("...")   #hit key to continue


#------------------------------------------
# Open DAx22000 API Driver
#------------------------------------------
x = chaseDLL.DAx22000_Open(C.c_int32(addr))

print('Open = ' + str(x))

if x != 0:
    print('Cannot open DAx API.')
    sys.exit(1);


#------------------------------------------
# Initialize Once
#------------------------------------------
print('Initialize = ' + str(chaseDLL.DAx22000_Initialize(C.c_int32(addr))))


#------------------------------------------
# Set Clock Rate
#------------------------------------------
print('DAx22000_SetClkRate = ' + \
      str(chaseDLL.DAx22000_SetClkRate(C.c_int32(addr), C.c_double(2e9))) + '\n')


#------------------------------------------
# Create Waveform Array and Download  (64 samples, external trigger, loop segment once each time)
#------------------------------------------

xpts = np.arange(8000)
ypts = np.zeros(len(xpts), dtype=np.uint16)
ypts += np.ceil(2047.5 + 2047.5 * np.sin(2.0 * np.pi * xpts / (32)))
print('Waveform Array' + '\n' + str(ypts) + '\n')

#print ypts

arr = (C.c_uint16 * len(ypts))(*ypts)

print('Array Size = ' + str(C.sizeof(arr)) + '\n')

# CardNum,          ChanNum,       NumPoints,     NumLoops,
chaseDLL.DAx22000_CreateSingleSegment(C.c_uint32(addr), C.c_uint32(1), C.c_uint32(len(ypts)), C.c_uint32(0),
                                      C.c_uint32(2047), C.c_uint32(2047), arr, C.c_uint32(1))

# If triggering a single waveform, then NumLoops should be set to 1. Using '0' will cause
# waveform to repeat continuously after first trigger.

#------------------------------------------
# Set MRK2, ExtTrig
#------------------------------------------

print(chaseDLL.DAx22000_Place_MRK2(C.c_int32(addr), C.c_int32(1)))
print(chaseDLL.DAx22000_SelExtTrig(C.c_int32(addr), C.c_bool(True)))

#------------------------------------------
# Multi-segment waveform
#------------------------------------------

print("multi-segment")

class SegmentStruct(C.Structure):
    """ Segment Structure
    """
    _fields_ = [('SegmentPtr', C.POINTER(C.c_uint16)),
                ('NumPoints', C.c_uint32),
                ('NumLoops', C.c_uint32),
                ('TrigEn', C.c_uint32)]


numsegs = 100

seglisttype = SegmentStruct * numsegs

seglist=seglisttype()

for ii in range(numsegs):
    ypts = np.zeros(len(xpts), dtype=np.uint16)
    ypts += xpts * 4095.0 / max(xpts) * ii / numsegs
    arr = (C.c_uint16 * len(ypts))(*ypts)
    seglist[ii].SegmentPtr = arr
    seglist[ii].NumPoints = len(ypts)
    seglist[ii].NumLoops = 1
    seglist[ii].TrigEn = 0

print(chaseDLL.DAx22000_CreateSegments(C.c_uint32(addr), C.c_uint32(1), C.c_uint32(numsegs), C.c_uint32(2047),
                                       C.c_uint32(2047), seglist, C.c_bool(True)))

print(chaseDLL.DAx22000_CreateSegments(C.c_uint32(addr), C.c_uint32(2), C.c_uint32(numsegs), C.c_uint32(2047),
                                       C.c_uint32(2047), seglist, C.c_bool(True)))

print(chaseDLL.DAx22000_Place_MRK2(C.c_int32(addr), C.c_int32(1)))
print(chaseDLL.DAx22000_SelExtTrig(C.c_int32(addr), C.c_bool(True)))


#------------------------------------------
# Run/Stop Output Waveform
#------------------------------------------

print(chaseDLL.DAx22000_Run(C.c_int32(addr), C.c_bool(True)))  # True will create (1) soft trigger.

#print chaseDLL.DAx22000_Stop(C.c_int32(addr))

#------------------------------------------
# Close DAx22000 API Driver
#------------------------------------------


#raw_input("...")   #hit key to continue

print(chaseDLL.DAx22000_Close(C.c_int32(addr)))  # No need to call until you close master program.
# Open/Close takes 200 msec.

