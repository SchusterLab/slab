from scipy.interpolate import interp1d
from math import pi,sqrt
from .MaskMaker import CPWInductiveShunt

length_table = [12, 18, 26, 38, 50, 60, 75, 88, 100, 110]           #inductor length
L_table = [1.87, 2.75, 3.98, 5.49, 7.63, 8.58, 11.2, 12.7, 15.4, 16.0]  #inductance (pH)

def inductor_length(inductance):
    f=interp1d (L_table,length_table)     #function length = inductor_length (inductance) which gives length for input inductance
    return float(f(inductance*1e12))

def shunt_ext_Q (inductance,frequency, Z0 =50):
    return (Z0/(2*pi*frequency*1e9*inductance))**2

def shunt_inductance_by_Q (frequency,Q, Z0=50):
    return (Z0/(2*pi*frequency*1e9))/sqrt(Q)

def shunt_by_Q(frequency, Q, Z0=50):
    inductance = shunt_inductance_by_Q (frequency,Q,Z0)
    #print inductance
    length=float(inductor_length(inductance))
    #print length
    segment_gap=4.
    segment_width=3.
    shunt=CPWInductiveShunt(num_segments = 2 , segment_length = (length-3*segment_gap)/2.0, segment_width = segment_width, segment_gap=segment_gap, taper_length = 50, inductance = inductance)
    return shunt
