from scipy import sqrt,pi,tanh,sinh
from scipy.special import ellipk
from scipy.optimize import newton, brentq
from scipy.interpolate import interp1d
from MaskMaker import ChannelFingerCap,CapDesc,CPWLCoupler,CPWGapCap,CPWFingerCap,CPWInductiveShunt,MaskError
from math import floor, ceil

mu0=1.25663706e-6
eps0=8.85418782e-12
speedoflight=299792458.0

def calculate_eps_eff_from_geometry(substrate_epsR,pinw,gapw,substrate_height):
    a=pinw
    b=pinw+2*gapw
    h=substrate_height
    k0 = float(a)/b
    k0p = sqrt(1-k0**2)
    #k3 = tanh(pi*a/(4*h))/  tanh(pi*b/(4*h))
    k3 = sinh(pi*a/(4*h)) / sinh(pi*b/(4*h))
    k3p= sqrt(1-k3**2)
    Ktwid= ellipk(k0p**2)*ellipk(k3**2)/(ellipk(k0**2)*ellipk(k3p**2))
    
    #return (1+substrate_epsR*Ktwid)/(1+Ktwid)
    return 1 + (substrate_epsR - 1) * Ktwid / 2

def calculate_eps_eff (phase_velocity):
    return (speedoflight/phase_velocity)**2

def calculate_impedance (pinw,gapw,eps_eff):
    #From Andreas' resonator paper or my thesis...agrees for values given in his paper
    k0 = float(pinw)/(pinw+2*gapw)
    k0p = sqrt(1-k0**2)
    L=(mu0/4)*ellipk(k0p**2)/ellipk(k0**2)
    C=4 *eps0*eps_eff*ellipk(k0**2)/ellipk(k0p**2)
    Z=sqrt(L/C)
    #print "pinw: %f, gapw: %f, k0: %f, k0p: %f, L: %f nH/m, C: %f pF/m, Z: %f" % (pinw,gapw,k0,k0p,L *1e9,C*1e12,Z)
    return Z

def calculate_resonator_frequency(length,eps_eff,impedance,resonator_type=0.5,harmonic=0,Ckin=None, Ckout=None):
    phase_velocity=speedoflight/sqrt(eps_eff)

    if (resonator_type==0.25): length_factor=0.25*(2*harmonic+1)
    else:                      length_factor=0.5*(harmonic+1)

    if Ckin is None:    in_cap=0.0
    else:               in_cap=Ckin.capacitance
    if Ckout is None:   out_cap=0.0
    else:               out_cap=Ckout.capacitance

    if (not Ckin is None) and (Ckin.type=='finger'):
        length+=0.4*Ckin.finger_length #subtract input finger length
    if (not Ckout is None) and (Ckout.type=='finger'):
        length+=0.4*+Ckout.finger_length #subtract output finger length
    
    #frequency=1e6*length_factor*phase_velocity/length
    #Csum=1/(2*pi*frequency*(harmonic+1)*impedance)
    #print "Csum= %f pF" % (Csum*1e12)
    #df=-frequency*(in_cap+out_cap)/(2*Csum)        #Calculate shift due to coupling capacitors
    #frequency+=df
    
    bf=1e6*length_factor*phase_velocity/length
    cap_factor=(in_cap+out_cap)*(1+harmonic)*impedance*pi
    if cap_factor==0: 
        frequency=bf
    else:
        frequency=(-1+sqrt(1+4*bf*cap_factor))/(2*cap_factor)
    
    return 1e-9 * frequency

def calculate_gap_width (eps_eff,impedance,pinw):
    f=lambda x: (calculate_impedance (pinw,x,eps_eff)-impedance)
    #return newton(f,pinw)
    return brentq(f, .2 * pinw, 5 * pinw)

def calculate_interior_length(frequency,phase_velocity,impedance,
                              resonator_type=0.5,harmonic=0,
                              Ckin=None, Ckout=None):
    """
        @param frequency:   frequency in GHz
    """
    
    #resonator type is 0.5 for lambda/2 and 0.25 for lambda/4
    #harmonic 0= fundamental
    frequency*=1e9
    
    if Ckin is None:    in_cap=0.0
    else:               in_cap=Ckin.capacitance
    if Ckout is None:   out_cap=0.0
    else:               out_cap=Ckout.capacitance
    
    #Todo: Not sure if this is 100% correct for both lambda/2 and lambda/4 may 
    #      require using something more like the length_factor to get it 
    #      right...the same for fundamentals though    
    Csum=1/(2*pi*frequency*(harmonic+1)*impedance)
    df=-frequency*(in_cap+out_cap)/(2*Csum)        #Calculate shift due to coupling capacitors

    if (resonator_type==0.25): length_factor=0.25*(2*harmonic+1)
    else:                      length_factor=0.5*(harmonic+1)
    
    #length=1e6*length_factor*phase_velocity/(frequency-df)                      #Calculate total length to get shifted frequency
    #fixed by PCR and LSB, but only checked for lambda/2
    length=1e6*(length_factor*phase_velocity/frequency)*(1-(frequency/(harmonic+1))*impedance*2*(in_cap+out_cap))
    if (not Ckin is None) and (Ckin.type=='finger'):
        length-=0.4*Ckin.finger_length #subtract input finger length
    if (not Ckout is None) and (Ckout.type=='finger'):
        length-=0.4*+Ckout.finger_length #subtract output finger length
    
    if (Ckin is not None and Ckout is not None):

        length -= Ckin.taper_length + Ckout.taper_length

    return length

def calculate_resonator_Q(frequency,impedance=50,Ckin=None, Ckout=None):
    
    if Ckin is None:  in_cap=0.
    else:             in_cap=Ckin.capacitance
    if Ckout is None: out_cap=0.
    else:             out_cap=Ckout.capacitance
    
    frequency=frequency*1e9
    qin=2.*pi*frequency*in_cap*impedance
    qout=2.*pi*frequency*out_cap*impedance
    Q=0
    if qin!=0:  
        Qin=pi/2*1/(qin**2)
        Q=Qin
    if qout!=0: 
        Qout=pi/2*1/(qout**2)
        Q=Qout
    if qout!=0 and qin!=0: 
        Q=Qin*Qout/(Qin+Qout)
    return Q

def capacitance_tables():
    finger_lengths = [10,20,30,40,50,60,70,80,90]
    caps_343_2F = [0.37612,0.652573,0.95197,1.2361,1.5183,1.80932,2.11591,2.44097,2.7851,3.15356]
    caps_343_4F = [1.82166,2.65303,3.79107,4.91664,6.05619,7.20684,8.36167,9.51948,10.6749,11.8352]
    caps_343_6F = [3.16549, 4.68083, 6.52342,8.44954, 10.4079,12.3822,14.3639,16.3536,18.3467,20.3426]
    caps_343_8F = [4.51186, 6.65667, 9.24498, 11.9754, 14.7498, 17.5535, 20.3704, 23.1956, 26.0261, 28.861]
    
    #caps_ABC_NF[i] corresponds to CPWFingerCap(num_fingers = N,finger_length = finger_lengths[i],finger_width = A,finger_gap = B ,taper_length = 50, gapw= 4.186 *(n*A+(n-1)*B)/10.0)
    
def capacitance_by_Q(frequency,Q, impedance=50,resonator_type=0.5):
    """Returns capacitance (in fF) that will give desired Q"""
    return 1/(2.*pi*frequency*1e9*sqrt(Q*pi*resonator_type)*impedance)

def sapphire_capacitor_by_Q(frequency,Q,impedance=50,resonator_type=0.5):
    """Calculates capacitance for desired Q and returns a CPWFingerCapacitor 
    with the appropriate geometry to yield the desired Q"""
    return sapphire_capacitor_by_C(capacitance_by_Q(frequency,Q,impedance,resonator_type))

def sapphire_capacitor_by_C(capacitance, taper_length=50):
    """def sapphire_capacitor_by_C(capacitance):
    Interpolates simulated capacitance tables to get specified capacitance values
    Simulations done in sonnet by Leo 
    Used eps_perp =9.27, eps_parallel = 11.34
    
    """

    finger_lengths = [10,20,30,40,50,60,70,80,90,100]
    caps_343_2F = [0.37612,0.652573,0.95197,1.2361,1.5183,1.80932,2.11591,2.44097,2.7851,3.15356]
    caps_343_4F = [1.82166,2.65303,3.79107,4.91664,6.05619,7.20684,8.36167,9.51948,10.6749,11.8352]
    caps_343_6F = [3.16549, 4.68083, 6.52342,8.44954, 10.4079,12.3822,14.3639,16.3536,18.3467,20.3426]
    caps_343_8F = [4.51186, 6.65667, 9.24498, 11.9754, 14.7498, 17.5535, 20.3704, 23.1956, 26.0261, 28.861]
    
    capacitance*=1e15
    #select table
    if capacitance<=3.0:
        num_fingers=2
        cap_table=caps_343_2F
    if capacitance>3.0 and capacitance <= 10.0:
        num_fingers=4
        cap_table=caps_343_4F
    if capacitance>10.0 and capacitance <= 20.0:
        num_fingers=6
        cap_table=caps_343_6F 
    if capacitance>20.0 and capacitance <= 28.0:
        num_fingers=8
        cap_table=caps_343_8F 
    if capacitance>28.0: 
        raise MaskError, "Error do not have simulated capacitors bigger than 28 fF, must specify geometry manually"
    
    get_length=interp1d (cap_table,finger_lengths)
    
    length=round(float(get_length(capacitance)))
    print "Capacitance: %f, Fingers: %d, Finger Length: %f " % (capacitance, num_fingers,length)
    return CPWFingerCap(num_fingers=num_fingers,finger_length=length,finger_width=3,finger_gap=4,taper_length = taper_length, capacitance=1e-15*capacitance)

#-------------------------------------------------------------------------------------------------------------
# CHANNEL CAPACITORS e on He
#-------------------------------------------------------------------------------------------------------------
def sapphire_capacitor_by_Q_Channels(frequency,Q,impedance=50,resonator_type=0.5,channelw=2):
    """Calculates capacitance for desired Q and returns a CPWFingerCapacitor 
    with the appropriate geometry to yield the desired Q"""
    return sapphire_capacitor_by_C_Channels(capacitance_by_Q(frequency,Q,impedance,resonator_type))

def sapphire_capacitor_by_C_Channels(capacitance,channelw=2):
    """def sapphire_capacitor_by_C(capacitance):
    Interpolates simulated capacitance tables to get specified capacitance values
    Simulations done in sonnet by Andy
    those capacitors have smaller dimensions that the ones we typically use and the gap size is smaller
    Used eps_perp =9.27, eps_parallel = 11.34
    
    """
    finger_lengths = [10,20,30,40,50,60,70,80,90,100]
    caps_2F = [0.37612,0.652573,0.95197,1.2361,1.5183,1.80932,2.11591,2.44097,2.7851,3.15356]
    
    capacitance*=1e15
    if capacitance<=3.0:
        num_fingers=2
        cap_table=caps_2F
    if capacitance>3.0: 
        raise MaskError, "Error do not have simulated capacitors bigger than 4 fF, must specify geometry manually"
    
    get_length=interp1d (cap_table,finger_lengths)
    
    length=round(float(get_length(capacitance)))
    print "Capacitance: %f, Fingers: %d, Finger Length: %f " % (capacitance, num_fingers,length)
    return ChannelFingerCap(num_fingers=num_fingers,finger_length=length,finger_width=2,finger_gap=2,taper_length = 50,channelw=channelw,capacitance=1e-15*capacitance)

#-------------------------------------------------------------------------------------------------------------
def inductor_length(inductance):
    length_table = [12, 18, 26, 38, 50, 60, 75, 88, 100, 110,120,130,140]           #inductor length
    L_table = [1.87, 2.75, 3.98, 5.49, 7.63, 8.58, 11.2, 12.7, 15.4, 16.0,17.6,18.9,20.47]  #inductance (pH)
    
    f=interp1d (L_table,length_table)     #function length = inductor_length (inductance) which gives length for input inductance
    try:
        return float(f(inductance*1e12))
    except:
        raise ValueError("inductance"+str(inductance)+"is out of range of simulated values")
    
def shunt_ext_Q (inductance,frequency, Z0 =50,resonator_type=0.5):
        q=2.*pi*frequency*self.capacitance*impedance
        Q=0
        if q!=0:
            Q=1/(resonator_type*pi) *1/ (q**2)
        return Q

def shunt_inductance_by_Q (frequency,Q, Z0=50,resonator_type=0.5):
    return Z0/(2.*pi*frequency*1e9*sqrt(Q*pi*resonator_type))

def shunt_by_Q(frequency, Q, Z0=50,resonator_type=0.5):
    inductance = shunt_inductance_by_Q (frequency,Q,Z0,resonator_type)
    #print inductance
    length=round(float(inductor_length(inductance)))
    min_seg_length=20
    segment_gap=4.
    segment_width=3.
    min_seg_length=20
    max_seg_length=100
    

    max_segments=int(floor((length-segment_gap)/(segment_gap+min_seg_length)))
    if max_segments == 0:
        num_segments=0
        segment_length=length
        segment_gap=10
    else:
        min_segments=int(ceil((length-segment_gap)/(segment_gap+max_seg_length)))
        num_segments=min_segments
        segment_length= (length-(num_segments+1)*segment_gap)/num_segments
    
    shunt=CPWInductiveShunt(num_segments = num_segments , segment_length =  segment_length, segment_width = segment_width, segment_gap=segment_gap, taper_length = 50, inductance = inductance)
    return shunt



if __name__=="__main__":
    phase_velocity=speedoflight/sqrt(5.7559)
    eps_eff=calculate_eps_eff(phase_velocity)
    print eps_eff
    eps_eff1=calculate_eps_eff_from_geometry(10.45,10,4.186,500)
    eps_eff2=calculate_eps_eff_from_geometry(10.45/1.057,10,4.186,500)
    print "eps_eff1: %f, eps_eff2: %f, ratio: %f, shift: %f GHz" % (eps_eff1,eps_eff2,eps_eff1/eps_eff2,5*(1-sqrt(eps_eff1/eps_eff2)))
    
    
    #print "Estimated eps_eff from geometry is %f" % calculate_eps_eff_from_geometry(9.8,pinw=10,gapw=4.186,substrate_height=500)
        
    print calculate_impedance(10,4.186,5.7559)
    w=calculate_gap_width(eps_eff,50.,10.)
    print "eps_eff: %f, pinw: %f, gapw: %f, Z: %f" % (eps_eff,10,w,calculate_impedance(10.,w,eps_eff))
    
    finger_cap=CPWFingerCap(4,100,2,2,capacitance=.5e-15)
    print "Resonator with finger capacitance=%f fF, has Q=%f" % (finger_cap.capacitance*1e15,calculate_resonator_Q(7.,Ckin=finger_cap))
    finger_cap=CPWFingerCap(4,100,2,2,capacitance=1e-15)
    print "Resonator with finger capacitance=%f fF, has Q=%f" % (finger_cap.capacitance*1e15,calculate_resonator_Q(7.,Ckin=finger_cap))
    finger_cap=CPWFingerCap(4,100,2,2,capacitance=2e-15)
    print "Resonator with finger capacitance=%f fF, has Q=%f" % (finger_cap.capacitance*1e15,calculate_resonator_Q(7.,Ckin=finger_cap))
    finger_cap=CPWFingerCap(4,100,2,2,capacitance=5e-15)
    print "Resonator with finger capacitance=%f fF, has Q=%f" % (finger_cap.capacitance*1e15,calculate_resonator_Q(7.,Ckin=finger_cap))
    finger_cap=CPWFingerCap(4,100,2,2,capacitance=10e-15)
    print "Resonator with finger capacitance=%f fF, has Q=%f" % (finger_cap.capacitance*1e15,calculate_resonator_Q(7.,Ckin=finger_cap))
    finger_cap=CPWFingerCap(4,100,2,2,capacitance=15.4e-15)
    print "Resonator with finger capacitance=%f fF, has Q=%f" % (finger_cap.capacitance*1e15,calculate_resonator_Q(7.,Ckin=finger_cap))
    finger_cap=CPWFingerCap(4,100,2,2,capacitance=30e-15)
    print "Resonator with finger capacitance=%f fF, has Q=%f" % (finger_cap.capacitance*1e15,calculate_resonator_Q(7.,Ckin=finger_cap))
    
    #Ckin_desc=CapDesc(capacitance=0.44e-15,num_fingers=0,finger_length=100.,finger_width=2,cap_gap=2,gapw=w)
    #Ckout_desc=CapDesc(capacitance=0.44e-15,num_fingers=0,finger_length=100.,finger_width=2,cap_gap=2,gapw=w)
    #coupler=CPWLCoupler(coupler_length=250,separation=30)
    cap=CPWGapCap(1)
    print "Interior Length: %f mm" % (1e-3*calculate_interior_length(4.8,phase_velocity,50.,resonator_type=0.25,harmonic=0,Ckin=cap))
   # print "lambda/4 for 11.5 GHz: %f" % calculate_interior_length(11.5e9,phase_velocity,50.,resonator_type=0.25,harmonic=0,Ckin_desc=Ckin_desc,Ckout_desc=Ckout_desc)