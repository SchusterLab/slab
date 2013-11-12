# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:35:36 2012

@author: slab
"""

from slab.circuits.MaskMaker import *
from slab.circuits.ResonatorCalculations import *
from numpy import zeros
from numpy import ones

eps0 = 8.85e-12
mu0 = 1.26e-6
eps_eff = 5.5 # * 1.235 # adjustment for inferred dielectric
phase_velocity = speedoflight/sqrt(eps_eff)
global_defaults = {'pinw':10, 'gapw':calculate_gap_width(5.5, 50, 10.), 'radius':40}

# See this SO discussion 
# http://stackoverflow.com/questions/3652851/what-is-the-best-way-to-do-automatic-attribute-assignment-in-python-and-is-it-a
import inspect
import functools
def autoargs(*include,**kwargs):   
    def _autoargs(func):
        attrs,varargs,varkw,defaults=inspect.getargspec(func)
        def sieve(attr):
            if kwargs and attr in kwargs['exclude']: return False
            if not include or attr in include: return True
            else: return False            
        @functools.wraps(func)
        def wrapper(self,*args,**kwargs):
            # handle default values
            for attr,val in zip(reversed(attrs),reversed(defaults)):
                if sieve(attr): setattr(self, attr, val)
            # handle positional arguments
            positional_attrs=attrs[1:]            
            for attr,val in zip(positional_attrs,args):
                if sieve(attr): setattr(self, attr, val)
            # handle varargs
            if varargs:
                remaining_args=args[len(positional_attrs):]
                if sieve(varargs): setattr(self, varargs, remaining_args)                
            # handle varkw
            if kwargs:
                for attr,val in kwargs.iteritems():
                    if sieve(attr): setattr(self,attr,val)            
            return func(self,*args,**kwargs)
        return wrapper
    return _autoargs

class CPWQubitBox2:
    """A straight section of CPW transmission line with fingers in the ground plane to add a capacitor"""
    @autoargs()
    def __init__(self, c_gap, finger_no_left, finger_no_right,
                 outer_finger_len_left, outer_finger_len_right,
                 inner_finger_len_left, inner_finger_len_right,
                 taper_len=0, int_len=30, pinw=None, gapw=None, align=False):
        
        finger_gapw = self.c_gap
        fingerw = 3 * self.c_gap        
        self.left_finger_section_length = (finger_no_left * (fingerw + finger_gapw)) - finger_gapw
        self.right_finger_section_length = (finger_no_right * (fingerw + finger_gapw)) - finger_gapw
        self.length = int_len + self.left_finger_section_length + self.right_finger_section_length

    def draw(self, s, flipped=False, qubit=False):
        if qubit:
            offset = rotate_pt((self.c_gap, 0), s.last_direction)
            new_start = s.last[0] + offset[0], s.last[1] + offset[1]
            qubit_struct = Structure(s.chip, start=new_start, layer='Qubit', color=5)
            
        gapw = self.gapw if self.gapw else s.defaults["gapw"]
        pinw = self.pinw if self.pinw else s.defaults["pinw"]        
        finger_gapw = self.c_gap
        center_gapw = fingerw = 3 * self.c_gap
        center_pinw_left = 2 * self.inner_finger_len_left + pinw
        center_pinw_right = 2 * self.inner_finger_len_right + pinw
        self.center_width = max(center_pinw_left, center_pinw_right) + (2*center_gapw)
        finger_no_left, finger_no_right = self.finger_no_left, self.finger_no_right
        outer_finger_len_left, outer_finger_len_right = self.outer_finger_len_left, self.outer_finger_len_right
        inner_finger_len_left, inner_finger_len_right = self.inner_finger_len_left, self.inner_finger_len_right
        if flipped:
            center_pinw_left, center_pinw_right = center_pinw_right, center_pinw_left
            finger_no_left, finger_no_right = finger_no_right, finger_no_left
            outer_finger_len_left, outer_finger_len_right = outer_finger_len_right, outer_finger_len_left
            inner_finger_len_left, inner_finger_len_right = inner_finger_len_right, inner_finger_len_left
        
        CPWLinearTaper(s, self.taper_len, pinw, center_pinw_left, gapw, center_gapw)
        if qubit:
            CPWInnerOuterFingerIsland(s, self.c_gap, finger_no_left,
                                      inner_finger_len_left, outer_finger_len_left)
            CPWStraight(s, self.int_len/2., self.c_gap, (self.center_width-self.c_gap)/2.)
            self.center_pt = s.last
            Channel(s, self.c_gap, self.center_width)
            CPWStraight(s, self.int_len/2., self.c_gap, (self.center_width-self.c_gap)/2.)
            CPWInnerOuterFingerIsland(s, self.c_gap, finger_no_right,
                                      inner_finger_len_right, outer_finger_len_right, flipped=True)    
        else:
            CPWInnerOuterFingers(s, center_pinw_left, center_gapw, finger_no_left, fingerw, 
                                 finger_gapw, inner_finger_len_left, outer_finger_len_left)
        
            Channel(s, self.int_len/2., self.center_width)
            self.center_pt = s.last
            Channel(s, self.int_len/2., self.center_width)
            CPWInnerOuterFingers(s, center_pinw_right, center_gapw, finger_no_right, fingerw, 
                                 finger_gapw, inner_finger_len_right, outer_finger_len_right)

        CPWLinearTaper(s, self.taper_len, center_pinw_right, pinw, center_gapw, gapw)
        
        if self.align:
            vgap = 25
            hgap = 0
            int_hgap = 10
            int_vgap = 10
            left_height = center_pinw_left + outer_finger_len_left
            right_height = center_pinw_right + outer_finger_len_right
            height = max(left_height, right_height)
            int_height = self.center_width/2.
            left_len = self.int_len/2. + finger_no_left*(fingerw + finger_gapw) - finger_gapw
            right_len = self.int_len/2. + finger_no_right*(fingerw + finger_gapw) - finger_gapw
            pts = []
            for lr, length in [(-1, left_len), (1, right_len)]:
                for tb in [1, -1]:
                    inner_pt = self.center_pt[0] + lr * ((self.int_len/2.) + hgap), self.center_pt[1] + tb * (height + vgap)
                    outer_pt = self.center_pt[0] + lr * (length - hgap), self.center_pt[1] + tb * (height + vgap)
                    interior_pt = self.center_pt[0] + lr * ((self.int_len/2.) - int_hgap), self.center_pt[1] + tb * (int_height + int_vgap)
                    pts.extend([inner_pt, outer_pt, interior_pt])
            for p in pts:
                self.draw_alignment_marker(s, p)
            for i in [0, 1, 3, 4]:                            
                self.draw_alignment_marker(s, (self.center_pt[0], i * s.chip.size[1]/4.), 30)
            
        return self.center_pt
    def draw_alignment_marker(self, s, pt, size=5):
        x, y = pt
        ds = size/2.
        s.append(sdxf.PolyLine([(x-ds, y-ds), (x+ds, y-ds), 
                                (x+ds, y+ds), (x-ds, y+ds), 
                                (x-ds, y-ds)]))

class CPWInnerOuterFingers:
    def __init__(self, s, start_pinw, start_gapw, n_fingers, finger_width,
                 gap_width, inner_finger_length, outer_finger_length):
        gap_pinw = start_pinw
        gap_gapw = start_gapw
        finger_pinw = start_pinw - (2*inner_finger_length)
        finger_gapw = start_gapw + inner_finger_length + outer_finger_length
        assert finger_pinw > 0
        for i in range(n_fingers):
            if i != 0 :
                CPWStraight(s, gap_width, pinw=gap_pinw, gapw=gap_gapw)
            CPWStraight(s, finger_width, pinw=finger_pinw, gapw=finger_gapw)
            
class CPWInnerOuterFingerIsland:
    def __init__(self, s, c_gap, n_fingers, inner_finger_length, outer_finger_length, flipped=False):
        pinw = s.defaults["pinw"]
        gapw = s.defaults["gapw"]
        # Initial Part
        if flipped:
            #CPWGapCap(c_gap, pinw, inner_finger_length+outer_finger_length+(5*c_gap)).draw(s)
            CPWStraight(s, c_gap, c_gap, ((pinw+c_gap)/2.)+inner_finger_length+outer_finger_length+(4*c_gap))
            CPWStraight(s, c_gap, pinw+(2*(inner_finger_length+outer_finger_length+(4*c_gap))), c_gap)
            start = s.last
            DoubleCPW(s, c_gap, pinw, inner_finger_length+(2*c_gap), c_gap, 
                      outer_finger_length+(2*c_gap))
            s.last = start
            CPWGapCap(c_gap, pinw, 0).draw(s)
        else:
            CPWStraight(s, c_gap, pinw, inner_finger_length+outer_finger_length+(5*c_gap))
            DoubleCPW(s, c_gap, pinw, c_gap, 
                      inner_finger_length+outer_finger_length+(3*c_gap), c_gap)
            DoubleCPW(s, c_gap, pinw, inner_finger_length+(2*c_gap), c_gap, 
                      outer_finger_length+(2*c_gap))
        # Middle Fingers
        for i in range(n_fingers-2):
            # gap bit
            DoubleCPW(s, c_gap, pinw+(2*(inner_finger_length+c_gap)), c_gap, c_gap, c_gap)
            # first bit
            DoubleCPW(s, c_gap, pinw, inner_finger_length+(2*c_gap), c_gap, 
                      outer_finger_length+(2*c_gap))
            # middle bit
            DoubleCPW(s, c_gap, pinw, c_gap, 
                      inner_finger_length+outer_finger_length+(3*c_gap), c_gap)
            # last bit == first bit
            DoubleCPW(s, c_gap, pinw, inner_finger_length+(2*c_gap), c_gap, 
                      outer_finger_length+(2*c_gap))

        # Last gap bit
        DoubleCPW(s, c_gap, pinw+(2*(inner_finger_length+c_gap)), c_gap, c_gap, c_gap)
        # Final part
        if flipped:
            DoubleCPW(s, c_gap, pinw, inner_finger_length+(2*c_gap), c_gap, 
                      outer_finger_length+(2*c_gap))            
            DoubleCPW(s, c_gap, pinw, c_gap, 
                      inner_finger_length+outer_finger_length+(3*c_gap), c_gap)
            CPWStraight(s, c_gap, pinw, inner_finger_length+outer_finger_length+(5*c_gap))
        else:            
            start = s.last
            DoubleCPW(s, c_gap, pinw, inner_finger_length+(2*c_gap), c_gap, 
                      outer_finger_length+(2*c_gap))
            s.last = start
            CPWGapCap(c_gap, pinw, 0).draw(s)
            CPWStraight(s, c_gap, pinw+(2*(inner_finger_length+outer_finger_length+(4*c_gap))), c_gap)
#            CPWGapCap(c_gap, pinw, inner_finger_length+outer_finger_length+(5*c_gap)).draw(s)
            CPWStraight(s, c_gap, c_gap, ((pinw+c_gap)/2.)+inner_finger_length+outer_finger_length+(4*c_gap))
            

class CPWRightJoint:
    def __init__(self, s, CCW=False, pinw=None, gapw=None):
        pinw = pinw if pinw else s.defaults["pinw"]
        gapw = gapw if gapw else s.defaults["gapw"]
        d = pinw/2.
        gap = gapw
        ext = 2*gapw + pinw
        if CCW:
            d *= -1
            gap *= -1
        inner = [(0,-d), (gapw, -d), (gapw, -(d+gap)), (0, -(d+gap)), (0,-d)]
        outer_1 = [(0, d), (ext, d), (ext, d+gap), (0, d+gap), (0, d)]
        outer_2 = [(ext-gapw, d), (ext-gapw, -(d+gap)), (ext, -(d+gap)), (ext, d), (ext-gapw, d)]
        for shape in [inner, outer_1, outer_2]:
            s.append(sdxf.PolyLine(orient_pts(shape, s.last_direction, s.last)))
        s.last = orient_pt((ext/2.,(1 if CCW else -1) * ext/2.), s.last_direction, s.last)
        if CCW:
            s.last_direction += 90
        else:
            s.last_direction -= 90
        
class RightJointWiggles:
    def __init__(self, s, total_length, num_wiggles, radius):
        pinw = s.defaults["pinw"]
        gapw = s.defaults["gapw"]
        cpwidth = pinw + 2*gapw
        hlength = (2*radius) - cpwidth
        #vlength = ((total_length - ((num_wiggles-1)*cpwidth))/ float(2*num_wiggles)) - hlength - (2*cpwidth)
        vlength = (total_length - (num_wiggles*hlength) - (((3*num_wiggles)+1)*cpwidth)) / (2*num_wiggles)

        assert hlength > 0 and vlength > 0
        
        tot_span = 0        
        
        CPWRightJoint(s,True)
        tot_span += cpwidth
        for ii in range(num_wiggles):
            CCW = (ii % 2) != 0
            CPWStraight(s,vlength,pinw,gapw)
            tot_span += vlength
            #CPWBend(s,isign*asign*180,pinw,gapw,radius, segments=segments)
            CPWRightJoint(s, CCW)
            tot_span += cpwidth
            CPWStraight(s, hlength)
            tot_span += hlength
            CPWRightJoint(s, CCW)            
            tot_span += cpwidth
            CPWStraight(s,vlength,pinw,gapw)
            tot_span += vlength
            if ii<num_wiggles-1:
                CPWStraight(s,cpwidth,pinw,gapw)
                tot_span += cpwidth
        CPWRightJoint(s, (not CCW))
        tot_span += cpwidth
        

class DoubleCPW:
    def __init__(self, s, length, inner_pin, inner_gap, outer_pin, outer_gap):
        start = s.last
        CPWStraight(s, length, pinw=inner_pin, gapw=inner_gap)
        s.last = start
        CPWStraight(s, length, pinw=inner_pin+(2*(inner_gap+outer_pin)), gapw=outer_gap)

def abs_rect(s, p0, p1):
    p01 = p0[0], p1[1]
    p10 = p1[0], p0[1]
    s.append(sdxf.PolyLine([p0, p01, p1, p10, p0]))

def rect(s, p0, p1):
    p01 = p0[0], p1[1]
    p10 = p1[0], p0[1]
    s.append(sdxf.PolyLine(orient_pts([p0, p01, p1, p10, p0], s.last_direction, s.last)))
    
def gds_channel(s, length, width, datatype=0, layer=0):
    w = width/2.
    gds_rect(s, (0, w), (length, -w), datatype, layer)
    #s.last = vadd(s.last, s.orient_pt((length, 0)))

'''
Calculations by DCM May 2013
Version 2 Lumped Resonator Design (two parallel C and series L)
These calculations assume cap fingers widths and gaps of 15um
and inductors with 5um widths and 20um gaps
''' 
def LumpedElementResonators_by_f_v2(s,fres,impedance, n_c_fingers, n_l_fingers, cin=0, cout=0, conn_length=20,flipped=False):
    
    #do a first order correction for capacitive loading
    fres = fres*(1+fres*2*pi*impedance*(cin+cout)/8)  
    
    print fres*2*pi*impedance*(cin+cout)/8
    
    lval = impedance/(fres*2*pi)/1e-9
    cval = lval/impedance**2/1e-6*2 #note that each cap is twice the LC cap
   
    print lval,cval   
   
    #Simulations done in designer with saphhire eps = 10.8
    #Mean of d/dw(im(Y(1,2))) 
    #Initially (first three lengths) from 1->3 GHz, but changed to 0.2->1GHz
    #Lengths 100,200 and 2 fingers are extrapolated
    c_sim_length = [100,200,300,400,500,600,700,800,900]
    c_sim_F = [2,4,6,8,10,12]
    c_sim = zeros((len(c_sim_F),len(c_sim_length)))
    c_sim[0] = [164./2,265./2,366./2,467./2,568./2,670./2,772./2,874./2,977./2]
    c_sim[1] = [164.,265.,366.,467.,568.,670.,772.,874.,977.]
    c_sim[2] = [229.,378.,527.,676.,824.,975.,1126.,1278.,1432.]
    c_sim[3] = [296.,493.,690.,887.,1084.,1285.,1487.,1691.,1898.]
    c_sim[4] = [358.,605.,852.,1099.,1345.,1597.,1853.,2112.,2373.]
    c_sim[5] = [425.,721.,1017.,1313.,1611.,1912.,2227.,2543.,2865.]
    
    #heuristic correction factor comparing to the resonator simulations
    c_sim *= (1.34+(0.10)*(fres-4.0e9)/6.0e9)*1.01
    
    print (1.34+(0.10)*(fres-4.0e9)/6.0e9)*1.01
  
    
    if not (n_c_fingers in c_sim_F):
        raise MaskError, "Incorrect specfication for the number of C fingers"
      
    cap_table = c_sim[(n_c_fingers-2)/2]
    
    if cap_table[0]>cval:
        raise MaskError, "Need fewer cap fingers"
        
    if cap_table[-1]<cval:
        raise MaskError, "Need more cap fingers"
    
    get_length=interp1d (cap_table,c_sim_length)
    c_length=round(float(get_length(cval)))
    
    #Simulations done in designer with saphhire eps = 10.8
    #Mean of d/dw(im(1/Y(1,2))) 
    #0.2->1GHz    
    l_sim_length = [300,400,500,600,700,800,900]
    l_sim_F = [1,2,3,4,5]
    l_sim = zeros((len(l_sim_F),len(l_sim_length)))
    l_sim[0] = [0.418,0.521,0.622,0.723,0.825,0.925,1.03]
    l_sim[1] = [0.704,0.895,1.08,1.27,1.46,1.65,1.84]
    l_sim[2] = [0.989,1.267,1.54,1.82,2.09,2.36,2.63]
    l_sim[3] = [1.273,1.637,1.99,2.35,2.71,3.05,3.41]
    l_sim[4] = [1.555,2.004,2.445,2.88,3.31,3.74,4.17]
    
    
    l_sim *= (0.89-(0.105)*(fres-4.0e9)/6.0e9)*1.01
    
    print (0.89-(0.105)*(fres-4.0e9)/6.0e9)*1.01
    
    
    if not (n_l_fingers in l_sim_F):
        raise MaskError, "Incorrect specfication for the number of L fingers"
         
    l_table = l_sim[n_l_fingers-1]
        
    if l_table[0]>lval:
        raise MaskError, "Need fewer inductor fingers"
        
    if l_table[-1]<lval:
        raise MaskError, "Need more inductor fingers"    
    
    #get the inductor length
    get_length=interp1d (l_table,l_sim_length)
    l_length=ceil(float(get_length(lval)))
    
    
    return LumpedElementResonatorv2(s, n_c_fingers, n_l_fingers, c_length, l_length, c_width=15, c_gap=15,
                           l_width=4, l_gap=20, conn_length=conn_length, flipped=flipped, describe=False)


'''
Calculations by DCM May 2013
These calculations assume cap fingers widths and gaps of 15um
and inductors with 4um widths and 20um gaps
''' 
def LumpedElementResonators_by_f(s,fres,impedance, n_c_fingers, n_l_fingers, v_offset=50,conn_length=20,flipped=False):
    
    lval = impedance/(fres*2*pi)/1e-9
    cval = lval/impedance**2/1e-6
    
    print cval, lval
    
    #Simulations done in designer with saphhire eps = 10.8
    #Mean of d/dw(im(Y(1,2))) 
    #Initially (first three lengths) from 1->3 GHz, but changed to 0.2->1GHz
    c_sim_length = [300,400,500,600,700,800,900]
    c_sim_F = [5,6,7,8,9]
    c_sim = zeros((len(c_sim_F),len(c_sim_length)))
    c_sim[0] = [152.,203.,255.,298.,337.,385.,433.]
    c_sim[1] = [188.,251.,316.,355.,414.,472.,533.]
    c_sim[2] = [224.,300.,380.,421.,491.,561.,633.]
    c_sim[3] = [261.,352.,447.,487.,568.,650.,735.]
    c_sim[4] = [280.,370.,462.,554.,647.,741.,838.]
    
    #heuristic correction factor comparing to the resonator simulations
    c_sim *= 1.1
    
    
    if not (n_c_fingers in c_sim_F):
        raise MaskError, "Incorrect specfication for the number of C fingers"
      
    cap_table = c_sim[n_c_fingers-4]
    
    if cap_table[0]>cval:
        raise MaskError, "Need fewer cap fingers"
        
    if cap_table[-1]<cval:
        raise MaskError, "Need more cap fingers"
    
    get_length=interp1d (cap_table,c_sim_length)
    c_length=round(float(get_length(cval)))
    
    #Simulations done in designer with saphhire eps = 10.8
    #Mean of d/dw(im(1/Y(1,2))) 
    #0.2->1GHz    
    l_sim_length = [300,400,500,600,700,800,900]
    l_sim_F = [1,2,3,4,5]
    l_sim = zeros((len(l_sim_F),len(l_sim_length)))
    l_sim[0] = [0.62,0.78,0.95,1.10,1.26,1.42,1.59]
    l_sim[1] = [0.93,1.18,1.42,1.67,1.92,2.18,2.43]
    l_sim[2] = [1.24,1.58,1.92,2.26,2.59,2.93,3.27]
    l_sim[3] = [1.55,1.97,2.4,2.83,3.24,3.68,4.1]
    l_sim[4] = [1.86,2.37,2.9,3.4,3.92,4.42,4.93]
    
    #heuristic correction factor comparing to the resonator simulations
    l_sim *= 1.1
    
    
    if not (n_l_fingers in l_sim_F):
        raise MaskError, "Incorrect specfication for the number of L fingers"
         
    l_table = l_sim[(n_l_fingers-10)/2]
    
    #get the inductor length
    get_length=interp1d (l_table,l_sim_length)
    l_length=ceil(float(get_length(lval)))
    
    
    return LumpedElementResonator(s, n_c_fingers, n_l_fingers, c_length, l_length, c_width=10, c_gap=10,
                           l_width=3, l_gap=20, v_offset=v_offset, conn_length=conn_length, flipped=flipped, describe=False)

'''
Draw the lumped element given specified parameters (new design)
Caps to ground and a meander inductor in the center
c_fingers: number of capacitor fingers
l_fingers: number of inductor wiggles (defined as one "S" shaped meander)
c_length: the capacitor finger length
l_length: the inductor length (the long edge of the "S")
c_width: width of the cap finger
c_gap: cap finger gap
l_width: width of the inductor meander line
l_gap: gap between meanders
v_offset: The resonator starts at the CPW and goes in one direction, this offset pushes the resonator start in the other direction
'''
def LumpedElementResonatorv2(s, c_fingers, l_fingers, c_length, l_length, c_width=3, c_gap=3,
                           l_width=5, l_gap=11, conn_length=20, flipped=False, describe=False):
    
    
    
    sign = -1 if flipped else 1    
    
    #start with a straight CPW section
    CPWStraight(s,conn_length)
    
    #create a cap to ground using "ground fingers"
    ground_cap = ground_fingers(c_fingers, c_length, c_width, align=False, finger_gap=c_gap)
    
    ground_cap(s,empty=False)    
    
    l_ground_gap = 25 #3*l_width    
    
    CPWStraight(s,c_gap+l_ground_gap)
    
    
    def myrect(s, p0, p1):
        x0, y0 = p0
        x1, y1 = p1
        p0 = x0, sign*y0
        p1 = x1, sign*y1
        rect(s, p0, p1)
    
    #Draw inductor region border 

    
    
    l, w, g = l_length, l_width, l_gap
    
    l_reg_width = 2*l_fingers*(w+g)+w    
    
    for i in [1,-1]:
        myrect(s,(-l_ground_gap,i*s.pinw/2),(0,i*(l/2+w+l_ground_gap)))
        myrect(s,(0,i*(l/2+w+l_ground_gap)),(l_reg_width,i*(l/2+w)))
        myrect(s,(l_reg_width,i*(l/2+w+l_ground_gap)),(l_reg_width+l_ground_gap,i*s.pinw/2))
   
       
    for i in range(l_fingers):
        myrect(s, (w, l/2), (w + g, -(l/2+w)))
        myrect(s, (2*w + g, -(l/2)), (2*w + 2*g, (l/2+w)))
        
        if i==0:
            myrect(s, (0, -s.pinw/2), (w, -(l/2+w)))
        
        if i==(l_fingers-1):
            myrect(s, (2*w + 2*g, s.pinw/2), (3*w + 2*g, (l/2+w)))
        
        s.last = orient_pt((2*g+2*w,0), s.last_direction, s.last)
    
    s.last = orient_pt((w,0), s.last_direction, s.last)
            
    
    #CPWStraight(s,100,pinw=0,gapw=100)
    CPWStraight(s,c_gap+l_ground_gap)
       
    
    #create a cap to ground using "ground fingers"
    ground_cap(s,empty=False)   
    
    CPWStraight(s,conn_length)
    
    return    
    
    

'''
Draw the lumped element given specified parameters
c_fingers: number of capacitor fingers
l_fingers: number of inductor wiggles (defined as one "S" shaped meander)
c_length: the capacitor finger length
l_length: the inductor length (the long edge of the "S")
c_width: width of the cap finger
c_gap: cap finger gap
l_width: width of the inductor meander line
l_gap: gap between meanders
v_offset: The resonator starts at the CPW and goes in one direction, this offset pushes the resonator start in the other direction
'''
def LumpedElementResonator(s, c_fingers, l_fingers, c_length, l_length, c_width=3, c_gap=3,
                           l_width=5, l_gap=11, v_offset=50, conn_length=20, flipped=False, describe=False):
    
    
    if v_offset < 0:
        raise MaskError, "Vertical offset of lumped resonator can only be positive"
    
    v_offset = v_offset + s.pinw/2+s.gapw-c_width 

    sign = -1 if flipped else 1    
    
    #start with a straight CPW section
    CPWStraight(s,conn_length)
    
    
    start = s.last
    
    #offset accounding to v_offset
    s.move(-1*sign*v_offset,s.last_direction+90)    
        
    #c_length, l_length = length, length + c_width + c_gap - l_width - l_gap
    tot_c_length =  c_length + 2*c_width + c_gap
    tot_l_length =  l_length+2*l_width+l_gap

    tot_width = max(tot_c_length,tot_l_length)
    tot_height = c_fingers*(2*c_width + 2*c_gap) + l_fingers*(2*l_width + 2*l_gap) + l_width
    line_width = s.pinw + 2*s.gapw
    v_offset = v_offset - line_width/2.
    w = c_width
    
    def myrect(s, p0, p1):
        x0, y0 = p0
        x1, y1 = p1
        p0 = x0, sign*y0
        p1 = x1, sign*y1
        rect(s, p0, p1)
        
    #Draw lumped element perimeter region
    #Draw box on the bottom side of the CPW
    myrect(s, (-w, -w), (tot_width+w, 0))                                     # Bottom
    myrect(s, (-w, -w), (0, v_offset))                                        # Lower Left
    myrect(s, (tot_width, -w), (tot_width+w, v_offset))                       # Lower Right
    
    #Draw box on the top side of the CPW
    myrect(s, (-w, tot_height), (tot_width+w, tot_height+w))                  # Top
    myrect(s, (-w, v_offset+line_width), (0, tot_height+w))                   # Upper Left
    myrect(s, (tot_width, v_offset+line_width), (tot_width+w, tot_height+w))  # Upper Right
    
    #Draw capacitor fingers
    l, w, g = c_length, c_width, c_gap
    for i in range(c_fingers):
        myrect(s, (l + w, 0), (l + w + g, w))
        myrect(s, (w, w), (l + w + g, w + g))
        myrect(s, (w, w + g), (w + g, 2*w + g))
        myrect(s, (w, 2*w + g), (l + w + g, 2*w + 2*g))
        #s.last = s.last[0], (s.last[1] + 2*w + 2*g)
        s.last = orient_pt((0, sign*(2*w + 2*g)), s.last_direction, s.last)
        
            
    #fill in the empty space if the cap length is not as long as the inductor length        
    if tot_c_length < tot_l_length:
        myrect(s,(l + w + g,0),(tot_width-l_width, -g))
        if (tot_l_length-tot_c_length)>g:
            for i in range(c_fingers):
                myrect(s,(l+w+g,-2*(i+0.5)*(w+g)),(tot_width-g, -2*(i+0.5)*(w+g)-2*g-w))
        
    #Draw inductor fingers   
    l, w, g = l_length, l_width, l_gap
    for i in range(l_fingers):
        myrect(s, (l + w, 0), (l + w + g, 2*w + g))
        myrect(s, (0, w), (l, w + g))
        myrect(s, (w, 2*w + g), (l + w + g, 2*w + 2*g))
        #s.last = s.last[0], (s.last[1] + 2*w + 2*g)
        s.last = orient_pt((0, sign*(2*w + 2*g)), s.last_direction, s.last)

    
    
    #fill in the empty space if the inductor length is not 
    #as long as the capacitor length    
    if tot_l_length< tot_c_length:
        myrect(s,(l+2*w+g,w),(tot_width,-l_fingers*(2*g+2*w)+w))
    #s.last = start[0] + tot_width, start[1]
    
    
    
    s.last = orient_pt((tot_width, 0), s.last_direction, start)
    CPWStraight(s,conn_length)
    
    if describe:
        c = 2 * c_fingers * c_length * 1e-6 * eps0 * eps_eff
        l = 2 * l_fingers * l_length * 1e-6 * mu0
        l += 2 * (tot_height - v_offset) * 1e-6 * mu0    
        print "Estimated c %.2e" % c
        print "Estimated l %.2e" % l
        print "Estimated freq %.2e" % (1/sqrt(l * c)/2/pi)
        print "Estimated Z %.2f" % sqrt(l / c)

def ShuntedLER(s, n_fingers, finger_height, n_meanders, meander_height):
    cap = ground_fingers(n_fingers, finger_height, s.pinw)
    ind = CPWInductiveShunt(n_meanders, meander_height, s.pinw, s.gapw).draw
    cap(s)
    ind(s)
    cap(s)

#This creates a capacitor *to ground* that has fingers coming off of a middle
#CPW straight section. The CPW section has the same characteristics as the CPW 
#defined in "s"
#length: length of the finger (in each direction, i.e. total length is 2*length)
#width: width of the metal finger
#gap: gap between the metal finger and the ground plane
#finger_gap: gap between fingers (i.e. width of the ground plane finger)
#align: draw alignment marks
#nalign: number of alignment marks
def ground_fingers(n_fingers, length, width, align=True, finger_gap=-1, nalign=2):
    def ground_finger_contents(s, gds_info=None):
            if gds_info is not None:
                channel = lambda s, length, width: gds_channel(s, length, width, gds_info[0], gds_info[1])
            else:
                channel = Channel
            
            pin, gap = s.pinw, s.gapw
            tot_length = pin + 2*length
            gap_width = width + 2*gap
            channel(s, gap, pin)
            for _ in range(n_fingers - 1):
                channel(s, width, tot_length)
                channel(s, gap_width, pin)
            channel(s, width, tot_length)
            channel(s, gap, pin)
    
    
    def builder(s, gap=None, empty=False, contents_struct=None, gds_info=None):
        dist = 65
        #pin, gap = s.pinw, s.gapw
        pin = s.pinw
        gap = gap if (gap is not None) else s.gapw
        if finger_gap<0:
            gap2 = gap
        else:
            gap2 = finger_gap
            
            
        #for drawing alignment crosses
        cross_height = length + gap + pin/2. + dist
        finger_start = s.last
                
        for i in range(n_fingers):
            if empty:
                pass
                CPWStraight(s, 2*gap2 + width, pinw=0, gapw=gap+gap2+length+pin/2.)
            else:
                CPWStraight(s, gap2, gapw=gap+gap2+length)
                CPWStraight(s, width, pinw=pin+2.*(length+gap), gapw=gap2)
                CPWStraight(s, gap2, gapw=gap+gap2+length)
            if i != (n_fingers - 1):
                if empty:
                    CPWStraight(s, width, pinw=0, gapw=gap+pin/2.)
                else:
                    CPWStraight(s, width, gapw=gap)
                    

        finger_dist = ((s.last[0]-finger_start[0])**2+(s.last[1]-finger_start[1])**2)**0.5       
        
        if align:
                        
            make_alignment_crosses(s,12,2,nx=nalign,ny=2,direction=s.last_direction,start_pt=finger_start,dx=finger_dist/(nalign-1),dy=2*cross_height)
            '''            
            for i in [1,-1]:
                s_test = Structure(None,finger_start,s.last_direction)
                s_test.last = finger_start
                s_test.move(length + gap + pin/2. + dist,s.last_direction+i*90)
                for j in range(nalign):
                    cross(s, s_test.last)
                    s_test.move()
                    cross(s, (0, i*(length + gap + pin/2. + dist)))
            '''
                    
                
            
        
        if contents_struct is not None:
            ground_finger_contents(contents_struct)
            
        if gds_info is not None:
            ground_finger_contents(s, gds_info=gds_info)
    return builder

#this draws alignment crosses from start_pt to end_pt in a grid nx by ny (x is along direction)
#omit is a nx x ny array of ones and zeros (zero is don't draw cross)
#start point is in the *middle* of the y on the far x side
def make_alignment_crosses(s,cross_x,cross_y,nx,ny,direction,start_pt,dx,dy,omit=None):
    
    if omit is None:
        omit = ones((nx,ny))
        
    cross = alignment_cross(cross_x, cross_y)
    
    s_test = Structure(Chip("test"),start_pt,direction)
    s_test.move(dy*(ny-1)/2,direction-90)
    start_pt = s_test.last
    
    for i in range(nx):
        s_test.last = start_pt
        s_test.move(i*dx)
        for j in range(ny):
            if omit[i,j]==1:
                cross(s,(s_test.last[0]-s.last[0],s_test.last[1]-s.last[1]))
            s_test.move(dy,direction+90)
            
            

def make_qubit(left_fingers, left_len, left_width, 
               right_fingers, right_len, right_width, cgap,
               left_cap, right_cap, fname='Qubit'):
    c = Chip(fname)
    s = Structure(c, defaults=global_defaults)
    #CPWFingerCapInside()
    left_cap.to_inner_cap().draw(s)
    ground_finger_contents(s, left_fingers, left_len, left_width)
    s.last = s.last[0] + cgap, s.last[1]
    ground_finger_contents(s, right_fingers, right_len, right_width)
    right_cap.to_inner_cap().draw(s)
    #CPWFingerCapInside(4, 70, 8, 4, 30).draw(s)
    c.save()

#import gdsii.elements as gds_elem
def gds_rect(s, p0, p1, datatype=0, layer=0):
    p01 = p0[0], p1[1]
    p10 = p1[0], p0[1]
    points = orient_pts([p0, p01, p1, p10, p0], s.last_direction, s.last)
    elt = gds_elem.Boundary(layer, datatype, points)
    s.gds_chip.append(elt)
    if isinstance(s, SpacerStructure):
        s.append(('GDS Element', elt))

def gds_line(s, p0, p1, datatype=0, layer=0):
    points = orient_pts([p0, p1], s.last_direction, s.last)
    elt = gds_elem.Path(layer, datatype, points)
    s.gds_chip.append(elt)
    if isinstance(s, SpacerStructure):
        s.append(('GDS Element', elt))

ALIGNMENT_IDENTIFIER = 0
def alignment_cross(size, weight=1):
    def builder(s, pos):
        to_recover = s.last
        (x0, y0), (x1, y1) = s.last, pos
        s.last = x0+x1, y0+y1
        h = size/2.
        w = weight/2.
        rect(s, (-w, -h), (w, h))
        rect(s, (-h, -w), (h, w))
        '''
        if s.gds_chip is not None:
            global ALIGNMENT_IDENTIFIER
            ident = ALIGNMENT_IDENTIFIER
            ALIGNMENT_IDENTIFIER += 1
            gds_rect(s, (-h, -h), (h, h), datatype=s.gds_alignment_dt, layer=ident)
            gds_line(s, (0, -h), (0, h), datatype=s.gds_alignment_dt, layer=ident)
            gds_line(s, (-h, 0), (h, 0), datatype=s.gds_alignment_dt, layer=ident)
        '''
        s.last = to_recover
        
    return builder

def test_element(name, elt_fn, d=global_defaults, caps=None, length=0, **kwargs):
    c = Chip(name)
    s = Structure(c, start=c.midpt, defaults=d)
    if caps is not None:
        try: cap = sapphire_capacitor_by_C(caps)
        except: cap = sapphire_capacitor_by_C(15e-15)
    CPWStraight(s, length)
    if caps: cap.draw(s)
    CPWStraight(s, 50)
    elt_fn(s, **kwargs)
    CPWStraight(s, 50)
    if caps: cap.draw(s)
    CPWStraight(s, length)
    c.save()

def CPWEmptyTaper(s, length, start_gap, stop_gap):
    h0, h1 = start_gap/2., stop_gap/2.
    pts = [(0, h0), (length, h1), (length, -h1), (0, -h0), (0, h0)]
    s.append(sdxf.PolyLine(orient_pts(pts, s.last_direction, s.last)))
    s.last = orient_pt((length, 0), s.last_direction, s.last)

class HalfCap(CPWFingerCap):
    def __init__(self, full_cap):
        self.__dict__.update(full_cap.__dict__)
        self.taper_length = 0 # This is false, but convenient. Actual taper_length is a constant 50um TODO - Change this?
    
    def half_cap_contents(self, s, n_fingers, gds_info=None):
        ChannelLinearTaper(s, 50, s.pinw, self.pinw)
        spacing = 2*(self.finger_width + self.finger_gap)
        if bool(self.num_fingers % 2) != self.flipped:
            n_fingers += 1
        for i in range(n_fingers):
            p0 = (0, self.pinw/2. - i*spacing)
            p1 = vadd(p0, (self.finger_length, -self.finger_width))
            rect(s, p0, p1)

    def draw(self, s, flipped=False, contents_struct=None, gds_info=None):
        self.fingers_remaining = 0
        init_pinw, init_gapw = s.pinw, s.gapw
        self.flipped = flipped
        self.gapw = self.pinw*s.gapw/float(s.pinw)
        if flipped:
            CPWEmptyTaper(s, 50, init_pinw + 2*init_gapw, self.pinw + 2*self.gapw)
            rect(s, (0, self.pinw/2.), (self.finger_length, self.pinw/2. - self.finger_width))
        else:
            CPWLinearTaper(s, 50, init_pinw, self.pinw, init_gapw, self.gapw)
        CPWFingerCap.draw(self, s)
        if flipped:
            CPWLinearTaper(s, 50, self.pinw, init_pinw, self.gapw, init_gapw)
        else:
            CPWEmptyTaper(s, 50, self.pinw + 2*self.gapw, init_pinw + 2*init_gapw)
        s.pinw, s.gapw = init_pinw, init_gapw
        
        if contents_struct is not None:
            if not flipped:
                contents_struct.last = s.last
                contents_struct.last_direction += 180
            self.half_cap_contents(contents_struct, self.fingers_remaining)
            if not flipped:
                contents_struct.last_direction -= 180
                contents_struct.last = s.last
        
        if gds_info is not None:
            self.half_cap_contents(s, self.fingers_remaining, gds_info=gds_info)
                
    def left_finger_points(self,finger_width,finger_length,finger_gap):
        if self.flipped:
            pts= [  (0,0),
                    (0,finger_width+finger_gap),
                    (finger_length+finger_gap,finger_width+finger_gap),
                    (finger_length+finger_gap,finger_width),
                    (finger_gap,finger_width),
                    (finger_gap,0),
                    (0,0)
                ]
        else:
            self.fingers_remaining += 1
            pts= [  (0,0),
                    (0,finger_width+finger_gap),
                    (finger_length+finger_gap,finger_width+finger_gap),
                    (finger_length+finger_gap,0),
                    (0,0)
                ]
        return pts
    def right_finger_points(self,finger_width,finger_length,finger_gap):
        if self.flipped:
            self.fingers_remaining += 1
            pts = [ (finger_length+finger_gap,0),
                    (finger_length+finger_gap,finger_width+finger_gap),
                    (0,finger_width+finger_gap),
                    (0,0),
                    (finger_length+finger_gap,0)
                    ]
        else:
            pts = [ (finger_length+finger_gap,0),
                    (finger_length+finger_gap,finger_width+finger_gap),
                    (0,finger_width+finger_gap),
                    (0,finger_width),
                    (finger_length,finger_width),
                    (finger_length,0),
                    (finger_length+finger_gap,0)
                    ]
        return pts

def vadd(a,b):
    return a[0] + b[0], a[1] + b[1]

# TODO :: Refactor out space filling and chip processing functions
class SpacerStructure(Structure):
    def __init__(self, *args, **kwargs):
        Structure.__init__(self, *args, **kwargs)
        self.pl_list = []
        self.n_spacers = 0
    def append(self, v):
        self.pl_list.append(v)
    def process_to_chip(self, goal_x=None):
        #if self.last_direction != 0:
        #   raise NotImplementedError
        if goal_x is None:
            goal_x = self.chip.size[0]
        delta = goal_x - self.last[0]
        assert delta >= 0
        spacer_len = delta / self.n_spacers if self.n_spacers else 0
        spacers_seen = 0
        move_pts = lambda pts: translate_pts(pts, (spacers_seen*spacer_len, 0))
        for pl in self.pl_list:
            if isinstance(pl, sdxf._Entity):
                pts = pl.points
                self.chip.append(sdxf.PolyLine(move_pts(pts)))
            elif pl[0] == "Substructure":
                for pl2 in pl[1].pl_list:
                    self.chip.append(sdxf.PolyLine(move_pts(pts), layer=pl[1].layer, color=pl[1].color))
            elif pl[0] == "GDS Element":
                pl[1].xy = move_pts(pl[1].xy)
            elif pl[0] == "Horizontal Spacer":
                straight_start = (pl[1][0] + (spacer_len * spacers_seen), pl[1][1])
                spacers_seen += 1
                new_s = Structure(self.chip, start=straight_start, defaults=self.defaults)
                CPWStraight(new_s, spacer_len)
            else:
                print "Unknown object in SpacerStructure", pl
        if spacers_seen != self.n_spacers:
            print "Not enough spacers?"
        return spacer_len
    def translate(self, offset):
        new_pl_list = []
        for pl in self.pl_list:
            if isinstance(pl, sdxf._Entity):
                new_pl_list.append(sdxf.PolyLine(translate_pts(pl.points, offset)))
            else:
                new_pl_list.append(pl)
        self.pl_list = new_pl_list
    def sub_struct(self, layer, color=1):
        s = SpacerStructure(self.chip, start=self.last, direction=self.last_direction, 
                            defaults=self.defaults, layer=layer, color=color)
        self.pl_list.append(('Substructure',s))
        return s

class CPWHorizontalSpacer:
    def __init__(self, structure):
        assert isinstance(structure, SpacerStructure)
        structure.append(("Horizontal Spacer", structure.last))
        structure.n_spacers += 1

class ChipDefaults(dict):
    @autoargs()
    def __init__(self, chip_size=(7000,2000), dicing_border=350, 
                 eps_eff=5.5, impedance=50, pinw=10, gapw=None, radius=25,
                 res_freq=10, res_length=None, Q=1e5, 
                 mask_id_loc=(300,1620), chip_id_loc=(6300,1620)):
        self.phase_velocity = speedoflight/sqrt(eps_eff)
        if not gapw:
            self.gapw = calculate_gap_width(eps_eff, impedance, pinw)
        else:
            self.gapw = gapw
        if not self.res_length:
            self.res_length = \
            calculate_interior_length(res_freq, self.phase_velocity, impedance)
    def __getitem__(self, name):
        return getattr(self, name)
    def __setitem__(self, name, value):
        setattr(self, name, value)
    def copy(self):
        import copy
        return copy.copy(self)

def MyWaferMask(name, defaults=ChipDefaults(), **kwargs):
    return WaferMask(name, chip_size=defaults.chip_size, 
                     dicing_border=defaults.dicing_border,
                     **kwargs)
        
def QBox(size=5, **kwargs):
    if size == 4:
        return CPWQubitBox2(4, 6, 6, 91.4-4, 87-4, 6.3-4, 24-4, **kwargs)
    if size == 5:
        return CPWQubitBox2(5, 14, 18, 160.5, 122.9, 14.2, 30.4, **kwargs)
    elif size == 10:
        return CPWQubitBox2(10, 13, 15, 183, 154, 10, 37.4, **kwargs)
    else:
        raise ValueError(str(size)+"um not yet simulated")
     
from copy import copy, deepcopy
import random    
     
def perforate(chip, grid_x, grid_y):
    nx, ny = map(int, [chip.size[0] / grid_x, chip.size[1] / grid_y])
    occupied = [[False]*ny for i in range(nx)]
    for i in range(nx):
        occupied[i][0] = True
        occupied[i][-1] = True
    for i in range(ny):
        occupied[0][i] = True
        occupied[-1][i] = True
    
    for e in chip.entities:
        o_x_list = []
        o_y_list = []
        for p in e.points:
            o_x, o_y = map(int, (p[0] / grid_x, p[1] / grid_y))
            if 0 <= o_x < nx and 0 <= o_y < ny:
                o_x_list.append(o_x)
                o_y_list.append(o_y)
        if o_x_list:
            for x in range(min(o_x_list), max(o_x_list)+1):
                for y in range(min(o_y_list), max(o_y_list)+1):
                    occupied[x][y] = True
        
    second_pass = deepcopy(occupied)
    for i in range(nx):
        for j in range(ny):
            if occupied[i][j]:
                for ip, jp in [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]:
                    try:
                        second_pass[ip][jp] = True
                    except IndexError:
                        pass
        
    for i in range(nx):
        for j in range(ny):
            if not second_pass[i][j]:
                size = random.uniform(8, 18)
                pos = i*grid_x + grid_x/2., j*grid_y + grid_y/2.
                p0 = vadd(pos, (-size, -size))
                p1 = vadd(pos, (size, size))
                abs_rect(chip, p0, p1)

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as mpatches
def polygon(plot, _points):
    points = copy(_points)
    if points[0] != points[-1]:
        points.append(points[0])
    n_points = len(points)
    codes = [Path.MOVETO] + ([Path.LINETO] * (n_points - 2)) + [Path.CLOSEPOLY]
    path = Path(points, codes)
    patch = mpatches.PathPatch(path, facecolor='blue', lw=0)
    plot.add_patch(patch)

def show_chip(chip):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for e in chip.entities:
        polygon(ax, e.points)
    ax.set_xlim(-.1*chip.size[0], 1.1*chip.size[0])
    ax.set_ylim(-.1*chip.size[1], 1.1*chip.size[1])
    plt.show()

def corner_crosses(chip, height, width, gap):
    s = Structure(chip)
    cross = alignment_cross(height, width)
    for x in (-gap, chip.size[0] + gap):
        for y in (-gap, chip.size[1] + gap):
            cross(s, (x, y))