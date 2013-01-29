# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:35:36 2012

@author: slab
"""

from slab.circuits.MaskMaker import *
from slab.circuits.ResonatorCalculations import *

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
            print s.last, "original"
            print new_start, "new"

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
        
        print "CHECK", tot_span, total_length
        

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

def LumpedElementResonator(s, c_fingers, l_fingers, length, 
                           c_width=3, c_gap=3, l_width=5, l_gap=11, v_offset=50):
    start = s.last
    s.last = s.last[0], s.last[1] - v_offset
    c_length, l_length = length, length + c_width + c_gap - l_width - l_gap
    

    tot_width = c_length + 2*c_width + c_gap
    tot_height = c_fingers*(2*c_width + 2*c_gap) + l_fingers*(2*l_width + 2*l_gap) + l_width
    line_width = s.pinw + 2*s.gapw
    v_offset = v_offset - line_width/2.
    w = c_width
    rect(s, (-w, -w), (tot_width+w, 0))                                      # Bottom
    rect(s, (-w, tot_height), (tot_width+w, tot_height+w))                   # Top
    rect(s, (-w, -w), (0, v_offset))                                         # Lower Left
    rect(s, (-w, v_offset+line_width), (0, tot_height+w))                    # Upper Left
    rect(s, (tot_width, -w), (tot_width+w, v_offset))                        # Lower Right
    rect(s, (tot_width, v_offset+line_width), (tot_width+w, tot_height+w))   # Upper Right
    l, w, g = c_length, c_width, c_gap
    for i in range(c_fingers):
        rect(s, (l + w, 0), (l + w + g, w))
        rect(s, (w, w), (l + w + g, w + g))
        rect(s, (w, w + g), (w + g, 2*w + g))
        rect(s, (w, 2*w + g), (l + w + g, 2*w + 2*g))
        s.last = s.last[0], (s.last[1] + 2*w + 2*g)
    l, w, g = l_length, l_width, l_gap
    for i in range(l_fingers):
        rect(s, (l + w, 0), (l + w + g, 2*w + g))
        rect(s, (0, w), (l, w + g))
        rect(s, (w, 2*w + g), (l + w + g, 2*w + 2*g))
        s.last = s.last[0], (s.last[1] + 2*w + 2*g)
    s.last = start[0] + tot_width, start[1]
    
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

def ground_fingers(n_fingers, length, width):
    def builder(s, empty=False):
        dist = 65
        pin, gap = s.pinw, s.gapw
        cross = alignment_cross(12, 2)
        cross(s, (0, length + gap + pin/2. + dist))
        cross(s, (0, -(length + gap + pin/2. + dist)))
        for i in range(n_fingers):
            if empty:
                CPWStraight(s, 2*gap + width, pinw=0, gapw=gap+length+pin/2.)
            else:
                CPWStraight(s, gap, gapw=gap+length)
                CPWStraight(s, width, pinw=pin+2*length)
                CPWStraight(s, gap, gapw=gap+length)
            if i != (n_fingers - 1):
                if empty:
                    CPWStraight(s, width, pinw=0, gapw=gap+pin/2.)
                else:
                    CPWStraight(s, width)
        cross(s, (0, length + gap + pin/2. + dist))
        cross(s, (0, -(length + gap + pin/2. + dist)))
        
    print "Ground Fingers: Estimated Cap", eps0 * eps_eff * n_fingers * (length * 1e-6) * 2
    return builder

def alignment_cross(size, weight=1):
    def builder(s, pos):
        to_recover = s.last
        (x0, y0), (x1, y1) = s.last, pos
        s.last = x0+x1, y0+y1
        h = size/2.
        w = weight/2.
        rect(s, (-w, -h), (w, h))
        rect(s, (-h, -w), (h, w))
        s.last = to_recover
    return builder

def test_element(name, elt_fn, d=global_defaults, caps=None, **kwargs):
    print d
    c = Chip(name)
    s = Structure(c, start=c.midpt, defaults=d)
    if caps is not None:
        try: cap = sapphire_capacitor_by_C(caps)
        except: cap = sapphire_capacitor_by_C(15e-15)
    CPWStraight(s, 50)
    if caps: cap.draw(s)
    CPWStraight(s, 50)
    elt_fn(s, **kwargs)
    CPWStraight(s, 50)
    if caps: cap.draw(s)
    CPWStraight(s, 50)
    c.save()

def CPWEmptyTaper(s, length, start_gap, stop_gap):
    h0, h1 = start_gap/2., stop_gap/2.
    pts = [(0, h0), (length, h1), (length, -h1), (0, -h0), (0, h0)]
    s.append(sdxf.PolyLine(orient_pts(pts, s.last_direction, s.last)))
    s.last = orient_pt((length, 0), s.last_direction, s.last)

class HalfCap(CPWFingerCap):
    def __init__(self, full_cap):
        self.__dict__.update(full_cap.__dict__)
        
        self.taper_length = 0
    def draw(self, s, flipped=False):
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
            pts= [  (0,0),
                    (0,finger_width+finger_gap),
                    (finger_length+finger_gap,finger_width+finger_gap),
                    (finger_length+finger_gap,0),
                    (0,0)
                ]
        return pts
    def right_finger_points(self,finger_width,finger_length,finger_gap):
        if self.flipped:
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
        for pl in self.pl_list:
            if isinstance(pl, sdxf._Entity):
                pts = pl.points
                self.chip.append(sdxf.PolyLine(translate_pts(pts, (spacers_seen * spacer_len,0))))
            elif pl[0] == "Horizontal Spacer":
                straight_start = (pl[1][0] + (spacer_len * spacers_seen), pl[1][1])
                spacers_seen += 1
                new_s = Structure(self.chip, start=straight_start, defaults=self.defaults)
                CPWStraight(new_s, spacer_len)
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
