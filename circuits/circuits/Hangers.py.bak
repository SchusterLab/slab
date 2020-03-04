from . import sdxf
from .MaskMaker import *
from .ResonatorCalculations import *


#HangerCoupler
#Should have information describing capacitor
#Should have effective_length function, used for auto-determining length of hanger
#Should have dimensions function describing how long and tall it is (relative to old start)

class HangerNotch:
    def __init__(self,notch_length,notch_depth,superfine_offset,superfine_spacing,superfine_size,fine_offset,fine_size,rough_size,padding=0,flipped=False,pinw=None,gapw=None):
        self.notch_length=notch_length
        self.notch_depth=notch_depth
        self.superfine_offset=superfine_offset
        self.superfine_spacing=superfine_spacing
        self.superfine_size=superfine_size
        self.fine_offset=fine_offset
        self.fine_size=fine_size
        self.rough_size=rough_size
        self.padding=padding
        self.flipped=flipped
        self.pinw=pinw
        self.gapw=gapw
    
    def draw(self,structure):
        s=structure
        if self.pinw is None: self.pinw=s.defaults['pinw']
        if self.gapw is None: self.gapw=s.defaults['gapw']
        
        CPWStraight(s,self.padding/2.)
        
        notch_pts= [ (-self.notch_length/2,-self.pinw/2.-self.gapw), (-self.notch_length/2,-self.pinw/2.-self.gapw-self.notch_depth), (self.notch_length/2,-self.pinw/2.-self.gapw-self.notch_depth),(self.notch_length/2,-self.pinw/2.-self.gapw), (-self.notch_length/2,-self.pinw/2.-self.gapw)]
        notch_pts=orient_pts(notch_pts,s.last_direction,s.last)
        
        box_pts= [ (-1,-1),(1,-1),(1,1),(-1,1),(-1,-1)]

        #Superfine
        sf_pts=scale_pts(box_pts,(self.superfine_size/2.,self.superfine_size/2.))
        sf1_pts=translate_pts(sf_pts,(self.superfine_spacing,-self.superfine_offset))
        sf2_pts=translate_pts(sf_pts,(-self.superfine_spacing,-self.superfine_offset))
        
        sf1_pts=orient_pts(sf1_pts,s.last_direction,s.last)
        sf2_pts=orient_pts(sf2_pts,s.last_direction,s.last)
        
        #Fine
        
        f_pts=scale_pts(box_pts,(self.fine_size/2.,self.fine_size/2.))
        f_pts=translate_pts(f_pts,(self.fine_offset,-self.superfine_offset))
        f_pts=orient_pts(f_pts,s.last_direction,s.last)
        
        #Rough
        chip_size=s.chip.size
        
        #dir_pt=(cos(s.last_direction*pi/180.)*chip_size[0]/2.,sin(s.last_direction*pi/180.)*chip_size[1]/2.)
        dir_pt=(cos(s.last_direction*pi/180.),sin(s.last_direction*pi/180.))
        
        r_cpt=orient_pt((0,-self.superfine_offset),s.last_direction,s.last)
        r_cpt=(r_cpt[0]*abs(dir_pt[1])+(1-dir_pt[1])*(chip_size[0]-self.rough_size/2.),r_cpt[1]*abs(dir_pt[0])+(1-dir_pt[0])*(chip_size[1]-self.rough_size/2.))
        
        r_pts=scale_pts(box_pts,(self.rough_size/2.,self.rough_size/2.))
        r_pts=orient_pts(r_pts,s.last_direction,r_cpt)
        
        if self.flipped:
            notch_pts=mirror_pts(notch_pts,s.last_direction,s.last)
            sf1_pts=mirror_pts(sf1_pts,s.last_direction,s.last)
            sf2_pts=mirror_pts(sf2_pts,s.last_direction,s.last)
            f_pts=mirror_pts(f_pts,s.last_direction,s.last)
            r_pts=mirror_pts(r_pts,s.last_direction,s.last)
            
            
        s.append(sdxf.PolyLine(notch_pts))
        s.append(sdxf.PolyLine(sf1_pts))
        s.append(sdxf.PolyLine(sf2_pts))
        s.append(sdxf.PolyLine(f_pts))
        s.append(sdxf.PolyLine(r_pts))
        
        CPWStraight(s,self.padding/2.)

def THanger_by_freq(frequency=None,phase_velocity=None, capacitor=None, stub_length=None, taper_length=None, flipped=None,pinw=None, gapw=None, radius=None, height=None, num_wiggles=None, notch=None,defaults=None):
    #print "Making L hanger by frequency"
    if frequency is None: frequency=defaults['frequency']
    if phase_velocity is None: phase_velocity=defaults['phase_velocity']
    #if impedance is None: impedance=defaults['impedance']    
    if capacitor is None: capacitor=defaults['capacitor']
    if pinw is None: pinw = defaults['pinw']
    if gapw is None: gapw = defaults['gapw']
    eps_eff=calculate_eps_eff (phase_velocity)
    impedance=calculate_impedance (pinw,gapw,eps_eff)
    interior_length=calculate_interior_length(frequency,phase_velocity,impedance,resonator_type=0.25,harmonic=0,Ckin=capacitor)    
    #print "Resonator at frequency=%f GHz is length=%f mm" % (frequency,interior_length*1e-3) 
    hanger=THanger(capacitor, stub_length,taper_length,flipped,interior_length, pinw, gapw, radius, height, num_wiggles,notch,eps_eff,defaults)
    #hanger.frequency=frequency
    #hanger.Q=calculate_resonator_Q(frequency,impedance,Ckin=capacitor)
    #hanger.impedance=impedance

    return hanger
    
class THanger:
    def __init__(self,capacitor=None,stub_length=None,taper_length=None,flipped=None,interior_length=None, pinw=None, gapw=None, radius=None, height=None, num_wiggles=None,notch=None,eps_eff=None,defaults=None):
        """If anything is left unspecified then defaults must be defined"""
        if capacitor is None: capacitor=defaults['capacitor']
        self.capacitor = capacitor
        if interior_length is None: interior_length=defaults['interior_length']
        self.interior_length=interior_length
        if stub_length is None: stub_length=defaults['stub_length']
        self.stub_length=stub_length
        if taper_length is None: taper_length=defaults['taper_length']
        self.taper_length=taper_length
        if flipped is None: 
            if 'flipped' in defaults: flipped=defaults['flipped']
            else:                           flipped=False
        self.flipped = flipped
        if pinw is None: pinw=defaults['pinw']
        self.pinw=pinw
        if gapw is None: gapw=defaults['gapw']
        self.gapw=gapw
        if radius is None: radius=defaults['radius']
        self.radius=radius
        if height is None: height=defaults['height']
        self.height=height
        if num_wiggles is None: num_wiggles=defaults['num_wiggles']
        self.num_wiggles=num_wiggles
        if notch is None: notch=defaults['notch']
        self.notch=notch
        if eps_eff is None: eps_eff=defaults['eps_eff']
        self.eps_eff=eps_eff
        
        if not self.notch is None: self.notch_padding=self.notch.padding
        else:                      self.notch_padding=0
        
        #calculate lengths
        self.width=2*num_wiggles*2*self.radius+self.pinw+2*self.gapw
        #print self.width
        #self.ilength=calculate_interior_length(freq,phase_velocity,impedance,resonator_type=0.25,harmonic=0,Ckin_desc=cap_desc)
        self.ilength=interior_length
        self.T_height=self.stub_length+2*self.taper_length+capacitor.length
        #self.T_width=2*self.gapw+self.pinw
        if (num_wiggles>0) and (self.height-self.T_height-self.notch_padding < self.ilength-self.taper_length):
            self.vlength=self.height-self.T_height-self.radius-self.notch_padding
            self.need_wiggles=True
            self.meander_length=self.ilength-self.vlength-self.taper_length-(pi+1)*self.radius
        else:
            self.vlength=self.ilength-self.taper_length-self.notch_padding
            self.need_wiggles=False
            self.meander_length=0
        #print "height: %f, T_height: %f, vlength: %f, meander_length: %f" %(height,self.T_height,self.vlength,self.meander_length)

        self.impedance=calculate_impedance (self.pinw,self.gapw,eps_eff)
        self.frequency=calculate_resonator_frequency(self.interior_length,eps_eff,self.impedance,resonator_type=0.25,Ckin=self.capacitor)
        self.Q=calculate_resonator_Q(self.frequency,self.impedance,Ckin=self.capacitor)

    def description(self):
        cap_desc=self.capacitor.description()
        desc="Estimated frequency:\t%.2f\tEstimated Q:\t%.2E\tEstimated impedance:\t%.1f\n%s\nCInterior Length:\t%f\tMeander Length:\t%f\theight:\t%f\npin width:\t%f\tgap width:\t%f\tradius:\t%f\n" % (
            self.frequency,self.Q,self.impedance,
            cap_desc,
            self.interior_length,self.meander_length,self.height,
            self.pinw,self.gapw,self.radius
            )
        return desc


    def draw_notch(self):
        if self.notch is None: return
        self.notch.draw(self.coupled_structure)
# Todo: support for hanger_notches

    def draw_coupler(self):
        s=self.structure
        cs=CPWTee(s,stub_length=self.stub_length,flipped=self.flipped)
        self.coupled_structure=cs        
        if self.capacitor.pinw is None: self.capacitor.pinw=cs.defaults['pinw']
        if self.capacitor.gapw is None: self.capacitor.gapw=cs.defaults['gapw']
        CPWLinearTaper(cs,self.taper_length,cs.defaults['pinw'],self.capacitor.pinw,cs.defaults['gapw'],self.capacitor.gapw)
               
        self.capacitor.draw(cs)
        cs.defaults['pinw']=self.pinw
        cs.defaults['gapw']=self.gapw 
        CPWLinearTaper(cs,self.taper_length,self.capacitor.pinw,self.pinw,self.capacitor.gapw,self.gapw)

    def draw_interior(self):
        hs=self.coupled_structure
        CPWStraight(hs,self.vlength)
        if self.need_wiggles:
            CPWBend(hs,-180)
            w=CPWWigglesByLength(hs,num_wiggles=self.num_wiggles,total_length=self.meander_length,start_bend_angle=0,symmetric=False)
            CPWStraight(hs,self.radius)

    def draw(self,structure,padding):
        #prepare
        self.structure=structure
        #print structure
        self.spinw=structure.defaults['pinw']
        self.sgapw=structure.defaults['gapw']
        
        
        #draw hanger
        self.T_width=2*self.sgapw+self.spinw
        CPWStraight(structure,padding+self.pinw/2.+self.gapw-self.T_width/2.)
        self.draw_coupler()
        self.draw_notch()
        self.draw_interior()
        CPWStraight(structure,self.width+padding-self.T_width/2.-self.pinw/2.-self.gapw)
#2* padding+self.width -self.T_width

def LHanger_by_freq(frequency,phase_velocity=None,coupler=None , pinw=None, gapw=None, radius=None,taper_length=None, height=None, num_wiggles=None, notch=None,defaults=None):
    #print "Making L hanger by frequency"
    if phase_velocity is None: phase_velocity=defaults['phase_velocity']
    #if impedance is None: impedance=defaults['impedance']
    if coupler is None: coupler=defaults['coupler']
    if pinw is None: pinw = defaults['pinw']
    if gapw is None: gapw = defaults['gapw']
    eps_eff=calculate_eps_eff (phase_velocity)
    impedance=calculate_impedance (pinw,gapw,eps_eff)
    interior_length=calculate_interior_length(frequency,phase_velocity,impedance,resonator_type=0.25,harmonic=0,Ckin=coupler)    
    #print "Resonator at frequency=%f GHz is length=%f mm" % (frequency,interior_length*1e-3) 
    hanger=LHanger(coupler, interior_length, pinw, gapw, radius, taper_length, height, num_wiggles,notch,eps_eff,defaults)
    
    return hanger

class LHanger:

    def __init__(self,coupler=None,interior_length=None, pinw=None, gapw=None, radius=None, taper_length=None, height=None, num_wiggles=None,notch=None,eps_eff=None,defaults=None):
        """If anything is left unspecified then defaults must be defined"""
        if coupler is None: coupler=defaults['coupler']
        self.coupler = coupler
        if interior_length is None: interior_length=defaults['interior_length']
        self.interior_length=interior_length
        if pinw is None: pinw=defaults['pinw']
        self.pinw=pinw
        if gapw is None: gapw=defaults['gapw']
        self.gapw=gapw
        if radius is None: radius=defaults['radius']
        self.radius=radius
        if height is None: height=defaults['height']
        self.height=height
        if num_wiggles is None: num_wiggles=defaults['num_wiggles']
        self.num_wiggles=num_wiggles
        if notch is None: notch=defaults['notch']
        self.notch=notch
        if taper_length is None: taper_length=defaults['taper_length']
        self.taper_length=taper_length
        
        if not self.notch is None: self.notch_padding=self.notch.padding
        else:                      self.notch_padding=0
                
        if eps_eff is None: eps_eff=defaults['eps_eff']
        self.eps_eff=eps_eff
        
        #calculate lengths
        self.meander_width=2*num_wiggles*2*self.radius+self.pinw+2*self.gapw
        self.coupler_width=self.radius+self.pinw/2.+self.gapw+coupler.coupler_length
        self.width=max(self.meander_width,self.coupler_width)
        #print self.width
        self.ilength=interior_length
        #self.vlength=min(self.height-coupler.separation-2*radius-taper_length-self.notch_padding,self.ilength-taper_length-coupler.coupler_length-pi/2.*radius-self.notch_padding)
        #self.meander_length=self.ilength-self.vlength-(1.5*pi+1)*self.radius-self.coupler.coupler_length-taper_length
        #print "Meander_length: %f mm" % (1.e-3 * self.meander_length)

        if (num_wiggles>0) and (self.height-coupler.separation-2*radius-taper_length-self.notch_padding<self.ilength-taper_length-coupler.coupler_length-pi/2.*radius-self.notch_padding):
            self.vlength=self.height-coupler.separation-2*radius-taper_length-self.notch_padding
            self.need_wiggles=True
            self.meander_length=self.ilength-self.vlength-(1.5*pi+1)*self.radius-self.coupler.coupler_length-taper_length
        else:
            self.vlength=self.ilength-taper_length-coupler.coupler_length-pi/2.*radius-self.notch_padding
            self.need_wiggles=False
            self.meander_length=0

        self.impedance=calculate_impedance (self.pinw,self.gapw,eps_eff)
        self.frequency=calculate_resonator_frequency(self.interior_length,eps_eff,self.impedance,resonator_type=0.25,Ckin=self.coupler)
        self.Q=calculate_resonator_Q(self.frequency,self.impedance,Ckin=self.coupler)
        
    def description(self):
        cap_desc=self.coupler.description()
        desc="Estimated frequency:\t%.2f\tEstimated Q:\t%.2E\tEstimated impedance:\t%.1f\n%s\nCInterior Length:\t%f\tMeander Length:\t%f\theight:\t%f\npin width:\t%f\tgap width:\t%f\tradius:\t%f\n" % (
            self.frequency,self.Q,self.impedance,
            cap_desc,
            self.interior_length,self.meander_length,self.height,
            self.pinw,self.gapw,self.radius
            )
        return desc
            
        
#    def estimate_hanger_parameters(self):
    def draw_coupler(self):
        self.coupler.draw(self.structure)
        

    def draw_notch(self):
        if self.notch is None: return
        self.notch.draw(self.coupler.coupled_structure)
        
    def draw_interior(self):
        hs=self.coupler.coupled_structure
        CPWLinearTaper(hs,self.taper_length,start_pinw=self.coupler.pinw,stop_pinw=self.pinw,start_gapw=self.coupler.gapw,stop_gapw=self.gapw)
        hs.defaults['pinw']=self.pinw
        hs.defaults['gapw']=self.gapw
        CPWStraight(hs,self.vlength)

        if self.need_wiggles:
            CPWBend(hs,-180)
            w=CPWWigglesByLength(hs,num_wiggles=self.num_wiggles,total_length=self.meander_length,start_bend_angle=0,symmetric=False)
            CPWStraight(hs,self.radius)

    def draw(self,structure,padding):
        #prepare
        self.structure=structure
        #print structure
        self.spinw=structure.defaults['pinw']
        self.sgapw=structure.defaults['gapw']
                
        #draw hanger
        CPWStraight(structure,padding+self.pinw/2.+self.gapw)
        self.draw_coupler()

        CPWStraight(structure,self.width+padding-self.coupler_width)

        self.draw_notch()
        self.draw_interior()
  
class HangerChip(Chip):
    def __init__(self,name,hangers,size,mask_id_loc,chip_id_loc,defaults=[],desc=None):
        self.name=name
        self.hangers=hangers
        self.desc=desc
        self.defaults=defaults
        Chip.__init__(self,name,size,mask_id_loc,chip_id_loc)        
        
    def short_description(self):
#            B	5	Hanger chip	pitch test		5.0, 5.2, 5.4, 5.6, 5.8, 6.0		1e3, 1e4, 1e5, 1e6,
        #print self.hangers
        desc=self.desc+"\tHanger Chip, %d hangers\t %.2f" % (self.hangers.__len__(),self.hangers[0].frequency)
        for h in self.hangers[1:]:
            desc+=", %.2f" %h.frequency
        
        #%.1E for exponential notation
        desc+="\t %.1E" % self.hangers[0].Q
        for h in self.hangers[1:]:
            desc+=", %.1E" %h.Q
            
        return desc
    def long_description(self):
        desc=self.desc+"\tHanger Chip, %d hangers\n" % (self.hangers.__len__())
        for (ii,h) in enumerate(self.hangers):
            desc+="Hanger %d\n" % ii
            desc+=h.description()+"\n"
        return desc
            

class StandardHangerChip(HangerChip):
    """Hanger chip"""    
    def __init__(self,name,hangers,size=(7000,2000),feedline_buffer=300,defaults={},desc=None):
        HangerChip.__init__(self,name,hangers,size=size,mask_id_loc=(300,1620),chip_id_loc=(6300,1620),defaults=defaults,desc=desc)
        #print defaults
        s=Structure(self,start=self.left_midpt,color=3,direction=0,defaults=defaults)

        #Launcher parameters
        bond_pad_length=350
        launcher_pinw=150
        launcher_gapw=67.305
        taper_length=300
        launcher_padding=350
        launcher_length=taper_length+bond_pad_length+launcher_padding
        launcher_radius=125
        
        
        feedline_offset=self.left_midpt[1]-feedline_buffer-4*launcher_radius        
        feedline_length=self.size[0]-2*launcher_length
        
        #input launcher
        CPWStraight(s,length=bond_pad_length,pinw=launcher_pinw,gapw=launcher_gapw)
        CPWLinearTaper(s,length=taper_length,start_pinw=launcher_pinw,start_gapw=launcher_gapw,stop_pinw=s.defaults['pinw'],stop_gapw=s.defaults['gapw'])
        CPWStraight(s,100)
        CPWBend(s,-180,radius=launcher_radius)
        CPWStraight(s,100)
        CPWBend(s,90,radius=launcher_radius)
        CPWStraight(s,feedline_offset)
        CPWBend(s,90,radius=launcher_radius)
        CPWStraight(s,launcher_padding)

        #CPWStraight(s,feedline_length/hangers.__len__() )
        total_width=0
        for h in hangers:
            total_width+=h.width
            #print h.width
        padding=((feedline_length-total_width)/(hangers.__len__()))/2
        if padding<0:
            raise MaskError("StandardHangerChip: Total hanger width=%f is too wide to fit in feedline length=%f, reduce width or number of hangers and try again!" % (total_width,feedline_length))
        for h in hangers:
            h.draw(s,padding)
        
        #output launcher
        CPWStraight(s,launcher_padding)
        CPWBend(s,90,radius=125)
        CPWStraight(s,length=feedline_offset)
        CPWBend(s,90,radius=125)
        CPWStraight(s,length=100)
        CPWBend(s,-180,radius=125)
        CPWStraight(s,length=100)
        CPWLinearTaper(s,length=taper_length,start_pinw=s.defaults['pinw'],start_gapw=s.defaults['gapw'],stop_pinw=launcher_pinw,stop_gapw=launcher_gapw)
        CPWStraight(s,length=bond_pad_length,pinw=launcher_pinw,gapw=launcher_gapw)


class StraightHangerChip(HangerChip):
    def __init__(self,name,hangers,size=(7000,2000),defaults={},desc=None):
        HangerChip.__init__(self,name,hangers,size=size,mask_id_loc=(6300,380),chip_id_loc=(6300,1620),defaults=defaults,desc=desc)

        s=Structure(self,start=(1030,0),color=3,direction=90,defaults=defaults)

        #Launcher parameters
        bond_pad_length=350
        launcher_pinw=150
        launcher_gapw=67.305
        taper_length=300
        launcher_padding=100
        launcher_length=taper_length+bond_pad_length+launcher_padding


        #input launcher
        CPWStraight(s,length=bond_pad_length,pinw=launcher_pinw,gapw=launcher_gapw)
        CPWLinearTaper(s,length=taper_length,start_pinw=launcher_pinw,start_gapw=launcher_gapw,stop_pinw=s.defaults['pinw'],stop_gapw=s.defaults['gapw'])
        CPWStraight(s,launcher_padding)
        CPWBend(s,-90,radius=90)
        CPWStraight(s,300)
        CPWBend(s,-90,radius=90)
        CPWStraight(s,300+launcher_padding)
        CPWBend(s,90,radius=90)
        CPWStraight(s,300)
        CPWBend(s,90,radius=90)
        #hanger/feedline parameters
        feedline_length=size[1]-2*launcher_length+2*300+2*launcher_padding
        
        
        #draw Hangers
        total_width=0
        for h in hangers:
            h.flipped=True          #For straight hangers must have Tee flipped
            total_width+=h.width
        padding=((feedline_length-total_width)/(hangers.__len__()))/2
        if padding<0: #raise MaskError, "StraightHangerChip: Total hanger width=%f is too wide to fit in feedline length=%f, reduce width or number of hangers and try again!" % (total_width,feedline_length)
            print("Warning StraightHangerChip (%s): Total hanger width=%f is too wide to fit in feedline length=%f, reduce width or number of hangers and try again!" % (self.name,total_width,feedline_length))
        
        for h in hangers: h.draw(s,padding)
 
        #output launcher

        CPWBend(s,90,radius=90)
        CPWStraight(s,300)
        CPWBend(s,90,radius=90)
        CPWStraight(s,300+launcher_padding)
        CPWBend(s,-90,radius=90)
        CPWStraight(s,300)
        CPWBend(s,-90,radius=90)


        CPWStraight(s,launcher_padding)
        CPWLinearTaper(s,length=taper_length,start_pinw=s.defaults['pinw'],start_gapw=s.defaults['gapw'],stop_pinw=launcher_pinw,stop_gapw=launcher_gapw)
        CPWStraight(s,length=bond_pad_length,pinw=launcher_pinw,gapw=launcher_gapw)
    

class SideLaunchedStraightHangers(Chip):
   
    def __init__(self,name):
        Chip.__init__(self,name,size=(7000,2000),mask_id_loc=(6300,380),chip_id_loc=(6300,1620))
        self.name=name
        s=Structure(self,start=(1030,0),color=3,direction=90)

        s.defaults['pinw']=10
        s.defaults['gapw']=4.186
        s.defaults['radius']=90

        #Launcher parameters
        bond_pad_length=350
        launcher_pinw=150
        launcher_gapw=67.305
        taper_length=300
        launcher_padding=100
        launcher_length=taper_length+bond_pad_length+launcher_padding

        #hanger/feedline parameters
        feedline_length=size[1]-2*launcher_length+2*launcher_padding

        #input launcher
        CPWStraight(s,length=bond_pad_length,pinw=launcher_pinw,gapw=launcher_gapw)
        CPWLinearTaper(s,length=taper_length,start_pinw=launcher_pinw,start_gapw=launcher_gapw,stop_pinw=s.defaults['pinw'],stop_gapw=s.defaults['gapw'])
        CPWStraight(s,launcher_padding)
        CPWBend(s,-90)
        CPWStraight(s,300)
        CPWBend(s,-90)
        CPWStraight(s,300+launcher_padding)
        CPWBend(s,90)
        CPWStraight(s,300)
        CPWBend(s,90)
        feedline_length=2000-2*launcher_length+2*300+2*launcher_padding
        
        
        hanger_freqs=[11e9,11.25e9,11.46e9,11.75e9,12e9]
        phase_velocity=speedoflight/sqrt(5.2)
        for f in hanger_freqs:
            hs=CPWTee(s,stub_length=100,padding_length=feedline_length/hanger_freqs.__len__(),flipped=True)
            Ckin=CapDesc(capacitance=4e-15,cap_gap=2,gapw=hs.defaults['gapw'],num_fingers=2,finger_length=100,finger_width=4)
            Ckin.draw_cap(hs)
            length=calculate_interior_length(f,phase_velocity,50.,resonator_type=0.25,harmonic=0,Ckin_desc=Ckin)
            CPWStraight(hs,length)

        #output launcher

        CPWBend(s,90)
        CPWStraight(s,300)
        CPWBend(s,90)
        CPWStraight(s,300+launcher_padding)
        CPWBend(s,-90)
        CPWStraight(s,300)
        CPWBend(s,-90)


        CPWStraight(s,launcher_padding)
        CPWLinearTaper(s,length=taper_length,start_pinw=s.defaults['pinw'],start_gapw=s.defaults['gapw'],stop_pinw=launcher_pinw,stop_gapw=launcher_gapw)
        CPWStraight(s,length=bond_pad_length,pinw=launcher_pinw,gapw=launcher_gapw)
