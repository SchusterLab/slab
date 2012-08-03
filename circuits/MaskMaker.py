import sdxf
from math import sin,cos,tan,pi,floor,asin,acos,atan,degrees,radians
from alphanum import alphanum_dict
import random
from numpy import sqrt
   
class MaskError:
    """MaskError is an exception to be raised whenever invalid parameters are 
    used in one of the MaskMaker functions, value is just a string"""
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

#===============================================================================       
#  POINT-WISE OPERATIONS   
#===============================================================================
def distance(tuple1,tuple2):
    dx=tuple1[0]-tuple2[0]
    dy=tuple1[1]-tuple2[1]
    return sqrt(dx**2+dy**2)
    
def rotate_pt(p,angle,center=(0,0)):
    """rotates point p=(x,y) about point center (defaults to (0,0)) by CCW angle (in degrees)"""
    dx=p[0]-center[0]
    dy=p[1]-center[1]
    theta=pi*angle/180
    return (center[0]+dx*cos(theta)-dy * sin(theta),center[1]+dx * sin(theta)+dy * cos(theta))
        
def rotate_pts(points,angle,center=(0,0)):
    """Rotates an array of points one by one using rotate_pt"""
    return [rotate_pt(p,angle,center) for p in points]

def translate_pt(p,offset):
    """Translates point p=(x,y) by offset=(x,y)"""
    return (p[0]+offset[0],p[1]+offset[1])

def translate_pts(points,offset):
    """Translates an array of points one by one using translate_pt"""
    return [translate_pt(p,offset) for p in points]

def orient_pt(p,angle,offset):
    """Orient_pt rotates point p=(x,y) by angle (in degrees) and then translates it to offset=(x,y)"""
    return translate_pt(rotate_pt(p,angle),offset)

def orient_pts(points,angle,offset):
    """Orients an array of points one by one using orient_pt"""
    return [orient_pt(p,angle,offset) for p in points]

def scale_pt(p,scale):
    """Scales p=(x,y) by scale"""
    return (p[0]*scale[0],p[1]*scale[1])

def scale_pts(points,scale):
    """Scales an array of points one by one using scale_pt"""    
    return [scale_pt(p,scale) for p in points]


def mirror_pt(p, axis_angle,axis_pt):
    """Mirrors point p about a line at angle "axis_angle" intercepting point "axis_pt" """
    theta=axis_angle*pi/180.
    return (axis_pt[0] + (-axis_pt[0] + p[0])* cos(2 * theta ) + (-axis_pt[1] + p[1])*sin(2 *theta), 
            p[1] + 2 * (axis_pt[1] - p[1])* cos(theta)**2 + (-axis_pt[0] + p[0])* sin(2*theta) )

def mirror_pts(points,axis_angle,axis_pt):
    """Mirrors an array of points one by one using mirror_pt"""    
    return [mirror_pt(p,axis_angle,axis_pt) for p in points]

#===============================================================================       
#  MASK- and CHIP GENERATION    
#===============================================================================

class WaferMask(sdxf.Drawing):
    """Mask class for placing chips on a wafer with a flat.  
    Contains functions which:
        - layout the chips, 
        - add chips to the mask
        - create a manifest of the mask.
        - etchtype 'False' allows you to make a chip without the dicing borders for a positive mask
        - etchtype 'True' is the standard version with dicing borders
    """
    
    def __init__(self,name,diameter=50800.,flat_distance=24100.,wafer_padding=2000,chip_size=(7000.,2000.),dicing_border=200,textsize=(800,800),etchtype=True,wafer_edge=True,dashed_dicing_border=0):
        sdxf.Drawing.__init__(self)
        self.name=name
        self.fileName=name+".dxf"
        self.diameter=diameter
        self.flat_distance=flat_distance
        self.textsize=textsize
        self.border_width=200          #width of line used to align wafer edge
        self.chip_size=chip_size
        self.dicing_border=dicing_border
        self.die_size=(chip_size[0]+dicing_border,chip_size[1]+dicing_border)
        self.wafer_padding=wafer_padding
        self.buffer=self.wafer_padding # + self.dicing_border/2
        self.etchtype=etchtype
        self.dashed_dicing_border=dashed_dicing_border
        
        start_angle=270.+180./pi *acos(2.*flat_distance/diameter)
        stop_angle=270.-180./pi* acos(2.*flat_distance/diameter)
        iradius=(diameter-self.border_width)/2.
        oradius=(diameter+self.border_width)/2.
        starti=rotate_pt((iradius,0.),start_angle)
        starto=rotate_pt((oradius,0.),start_angle)
        stopi=rotate_pt((iradius,0.),stop_angle)
        stopo=rotate_pt((oradius,0.),stop_angle)
        
        #print "wafer info: iradis=%f, oradius=%f, start_angle=%f, stop_angle=%f" %(iradius,oradius,start_angle,stop_angle)
        
        stop_angle+=360
        opts=arc_pts(start_angle,stop_angle,oradius)
        ipts=arc_pts(stop_angle,start_angle,iradius)
        pts=opts
        pts.append(opts[0])
        pts.append(ipts[-1])
        pts.extend(ipts)
        pts.append(opts[0])
        
        #Writes the waffer shape
        if wafer_edge:
            self.append(sdxf.PolyLine(pts))
        
        #self.append(sdxf.Arc((0.,0.),iradius,start_angle,stop_angle))
        #self.append(sdxf.Line( [ stopi,stopo]))
        #self.append(sdxf.Arc((0.,0.),oradius,start_angle,stop_angle))
        #self.append(sdxf.Line( [ starti,starto]))
        #self.append(sdxf.PolyLine([stopi,starti,starto,stopo]))
        
        self.chip_points=self.get_chip_points()
        self.chip_slots=self.chip_points.__len__()
        self.current_point=0
        
        self.manifest=[]
        self.num_chips=0

    def randomize_layout(self):
        """Shuffle the order of the chip_points array so that chips will be inserted (pseudo-)randomly"""
        rnd = random.Random()   
        seed=124279234
        for ii in range(10000):
            i1=rnd.randrange(self.chip_points.__len__())
            i2=rnd.randrange(self.chip_points.__len__())
            tp=self.chip_points[i1]
            self.chip_points[i1]=self.chip_points[i2]
            self.chip_points[i2]=tp
        
    def randomize_layout_seeded(self):
        """Shuffle the order of the chip_points array so that chips will be inserted (pseudo-)randomly"""
        rnd = random.Random()        
        rnd.seed(124279234)
        for ii in range(10000):
            i1=rnd.randrange(self.chip_points.__len__())
            i2=rnd.randrange(self.chip_points.__len__())
            tp=self.chip_points[i1]
            self.chip_points[i1]=self.chip_points[i2]
            self.chip_points[i2]=tp

#    def label_chip(self,chip,pt,maskid,chipid):
#        """Labels chip on wafer at position pt where pt is the bottom left corner of chip"""
#        AlphaNumText(self,maskid,chip.textsize,pt)
#        AlphaNumText(self,chipid,chip.textsize,pt)
        
    def add_chip(self,chip,copies,label=True):
        """Adds chip design 'copies' times into mask.  chip must have a unique name as it will be inserted as a block"""
        if self.etchtype:
            ChipBorder(chip,self.dicing_border/2.)
            
        if self.dashed_dicing_border>0:
            DashedChipBorder(chip,self.dicing_border/2.)
        if chip.two_layer:
            self.layers.append(sdxf.Layer(name='gap', color=1))
            self.layers.append(sdxf.Layer(name='pin', color=3))
            self.blocks.append(chip.gap_layer)
            self.blocks.append(chip.pin_layer)    
        self.blocks.append(chip)
        slots_remaining=self.chip_points.__len__()-self.current_point
        for ii in range (copies):
            if self.current_point>= self.chip_points.__len__():
                raise MaskError, "MaskError: Cannot add %d copies of chip '%s' Only %d slots on mask and %d remaining." % (copies,chip.name,self.chip_points.__len__(),slots_remaining)
            p=self.chip_points[self.current_point]
            self.current_point+=1
            self.append(sdxf.Insert(chip.name,point=p))
            if chip.two_layer:
                self.append(sdxf.Insert(chip.name+'gap', point=p, layer='gap'))
                self.append(sdxf.Insert(chip.name+'pin', point=p, layer='pin'))
            if label:
                chip.label_chip(self,maskid=self.name,chipid=chip.name+str(ii+1),offset=p)
            self.num_chips+=1
        
        self.manifest.append({'chip':chip,'name':chip.name,'copies':copies,'short_desc':chip.short_description(),'long_desc':chip.long_description()})
        #print "%s\t%d\t%s" % (chip.name,copies,chip.short_description())
        chip.save(fname=self.name+"-"+chip.name,maskid=self.name,chipid=chip.name)
    
    
    def save_manifest(self,name=None):
        if name is None: name=self.name
        if name[-4:]!=".txt": name+="_manifest.txt"
        f=open(name,'w')
        f.write("Mask:\t%s\tTotal Chips:\t%d\n" % (self.name,self.current_point))
        f.write("ID\tCopies\tShort Description\tChip Type\tChip Info\n")
        for m in self.manifest:
            f.write("%(name)s\t%(copies)d\t%(short_desc)s\n" %m )

        for m in self.manifest:
            f.write("______________________\n%(name)s\t%(copies)d\t%(long_desc)s\n\n" % m)
        f.close()
    
    def save_dxf(self,name=None):
        if name is None: name=self.name
        if name[-4:]!=".dxf": name+=".dxf"
        #print name
        f=open(name,'w')
        f.write(str(self))
        f.close()
        
    
    def save(self,name=None):
        #print "Saving mask"
        self.save_dxf(name)
        self.save_manifest(name)

    def point_inside(self,pt):
        """True if point is on wafer"""
        return (pt[0]**2+pt[1]**2<(self.diameter/2.-self.buffer)**2) and (pt[1]>-self.flat_distance+self.buffer)
        
    def die_inside(self,pt):
        """Tell if chip of size self.chip_size is completely on the wafer"""
        return self.point_inside(pt) and self.point_inside(translate_pt(pt,(self.die_size[0],0))) and self.point_inside(translate_pt(pt,(self.die_size[0],self.die_size[1]))) and self.point_inside(translate_pt(pt,(0,self.die_size[1])))
    
    def get_chip_points(self):
        """Get insertion points for all of the chips (layout wafer)"""
        max_cols = int((self.diameter-2*self.buffer)/self.die_size[0])
        max_rows = int((self.diameter-2*self.buffer)/self.die_size[1])
        print "Maximum number of rows=%d and cols=%d" %(max_rows,max_cols)
        #figure out offset for chips (centered on chip or between chips)
        xoffset=-max_cols/2.*self.die_size[0]
        yoffset=-max_rows/2.*self.die_size[1]
        #if max_cols%2==1:
        #    print "offset X"
        #    xoffset+=self.chip_size[0]/2.
        #if max_rows%2==1:
        #    yoffset+=self.chip_size[1]/2.
        
        chip_points=[]
        for ii in range(max_rows):
            for jj in range(max_cols):
                pt=(xoffset+jj*self.die_size[0],yoffset+ii*self.die_size[1])
                if self.die_inside(pt):
                    chip_points.append(translate_pt(pt,(self.dicing_border/2.,self.dicing_border/2.)))
        print "Room for %d chips on wafer." % chip_points.__len__()
        return chip_points
        
class Chip(sdxf.Block):
    """Chip is a class which contains structures
       Perhaps it will also be used to do some error checking
    """
    def __init__(self,name,size=(7000.,2000.),mask_id_loc=(0,0),chip_id_loc=(0,0),textsize=(160,160), two_layer=False, layer=None, solid=False, **kwargs):
        """size is a tuple size=(xsize,ysize)"""
        self.two_layer = two_layer
        if two_layer:
            self.gap_layer = Chip(name+"gap", size, mask_id_loc, chip_id_loc,
                                  textsize, layer='gap', solid=solid)
            self.pin_layer = Chip(name+"pin", size, mask_id_loc, chip_id_loc,
                                  textsize, layer='pin', solid=solid)
            #self.rest_layer = Chip(name, size, mask_id_loc, chip_id_loc,
            #                       textsize, layer='rest', solid=solid)
            #self.append = self.rest_layer.append
            
        #else:    
        if layer:
            sdxf.Block.__init__(self,name,layer=layer)
        else:
            sdxf.Block.__init__(self,name)
        self.size=size
        self.solid=solid
        self.mask_id_loc=mask_id_loc
        self.chip_id_loc=chip_id_loc
        self.name=name
        self.textsize=textsize
        self.left_midpt=(0,size[1]/2.)
        self.right_midpt=(size[0],size[1]/2.)
        self.top_midpt=(size[0]/2.,size[1])
        self.bottom_midpt=(size[0]/2.,0)
        self.midpt=(size[0]/2.,size[1]/2.)
        self.bottomleft_corner=(0,0)
        self.topleft_corner=(0,size[1])
        self.topright_corner=(size[0],size[1])
        self.bottomright_corner=(size[0],0)
        self.center=(size[0]/2.,size[1]/2.)
                    
    def label_chip(self,drawing,maskid,chipid,offset=(0,0)):
        """Labels chip in drawing at locations given by mask_id_loc and chip_id_loc with an optional offset.
        Note that the drawing can be a drawing or a Block including the chip itself"""
        AlphaNumText(drawing,maskid,self.textsize,translate_pt(self.mask_id_loc,offset))
        AlphaNumText(drawing,chipid,self.textsize,translate_pt(self.chip_id_loc,offset))        
        
    def save(self,fname=None,maskid=None,chipid=None):
        """Saves chip to .dxf, defaults naming file by the chip name, and will also label the chip, if a label is specified"""
        if fname is None:
            fname=self.name+'.dxf'
        if fname[-4:]!='.dxf':
            fname+='.dxf'
            
        d=sdxf.Drawing()

        if self.two_layer:
            d.layers.append(sdxf.Layer(name='gap', color=1))
            d.layers.append(sdxf.Layer(name='pin', color=3))
            d.blocks.append(self.gap_layer)
            d.append(sdxf.Insert(self.gap_layer.name,point=(0,0),layer='gap'))
            d.blocks.append(self.pin_layer)
            d.append(sdxf.Insert(self.pin_layer.name,point=(0,0),layer='pin'))
            print d.layers
            print self.gap_layer.layer
            print self.pin_layer.layer
            

        d.blocks.append(self)
        d.append(sdxf.Insert(self.name,point=(0,0)))
        self.label_chip(d,maskid,chipid)
        d.saveas(fname)
        
    def short_description(self):
        try: return self.__doc__
        except: return "No description"
    def long_description(self):
        return self.short_description()
        
        
class Structure(object):
    """Structure keeps track of current location and direction, 
    defaults is a dictionary with default values that substructures can call
    """
    def __init__(self,chip,start=(0,0),direction=0,layer="structures",color=1, 
                 defaults={}):
        if chip.two_layer:
            self.gap_layer = Structure(chip.gap_layer, start, direction, 'gap', 
                                       1, defaults)
            self.pin_layer = Structure(chip.pin_layer, start, direction, 'pin', 
                                       2, defaults)
        self.chip=chip        
        self.start=start
        self.last=start
        self.last_direction=direction
        self.layer=layer
        self.color=color
        self.defaults=defaults.copy()
        self.structures=[]
        print self.defaults
        try :self.pinw=chip.pinw 
        except AttributeError: 
            try : self.pinw=self.defaults['pinw']
            except KeyError: print 'no pinw for chips',chip.name, 'at initialization'
        try :self.gapw=chip.gapw 
        except AttributeError: 
            try : self.gapw=self.defaults['gapw']
            except KeyError: print 'no gapw for chips',chip.name, 'at initialization'
        
    def append(self,shape):
        """gives a more convenient reference to the chips.append method"""
        self.chip.append(shape)
        
    def __setattr__(self, name, value):
        if hasattr(self, "chip") and self.chip.two_layer:
            if name is "last":
                self.gap_layer.last = value
                self.pin_layer.last = value
            if name is "last_direction":
                self.gap_layer.last_direction = value
                self.pin_layer.last_direction = value
        object.__setattr__(self, name, value)
    

#===============================================================================       
#  CPW COMPONENTS    
#===============================================================================
class Launcher:
    def __init__(self,structure,flipped=False,pad_length=250,taper_length=250,pad_to_length=500,pinw=None,gapw=None):
        s=structure
        if pinw is None: pinw = s.defaults['pinw']
        if gapw is None: gapw = s.defaults['gapw']

        padding = pad_to_length-pad_length-taper_length
        if padding <0: 
            padding=0
            self.length=pad_length+taper_length
        else:
            self.length=pad_to_length
        
        if not flipped:
            #input launcher
            CPWStraight(s,length=pad_length,pinw=150,gapw=75)
            CPWLinearTaper(s,length=taper_length,start_pinw=150,start_gapw=75,stop_pinw=pinw,stop_gapw=gapw)
            CPWStraight(s,length=padding)
        else:
            CPWStraight(s,length=padding)
            CPWLinearTaper(s,length=taper_length,start_pinw=pinw,start_gapw=gapw,stop_pinw=150,stop_gapw=75)
            CPWStraight(s,length=pad_length,pinw=150,gapw=75)
            
        
#===============================================================================       
#  CPW COMPONENTS    
#===============================================================================

class CPWStraight:
    """A straight section of CPW transmission line"""
    def __init__(self, structure,length,pinw=None,gapw=None):
        """ Adds a straight section of CPW transmission line of length = length to the structure"""
        if length==0: return

        s=structure
        if pinw is None: pinw=structure.defaults['pinw']
        if gapw is None: gapw=structure.defaults['gapw']

        if s.chip.two_layer:
            CPWStraight(s.gap_layer, length, 0, pinw/2. + gapw)
            CPWStraight(s.pin_layer, length, 0, pinw/2.)
            assert s.gap_layer.last == s.pin_layer.last
            s.last = s.gap_layer.last
            return

        else:
            start=structure.last

            gap1=[  (start[0],start[1]+pinw/2),
                    (start[0]+length,start[1]+pinw/2),
                    (start[0]+length,start[1]+pinw/2+gapw),
                    (start[0],start[1]+pinw/2+gapw),
                    (start[0],start[1]+pinw/2)
                    ]

            gap2=[  (start[0],start[1]-pinw/2),
                    (start[0]+length,start[1]-pinw/2),
                    (start[0]+length,start[1]-pinw/2-gapw),
                    (start[0],start[1]-pinw/2-gapw),
                    (start[0],start[1]-pinw/2)
                    ]

            gap1=rotate_pts(gap1,s.last_direction,start)
            gap2=rotate_pts(gap2,s.last_direction,start)
      
            stop=rotate_pt((start[0]+length,start[1]),s.last_direction,start)
            s.last=stop

            if s.chip.solid:
                s.append(sdxf.Solid(gap1[:-1]))
                s.append(sdxf.Solid(gap2[:-1]))

            s.append(sdxf.PolyLine(gap1))
            s.append(sdxf.PolyLine(gap2))
class CPWConnect:   
    def __init__(self,s,endpoint,pinw=None,gapw=None):
        length = distance(endpoint,s.last)
        if length==0: return
        CPWStraight(s,length,pinw=pinw,gapw=gapw)
        self.length=length

class CPWQubitBox:
    """A straight section of CPW transmission line with fingers in the ground plane to add a capacitor"""
    def __init__(self,structure,fingerlen,fingerw,finger_gapw,finger_no,int_len=10,pinw=None,gapw=None,align=True,small=10,medium=20,big=50):
        """ Adds a straight section of CPW transmission line of length = length to the structure"""
        self.fingerlen = fingerlen
        self.fingerw = fingerw
        self.finger_gapw = finger_gapw
        self.finger_no = finger_no
        #This is just the length of one comb of fingers
        length = self.finger_no*self.fingerw+(self.finger_no+1)*(self.fingerw+2*self.finger_gapw)
        self.comb_length = length
        self.total_length = 2*length + int_len
        self.interior_length = int_len

        self.s=structure
        start=structure.last
        
        if pinw is None: pinw=structure.defaults['pinw']
        if gapw is None: gapw=structure.defaults['gapw']
        
        self.pinw = pinw
        self.gapw = gapw
        
        self.top = [  (start[0],start[1]+pinw/2.),
                     (start[0],start[1]+pinw/2.+gapw+self.fingerlen)
                ]        
        self.bot = [  (start[0],start[1]-pinw/2.),
                     (start[0],start[1]-pinw/2.-gapw-self.fingerlen)
                ]
        
        for n in range(finger_no):
            self.add_pin(n)
            
        self.top.extend(   [(start[0]+length,start[1]+pinw/2.+gapw+self.fingerlen),
                (start[0]+length,start[1]+pinw/2.+gapw),
                (start[0]+length+int_len/2.,start[1]+pinw/2.+gapw)
                ])   
        self.bot.extend(   [(start[0]+length,start[1]-pinw/2.-gapw-self.fingerlen),
                (start[0]+length,start[1]-pinw/2.-gapw),
                (start[0]+length+int_len/2.,start[1]-pinw/2.-gapw)
                ])
        
        self.pin = [   (start[0],start[1]+pinw/2.),
                    (start[0]+length-fingerw-2*finger_gapw,start[1]+pinw/2.),
                    (start[0]+length-fingerw-2*finger_gapw,start[1]-pinw/2.),
                    (start[0],start[1]-pinw/2.)
                    ]        
        
        self.top=rotate_pts(self.top,self.s.last_direction,start)
        self.bot=rotate_pts(self.bot,self.s.last_direction,start)
        self.pin=rotate_pts(self.pin,self.s.last_direction,start)
        stop=rotate_pt((start[0]+length+int_len/2.,start[1]),self.s.last_direction,start)
        midpt=stop
        self.s.last=stop

        self.s.append(sdxf.PolyLine(self.top))
        self.s.append(sdxf.PolyLine(self.bot))
        self.s.append(sdxf.PolyLine(self.pin))
                
        self.top=rotate_pts(self.top,180,stop)
        self.bot=rotate_pts(self.bot,180,stop)
        self.pin=rotate_pts(self.pin,180,stop)
        stop=rotate_pt((start[0]+2*length+int_len,start[1]),self.s.last_direction,start)
        self.s.last=stop
                
        self.s.append(sdxf.PolyLine(self.top))
        self.s.append(sdxf.PolyLine(self.bot))
        self.s.append(sdxf.PolyLine(self.pin))
        
        #Adds the proper alignment marks
        
        #s1 = Structure(self,start=start,color=3,direction=0)
           
        
        small_box = [ (-small/2.,-small/2.),
                    (-small/2.,+small/2.),
                    (+small/2.,+small/2.),
                    (+small/2.,-small/2.),
                    (-small/2.,-small/2.)
                    ]
                    
        medium_box = [ (-medium/2.,-medium/2.),
                    (-medium/2.,+medium/2.),
                    (+medium/2.,+medium/2.),
                    (+medium/2.,-medium/2.),
                    (-medium/2.,-medium/2.)
                    ]
                    
        large_box = [ (-big/2.,-big/2.),
                    (-big/2.,+big/2.),
                    (+big/2.,+big/2.),
                    (+big/2.,-big/2.),
                    (-big/2.,-big/2.)
                    ]
                    
        if small == 0:
            small_box = []
                    
        if medium == 0:
            medium_box = []
                    
        if big == 0:
            large_box = []
        
        self.s.append(sdxf.PolyLine(rotate_pts(translate_pts(small_box,(start[0]+small/2.,start[1]+small/2.+pinw/2.+gapw+self.fingerlen+2*small)),self.s.last_direction,center=start)))
        self.s.append(sdxf.PolyLine(rotate_pts(translate_pts(medium_box,(start[0]+self.total_length/4.,start[1]+small/2.+pinw/2.+gapw+self.fingerlen+2*small+400)),self.s.last_direction,center=start)))
        self.s.append(sdxf.PolyLine(rotate_pts(translate_pts(large_box,(start[0]+self.total_length/2.,start[1]+small/2.+pinw/2.+gapw+self.fingerlen+2*small+800)),self.s.last_direction,center=start)))
        self.s.append(sdxf.PolyLine(rotate_pts(translate_pts(medium_box,(start[0]+3*self.total_length/4.,start[1]+small/2.+pinw/2.+gapw+self.fingerlen+2*small+400)),self.s.last_direction,center=start)))
        self.s.append(sdxf.PolyLine(rotate_pts(translate_pts(small_box,(start[0]+self.total_length-small/2.,start[1]+small/2.+pinw/2.+gapw+self.fingerlen+2*small)),self.s.last_direction,center=start)))
        
        self.s.append(sdxf.PolyLine(rotate_pts(translate_pts(small_box,(start[0]+small/2.,start[1]-small/2.-pinw/2.-gapw-self.fingerlen-2*small)),self.s.last_direction,center=start)))
        self.s.append(sdxf.PolyLine(rotate_pts(translate_pts(medium_box,(start[0]+self.total_length/4.,start[1]-small/2.-pinw/2.-gapw-self.fingerlen-2*small-400)),self.s.last_direction,center=start)))
        self.s.append(sdxf.PolyLine(rotate_pts(translate_pts(large_box,(start[0]+self.total_length/2.,start[1]-small/2.-pinw/2.-gapw-self.fingerlen-2*small-800)),self.s.last_direction,center=start)))
        self.s.append(sdxf.PolyLine(rotate_pts(translate_pts(medium_box,(start[0]+3*self.total_length/4.,start[1]-small/2.-pinw/2.-gapw-self.fingerlen-2*small-400)),self.s.last_direction,center=start)))
        self.s.append(sdxf.PolyLine(rotate_pts(translate_pts(small_box,(start[0]+self.total_length-small/2.,start[1]-small/2.-pinw/2.-gapw-self.fingerlen-2*small)),self.s.last_direction,center=start)))
        
    def add_pin(self,n):
        '''"This adds the nth pin to gap1 and gap2'''
        start = self.s.last
        self.top.extend([ (start[0]+(2*n+1)*self.fingerw+2*(n+1)*self.finger_gapw,start[1]+self.pinw/2.+self.gapw+self.fingerlen),
                          (start[0]+(2*n+1)*self.fingerw+2*(n+1)*self.finger_gapw,start[1]+self.pinw/2.+self.gapw),
                          (start[0]+2*(n+1)*self.fingerw+2*(n+1)*self.finger_gapw,start[1]+self.pinw/2.+self.gapw),
                          (start[0]+2*(n+1)*self.fingerw+2*(n+1)*self.finger_gapw,start[1]+self.pinw/2.+self.gapw+self.fingerlen)
                        ])
        self.bot.extend([ (start[0]+(2*n+1)*self.fingerw+2*(n+1)*self.finger_gapw,start[1]-self.pinw/2.-self.gapw-self.fingerlen),
                          (start[0]+(2*n+1)*self.fingerw+2*(n+1)*self.finger_gapw,start[1]-self.pinw/2.-self.gapw),
                          (start[0]+2*(n+1)*self.fingerw+2*(n+1)*self.finger_gapw,start[1]-self.pinw/2.-self.gapw),
                          (start[0]+2*(n+1)*self.fingerw+2*(n+1)*self.finger_gapw,start[1]-self.pinw/2.-self.gapw-self.fingerlen)
                        ])
        
class CPWLinearTaper:
    """A section of CPW which (linearly) tapers from one set of start_pinw and start_gapw to stop_pinw and stop_gapw over length=length"""
    def __init__(self, structure,length,start_pinw,stop_pinw,start_gapw,stop_gapw):
        if length==0: return
        #load attributes
        s=structure
        start=s.last

        if s.chip.two_layer:
            CPWLinearTaper(s.gap_layer, length, 0, 0, (start_pinw/2.)+start_gapw, (stop_pinw/2.)+stop_gapw)
            CPWLinearTaper(s.pin_layer, length, 0, 0, start_pinw/2., stop_pinw/2.)
            s.last = s.gap_layer.last
        else:
            #define geometry of gaps
            gap1= [ 
                (start[0],start[1]+start_pinw/2.),
                (start[0]+length,start[1]+stop_pinw/2.),
                (start[0]+length,start[1]+stop_pinw/2.+stop_gapw),
                (start[0],start[1]+start_pinw/2.+start_gapw),
                (start[0],start[1]+start_pinw/2.)
                ]

            gap2= [ 
                (start[0],start[1]-start_pinw/2.),
                (start[0]+length,start[1]-stop_pinw/2.),
                (start[0]+length,start[1]-stop_pinw/2.-stop_gapw),
                (start[0],start[1]-start_pinw/2.-start_gapw),
                (start[0],start[1]-start_pinw/2.)
                ]

            #rotate structure to proper orientation
            gap1=rotate_pts(gap1,s.last_direction,start)
            gap2=rotate_pts(gap2,s.last_direction,start)

            #create polylines and append to drawing
            if s.chip.solid:
                s.append(sdxf.Solid(gap1[:-1]))
                s.append(sdxf.Solid(gap2[:-1]))
            else:
                s.append(sdxf.PolyLine(gap1))
                s.append(sdxf.PolyLine(gap2))

            #update last anchor position
            stop=rotate_pt((start[0]+length,start[1]),s.last_direction,start)
            s.last=stop
        
#Inside and Outside Versions are for two layer capacitors        
        
class CPWLinearTaperInside:
    """A section of CPW which (linearly) tapers from one set of start_pinw and start_gapw to stop_pinw and stop_gapw over length=length"""
    def __init__(self, structure,length,start_pinw,stop_pinw,start_gapw,stop_gapw):
        if length==0: return
        #load attributes
        s=structure
        start=s.last
        
        #define geometry of gaps
        gap1= [ 
            (start[0],start[1]+start_pinw/2.),
            (start[0]+length,start[1]+stop_pinw/2.)
            ]
                    
        gap2= [ 
            (start[0],start[1]-start_pinw/2.),
            (start[0]+length,start[1]-stop_pinw/2.)
            ]
        
        #rotate structure to proper orientation
        gap1=rotate_pts(gap1,s.last_direction,start)
        gap2=rotate_pts(gap2,s.last_direction,start)

        #create polylines and append to drawing
        s.append(sdxf.PolyLine(gap1))
        s.append(sdxf.PolyLine(gap2))
        
        #update last anchor position
        stop=rotate_pt((start[0]+length,start[1]),s.last_direction,start)
        s.last=stop    
        
class CPWBend:
    """A CPW bend"""
    def __init__(self,structure,turn_angle,pinw=None,gapw=None,radius=None,polyarc=True,segments=60):
        """creates a CPW bend with pinw/gapw/radius
            @param turn_angle: turn_angle is in degrees, positive is CCW, negative is CW
        """
        #load default values if necessary
        
        if turn_angle==0: return
        
        s=structure

        if radius is None: radius=s.defaults['radius']
        if pinw is None:   pinw=s.defaults['pinw']
        if gapw is None:   gapw=s.defaults['gapw']

        if s.chip.two_layer:
            CPWBend(s.gap_layer, turn_angle, 0, pinw/2. + gapw, radius, polyarc, segments)
            CPWBend(s.pin_layer, turn_angle, 0, pinw/2., radius, polyarc, segments)
            s.last = s.gap_layer.last
            s.last_direction = s.gap_layer.last_direction
        else:
            if s.chip.solid:
                left_offset = rotate_pt((0, (pinw + gapw)/2.), s.last_direction)
                right_offset = rotate_pt((0, -(pinw + gapw)/2.), s.last_direction)
                left_gap = Structure(s.chip, direction=s.last_direction,
                                     start=(s.last[0] + left_offset[0],
                                            s.last[1] + left_offset[1]))
                right_gap = Structure(s.chip, direction=s.last_direction,
                                      start=(s.last[0] + right_offset[0],
                                             s.last[1] + right_offset[1]))

                if turn_angle >= 0:
                    ChannelBendSolid(left_gap, turn_angle=turn_angle,
                                     radius=radius-((pinw + gapw)/2.),
                                     channelw=gapw)
                    ChannelBendSolid(right_gap, turn_angle=turn_angle,
                                     radius=radius+((pinw + gapw)/2.),
                                     channelw=gapw)
                else:
                    ChannelBendSolid(left_gap, turn_angle=turn_angle,
                                     radius=radius+((pinw + gapw)/2.),
                                     channelw=gapw)
                    ChannelBendSolid(right_gap, turn_angle=turn_angle,
                                     radius=radius-((pinw + gapw)/2.),
                                     channelw=gapw)
            self.structure=structure
            self.turn_angle=turn_angle
            self.pinw=pinw
            self.gapw=gapw
            self.radius=radius
            self.segments=segments

            self.start=s.last
            self.start_angle=s.last_direction
            self.stop_angle=self.start_angle+self.turn_angle

            if turn_angle>0: self.asign=1
            else:            self.asign=-1

            #DXF uses the angle of the radial vector for its start and stop angles
            #so we have to rotate our angles by 90 degrees to get them right
            #also it only knows about arcs with CCW sense to them, so we have to rotate our angles appropriately
            self.astart_angle=self.start_angle-self.asign*90
            self.astop_angle=self.stop_angle-self.asign*90
            #calculate location of Arc center
            self.center=rotate_pt( (self.start[0],self.start[1]+self.asign*self.radius),self.start_angle,self.start)

            if polyarc: self.poly_arc_bend()
            else:       self.arc_bend()

            self.structure.last=rotate_pt(self.start,self.stop_angle-self.start_angle,self.center)
            self.structure.last_direction=self.stop_angle


    def arc_bend(self):    
            
        #print "start: %d, stop: %d" % (start_angle,stop_angle)
        
        if self.turn_angle>0:
            self.astart_angle=self.start_angle-90
            self.astop_angle=self.stop_angle-90
            #calculate location of Arc center
            self.center=rotate_pt( (self.start[0],self.start[1]+self.radius),self.start_angle,self.start)
        else:
            self.astart_angle=self.stop_angle+90
            self.astop_angle=self.start_angle+90
   
        #make endlines for inner arc
        #start first gap
        points1=[   (self.start[0],self.start[1]+self.pinw/2.),
                    (self.start[0],self.start[1]+self.pinw/2.+self.gapw)
                ]
                
        points1=rotate_pts(points1,self.start_angle,self.start)
        points2=rotate_pts(points1,self.stop_angle-self.start_angle,self.center)
        
        #start 2nd gap
        points3=[   (self.start[0],self.start[1]-self.pinw/2.),
                    (self.start[0],self.start[1]-self.pinw/2.-self.gapw)
                ]
        points3=rotate_pts(points3,self.start_angle,self.start)
        points4=rotate_pts(points3,self.stop_angle-self.start_angle,self.center)

        
        #make inner arcs
        self.structure.append(sdxf.Line(points1))
        self.structure.append(sdxf.Arc(self.center,self.radius+self.pinw/2.,self.astart_angle,self.astop_angle))
        self.structure.append(sdxf.Arc(self.center,self.radius+self.pinw/2.+self.gapw,self.astart_angle,self.astop_angle))
        self.structure.append(sdxf.Line(points2))
        
        
        self.structure.append(sdxf.Line(points3))
        self.structure.append(sdxf.Arc(self.center,self.radius-self.pinw/2.,self.astart_angle,self.astop_angle))
        self.structure.append(sdxf.Arc(self.center,self.radius-self.pinw/2.-self.gapw,self.astart_angle,self.astop_angle))
        self.structure.append(sdxf.Line(points4))            
            

    def poly_arc_bend(self):
    
        #lower gap
        pts1=arc_pts(self.astart_angle,self.astop_angle,self.radius+self.pinw/2.+self.gapw,self.segments)
        pts1.extend(arc_pts(self.astop_angle,self.astart_angle,self.radius+self.pinw/2.,self.segments))
        pts1.append(pts1[0])
       
        pts2=arc_pts(self.astart_angle,self.astop_angle,self.radius-self.pinw/2.,self.segments)
        pts2.extend(arc_pts(self.astop_angle,self.astart_angle,self.radius-self.pinw/2.-self.gapw,self.segments))
        pts2.append(pts2[0])
      
        self.structure.append(sdxf.PolyLine(translate_pts(pts1,self.center)))
        self.structure.append(sdxf.PolyLine(translate_pts(pts2,self.center)))
 
#class TaperedCPWFingerCap:
#    def __init__(self, structure,num_fingers,finger_length=None,finger_width=None,finger_gap=None,gapw=None):

class CPWWiggles:
    """CPW Wiggles (meanders)
        CPWWiggles(structure,num_wiggles,total_length,start_up=True,
                   radius=None,pinw=None,gapw=None, segments=60)
    """
    def __init__(self,structure,num_wiggles,total_length,start_up=True,radius=None,pinw=None,gapw=None, segments=60):
        """ 
            @param num_wiggles: a wiggle is from the center pin up/down and back
            @param total_length: The total length of the meander
            @param start_up: Start with a CCW 90 degree turn or a CW turn
        """
            
        s=structure
        start=structure.last
        if pinw is None:   pinw=s.defaults['pinw']
        if gapw is None:   gapw=s.defaults['gapw']
        if radius is None: radius=s.defaults['radius']

        #calculate vertical segment length:
        #total length=number of 180 degree arcs + number of vertical segs + vertical radius spacers
        #total_length=(1+num_wiggles)*(pi*radius)+2*num_wiggles*vlength+2*(num_wiggles-1)*radius
        vlength=(total_length-((1+num_wiggles)*(pi*radius)+2*(num_wiggles-1)
        *radius))/(2*num_wiggles)
        self.height = vlength + radius
        if vlength<0: print "Warning: length of vertical segments is less than 0, increase total_length or decrease num_wiggles"
        
        if start_up:  asign=1
        else:         asign=-1
        if not segments:  segments=s.defalts['segments']      
        
        CPWBend(s,asign*90,pinw,gapw,radius, segments=segments)
        for ii in range(num_wiggles):
            isign=2*(ii%2)-1
            CPWStraight(s,vlength,pinw,gapw)
            CPWBend(s,isign*asign*180,pinw,gapw,radius, segments=segments)
            CPWStraight(s,vlength,pinw,gapw)
            if ii<num_wiggles-1:
                CPWStraight(s,2*radius,pinw,gapw)
        CPWBend(s,-isign*asign*90,pinw,gapw,radius, segments=segments)

class CPWWigglesByLength:
    """An updated version of CPWWiggles which is more general.  
    Specifies a meander by length but allows for starting at different angles 
    and also allows meanders which are symmetric or asymmetric about the center pin.
    """
    def __init__(self,structure,num_wiggles,total_length,start_bend_angle=None,
                 symmetric=True,radius=None,pinw=None,gapw=None,flipped=False):
        """
            @param num_wiggles: a wiggle is from the center pin up/down and back
            @param total_length: The total length of the meander
            @param start_bend_angle: Start with a start_bend_angle degree turn (CCW)
            @param symmetric: If True then meander symmetric about current direction, other wise only above or below depending on start_bend_angle
        """

        s=structure
        start=structure.last
        if pinw is None:    pinw=s.defaults['pinw']
        if gapw is None:    gapw=s.defaults['gapw']
        if radius is None:  radius=s.defaults['radius']
        
        if num_wiggles == 0 or total_length == 0:
            self.vlength=0
            return
            

        if start_bend_angle is None:
            start_bend_angle=0
        if start_bend_angle>0:
            asign=1
        else:
            asign=-1
            
        if flipped:
            flip=-1
        else:
            flip=1
        
        if symmetric:
            vlength=(total_length-2*(abs(start_bend_angle)*pi/180*radius)-
            num_wiggles*pi*radius-2*radius*(num_wiggles-1))/(2*num_wiggles)
        else:
            vlength=(total_length-2*(abs(start_bend_angle)*pi/180*radius)-
            pi*radius*(2*num_wiggles-1))/(2*num_wiggles)

        if vlength<0:
            raise MaskError, "Warning: length of vertical segments is less than 0, increase total_length or decrease num_wiggles"

        self.vlength=vlength
                
        CPWBend(s,start_bend_angle,pinw,gapw,radius)
        for ii in range(num_wiggles):
            if symmetric:
                isign=2*(ii%2)-1
            else:
                isign=-1
                
            CPWStraight(s,vlength,pinw,gapw)
            CPWBend(s,flip*isign*asign*180,pinw,gapw,radius)
            CPWStraight(s,vlength,pinw,gapw)
            if ii<num_wiggles-1:
                if symmetric:
                    CPWStraight(s,2*radius,pinw,gapw)           #if symmetric must account for initial bend height
                else:
                    CPWBend(s,asign*180,pinw,gapw,radius)      #if asymmetric must turn around
        CPWBend(s,-flip*isign*start_bend_angle,pinw,gapw,radius)

class ChannelWigglesByLength:
    """An updated version of CPWWiggles which is more general.  
    Specifies a meander by length but allows for starting at different angles 
    and also allows meanders which are symmetric or asymmetric about the center pin.
    """
    def __init__(self,structure,num_wiggles,total_length,start_bend_angle=None,
                 symmetric=True,radius=None,channelw=None):
        """
            @param num_wiggles: a wiggle is from the center pin up/down and back
            @param total_length: The total length of the meander
            @param start_bend_angle: Start with a start_bend_angle degree turn (CCW)
            @param symmetric: If True then meander symmetric about current direction, other wise only above or below depending on start_bend_angle
        """

        s=structure
        start=structure.last
        if channelw is None:    channelw=s.defaults['pinw']+2*s.defaults['gapw']
        if radius is None:  radius=s.defaults['radius']
        
        if num_wiggles == 0 or total_length == 0:
            self.vlength=0
            return
            

        if start_bend_angle is None:
            start_bend_angle=0
        if start_bend_angle>0:
            asign=1
        else:
            asign=-1
        
        if symmetric:
            vlength=(total_length-2*(abs(start_bend_angle)*pi/180*radius)
            -num_wiggles*pi*radius-2*radius*(num_wiggles-1))/(2*num_wiggles)
        else:
            vlength=(total_length-2*(abs(start_bend_angle)*pi/180*radius)
            -pi*radius*(2*num_wiggles-1))/(2*num_wiggles)

        if vlength<0:
            raise MaskError, "Warning: length of vertical segments is less than 0, increase total_length or decrease num_wiggles"

        self.vlength=vlength
                
        ChannelBend(s,start_bend_angle,channelw=channelw,radius=radius)
        for ii in range(num_wiggles):
            if symmetric:
                isign=2*(ii%2)-1
            else:
                isign=-1
                
            CPWStraight(s,vlength,s.pinw,s.gapw)
            CPWBend(s,isign*asign*180,s.pinw,s.gapw,radius=radius)
            CPWStraight(s,vlength,s.pinw,s.gapw)
            if ii<num_wiggles-1:
                if symmetric:
                    CPWStraight(s,2*radius,s.pinw,s.gapw)           #if symmetric must account for initial bend height
                else:
                    CPWBend(s,asign*180,s.pinw,s.gapw,radius=radius)      #if asymmetric must turn around
        CPWBend(s,-isign*start_bend_angle,s.pinw,s.gapw,radius=radius)
        
class CPWWigglesByArea:
    """CPW Wiggles which fill an area specified by (length,width)"""
    def __init__(self,structure,length,width,start_up=True,radius=None,pinw=None,gapw=None):
        s=structure
        if pinw is None:
            pinw=s.defaults['pinw']
        if gapw is None:
            gapw=s.defaults['gapw']
        if radius is None:
            radius=s.defaults['radius']

        #figure out how many wiggles you can fit
        #length=2*(num_wiggles+1)*radius
        num_wiggles=int(floor(length/(2*radius)-1))
        padding=length-2*(num_wiggles+1)*radius
        vlength=(width-4*radius)/2.
        total_length=(1+num_wiggles)*(pi*radius)+2*num_wiggles*vlength+2*(num_wiggles-1)*radius
        
        self.num_wiggles=num_wiggles
        self.padding=padding
        self.vlength=vlength
        self.total_length=total_length
        self.properties= { 'num_wiggles':num_wiggles,'padding':padding,'vlength':vlength,'total_length':total_length}
        
        CPWStraight(s,padding/2.,pinw,gapw)
        CPWWiggles(s,num_wiggles,total_length,start_up,radius,pinw,gapw)
        CPWStraight(s,padding/2.,pinw,gapw)


class CPWPaddedWiggles:
    def __init__(self,structure,length,width,cpw_length,start_up=True,radius=None,pinw=None,gapw=None):
        s=structure
        if pinw is None:
            pinw=s.defaults['pinw']
        if gapw is None:
            gapw=s.defaults['gapw']
        if radius is None:
            radius=s.defaults['radius']
            
        if cpw_length<length+(2*pi-4)*radius:
            raise MaskError, "Error in CPWPaddedWiggles: cpw_length=%f needs less than one wiggle!" %(cpw_length)
        
        #calculate maximum length possible in area
        num_wiggles=int(floor(length/(2*radius)-1))
        padding=length-2*(num_wiggles+1)*radius
        vlength=(width-4*radius)/2.
        max_length=(1+num_wiggles)*(pi*radius)+2*num_wiggles*vlength+2*(num_wiggles-1)*radius
        if cpw_length > max_length:
            raise MaskError, "Error in CPWPaddedWiggles: cpw_length=%f > max_length=%f that can be fit into alotted area!" %(cpw_length,max_length)
        
        #to be finished
        
        
class ChipBorder(Structure):
    """Chip border for dicing"""
    def __init__(self,chip,border_thickness,layer="border",color=1):
        Structure.__init__(self,chip,layer=layer,color=color)
        
        chip_size=(chip.size[0]+2*border_thickness,chip.size[1]+2*border_thickness)
        
        pts1=[  (0,chip_size[1]),
                (chip_size[0],chip_size[1]),
                (chip_size[0],chip_size[1]-border_thickness),
                (0,chip_size[1]-border_thickness),
                (0,chip_size[1])
                ]
        pts1=translate_pts(pts1,(-border_thickness,-border_thickness))
        
        pts2=[  (0,0),
                (chip_size[0],0),
                (chip_size[0],border_thickness),
                (0,border_thickness),
                (0,0)
                ]
        pts2=translate_pts(pts2,(-border_thickness,-border_thickness))

        pts3=[  (0,border_thickness),
                (border_thickness,border_thickness),
                (border_thickness,chip_size[1]-border_thickness),
                (0,chip_size[1]-border_thickness),
                (0,border_thickness)
                ]
        pts3=translate_pts(pts3,(-border_thickness,-border_thickness))

                
        pts4=[  (chip_size[0]-border_thickness,border_thickness),
                (chip_size[0],border_thickness),
                (chip_size[0],chip_size[1]-border_thickness),
                (chip_size[0]-border_thickness,chip_size[1]-border_thickness),
                (chip_size[0]-border_thickness,border_thickness)
                ]
        pts4=translate_pts(pts4,(-border_thickness,-border_thickness))

        self.append(sdxf.PolyLine(pts1))
        self.append(sdxf.PolyLine(pts2))
        self.append(sdxf.PolyLine(pts3))
        self.append(sdxf.PolyLine(pts4))
        
class DashedChipBorder(Structure):
    """Dashed Chip border for e-beam drawing and then dicing"""
    def __init__(self,chip,border_thickness,dash_width=40,dash_length=200,layer="border",color=1):
        Structure.__init__(self,chip,layer=layer,color=color)

        '''Caution: border_thickness refers to the bid dicing border. Other quantities refer to dashes.'''        
        
        chip_size=(chip.size[0]+2*border_thickness,chip.size[1]+2*border_thickness)
        
        #Top
        pts1=[  (chip_size[0]/2.-dash_length/2.,chip_size[1]),
              (chip_size[0]/2.+dash_length/2.,chip_size[1]),
              (chip_size[0]/2.+dash_length/2.,chip_size[1]-dash_width/2.),
              (chip_size[0]/2.-dash_length/2.,chip_size[1]-dash_width/2.),
              (chip_size[0]/2.-dash_length/2.,chip_size[1]),
                ]
        pts1=translate_pts(pts1,(-border_thickness,-border_thickness))
        
        #Bottom
        pts2=[  (chip_size[0]/2.-dash_length/2.,0),
                (chip_size[0]/2.+dash_length/2.,0),
                (chip_size[0]/2.+dash_length/2.,dash_width/2.),
                (chip_size[0]/2.-dash_length/2.,dash_width/2.),
                (chip_size[0]/2.-dash_length/2.,0),
                ]
        pts2=translate_pts(pts2,(-border_thickness,-border_thickness))

        #Left
        pts3=[  (0,chip_size[1]/2.-dash_length/2.),
                (0,chip_size[1]/2.+dash_length/2.),
                (dash_width/2.,chip_size[1]/2.+dash_length/2.),
                (dash_width/2.,chip_size[1]/2.-dash_length/2.),
                (0,chip_size[1]/2.-dash_length/2.),
                ]
        pts3=translate_pts(pts3,(-border_thickness,-border_thickness))
        
        #Right
        pts4=[  (chip_size[0],chip_size[1]/2.-dash_length/2.),
              (chip_size[0],chip_size[1]/2.+dash_length/2.),
              (chip_size[0]-dash_width/2.,chip_size[1]/2.+dash_length/2.),
              (chip_size[0]-dash_width/2.,chip_size[1]/2.-dash_length/2.),
              (chip_size[0],chip_size[1]/2.-dash_length/2.),
                ]
        pts4=translate_pts(pts4,(-border_thickness,-border_thickness))

        self.append(sdxf.Solid(pts1))
        self.append(sdxf.Solid(pts2))
        self.append(sdxf.Solid(pts3))
        self.append(sdxf.Solid(pts4))

class CPWGapCap:
    """A CPW gap capacitor (really just a gap in the CPW center pin with no padding)"""
    def __init__(self, gap,pinw=None,gapw=None,capacitance=0.0):
        self.type='gap'
        self.gap=gap
        self.pinw=pinw
        self.gapw=gapw
        self.capacitance=capacitance
        self.length=gap
        
    def description(self):
        return "Type:\t%s\tAssumed Capacitance:\t%f\tGap Distance:\t%f\tPin Width:\t%f\t,Gap Width:\t%f\t" % (
                self.type,self.capacitance,self.gap,self.pinw,self.gapw
                )
        
    def draw(self,structure):
        s=structure
        start=structure.last
        
        if self.pinw is None: self.pinw=structure.defaults['pinw']
        if self.gapw is None: self.gapw=structure.defaults['gapw']
        
        pinw=self.pinw
        gapw=self.gapw
            
##        gpoints=[   (start[0],start[1]+pinw/2.+gapw),
##                    (start[0]+self.gap,start[1]+pinw/2.+gapw),
##                    (start[0]+self.gap,start[1]-pinw/2.-gapw),
##                    (start[0],start[1]-pinw/2.-gapw),
##                    (start[0],start[1]+pinw/2.+gapw)
##                ]
##                
##        gpoints=rotate_pts(gpoints,s.last_direction,start)

        gpoints=[   (0,pinw/2.+gapw),
                    (self.gap,pinw/2.+gapw),
                    (self.gap,-pinw/2.-gapw),
                    (0,-pinw/2.-gapw),
                    (0,pinw/2.+gapw)
                ]
                
        gpoints=orient_pts(gpoints,s.last_direction,start)



        #create polylines and append to drawing
        s.append(sdxf.PolyLine(gpoints))
          
        #update last anchor position
        #stop=rotate_pt((start[0]+self.gap,start[1]),s.last_direction,start)
        s.last=orient_pt((self.gap,0),s.last_direction,start)
        
    def ext_Q(frequency,impedance=50,resonator_type=0.5):
        if self.capacitance==0: 
            return 0
        frequency=frequency*1e9
        q=2.*pi*frequency*self.capacitance*impedance
        Q=0
        if q!=0:  
            Q=resonator_type*pi*1/(q**2)
        return Q




class CPWInductiveShunt:
    """An inductive shunt"""
    def __init__(self,num_segments, segment_length, segment_width, segment_gap, taper_length = 0,  pinw=None, inductance = 0.0):
        self.type='inductive shunt'
        self.inductance = inductance
        self.num_segments = num_segments
        self.segment_length = segment_length
        self.segment_width = segment_width
        self.segment_gap = segment_gap
        self.taper_length = taper_length

        self.pinw=pinw
        #self.gapw=gapw
        
        if (num_segments >0 ):
            self.gapw = (num_segments+1)*segment_gap+num_segments*segment_width
        else:
            self.gapw = segment_length
        
        
    def description(self):
        #print self.type,self.inductance,self.num_segments,self.segment_length,self.segment_width,self.segment_gap,self.pinw,self.gapw
        return "type:\t%s\tAssumed Inductance:\t%f pH\t# of segments:\t%d\tSegment length:\t%f\tSegment width:\t%f\tSegment gap:\t%f\tTotal inductor length:\t%f\tPin width:\t%f\tGap width:\t%f\tTaper length:\t%f" % (
            self.type,self.inductance*1e12,self.num_segments,self.segment_length,self.segment_width,self.segment_gap,self.segment_length*self.num_segments+(self.num_segments+1)*self.segment_gap,self.pinw,self.gapw,self.taper_length
            )

    def draw(self,structure,pad_to_length = 0, flipped= False):
        s=structure
        if self.pinw is None: self.pinw=s.defaults['pinw']
        pinw=self.pinw
        gapw=self.gapw
        
        self.flipped = flipped
        if pad_to_length < self.segment_length+self.taper_length: 
            self.padding=0
        else:
            self.padding=pad_to_length-self.segment_length-self.taper_length

        if not self.flipped: CPWStraight(s,self.padding)
        CPWLinearTaper(s,length=self.taper_length,start_pinw=s.defaults['pinw'],start_gapw=s.defaults['gapw'],stop_pinw=pinw,stop_gapw=gapw)
        start=structure.last
        
        if self.num_segments >0:
            gap = [ (0,0), (self.segment_length-self.segment_width,0), (self.segment_length-self.segment_width,self.segment_gap), (0,self.segment_gap), (0,0) ]
        
            gaps=[]
            if self.flipped:
                flipped=1
            else:
                flipped=0
            for ii in range (self.num_segments+1):
                gaps.append(
                    orient_pts(
                        translate_pts(gap, (self.segment_width*((ii+flipped)%2),+pinw/2.0+ii*(self.segment_gap+self.segment_width))),
                    s.last_direction,start)
                )
                
                gaps.append(
                    orient_pts(
                        translate_pts(gap,(self.segment_width*((ii+flipped)%2),-(pinw/2.0+self.segment_gap+ii*(self.segment_gap+self.segment_width)))),
                    s.last_direction,start)
                )
            
            
            for pts in gaps:
                s.append(sdxf.PolyLine(pts))
            s.last=orient_pt((self.segment_length,0),s.last_direction,start)
        else:       #If num_segments == 0 then 
            ugap1 = [ (0,pinw/2.), (0, pinw/2.+self.segment_length), (self.segment_gap, pinw/2.+self.segment_length), (self.segment_gap, pinw/2.), (0,pinw/2.0) ]
            ugap2 = translate_pts(ugap1,(self.segment_width+self.segment_gap,0))
            lgap1 = mirror_pts(ugap1,0,(self.segment_width+self.segment_gap,0))
            lgap2 = mirror_pts(ugap2,0,(self.segment_width+self.segment_gap,0))
            
            ugap1 = orient_pts(ugap1,s.last_direction,s.last)
            ugap2 = orient_pts(ugap2,s.last_direction,s.last)
            lgap1 = orient_pts(lgap1,s.last_direction,s.last)
            lgap2 = orient_pts(lgap2,s.last_direction,s.last)
            
            for pts in [ugap1,ugap2,lgap1,lgap2]:
                s.append(sdxf.PolyLine(pts))
            s.last=orient_pt((2*self.segment_gap+self.segment_width,0),s.last_direction,s.last)
            
        CPWLinearTaper(s,length=self.taper_length,start_pinw=pinw,start_gapw=gapw,stop_pinw=s.defaults['pinw'],stop_gapw=s.defaults['gapw'])
        if self.flipped: CPWStraight(s,self.padding)
        
    def ext_Q (self,frequency, impedance=50, resonator_type=0.5):
        if (self.inductance !=0):
            if resonator_type==0.5:
                return (2/pi)*(impedance/(self.inductance*2*pi*frequency*1e9))**2
            if resonator_type==0.25:
                return (2./pi)*(impedance/(2*pi*frequency*1e9*self.inductance))**2
        else:
            return 0.0
        
def rectangle_points(size,orientation=0,center=(0,0)):
    return orient_pts([ (-size[0]/2.,-size[1]/2.),(size[0]/2.,-size[1]/2.),(size[0]/2.,size[1]/2.),(-size[0]/2.,size[1]/2.),(-size[0]/2.,-size[1]/2.)],orientation,center)
    
    
class CPWFingerCap:
    """A CPW finger capacitor"""
    def __init__(self,num_fingers,finger_length,finger_width,finger_gap,taper_length = 0, gapw=None,capacitance=0.0):
        self.type='finger'
        self.capacitance=capacitance        #simulated capacitance
        self.num_fingers=num_fingers        #number of fingers
        if num_fingers<2:
            raise MaskError, "CPWFingerCap must have at least 2 fingers!"
        self.finger_length=finger_length    #length of fingers
        self.finger_width=finger_width      #width of each finger
        self.finger_gap=finger_gap
        self.gapw = gapw                    #gap between "center pin" and gnd planes        
        self.pinw = num_fingers*finger_width+ (num_fingers-1)*finger_gap    #effective center pin width sum of finger gaps and widths
        self.length=finger_length+finger_gap
        self.taper_length=taper_length
        self.total_length=finger_length+finger_gap+2.*taper_length
    
    def description(self):
        return "type:\t%s\tAssumed Capacitance:\t%f\t# of fingers:\t%d\tFinger Length:\t%f\tFinger Width:\t%f\tFinger Gap:\t%f\tTotal Pin Width:\t%f\tGap Width:\t%f\tTaper Length:\t%f" % (
                self.type,self.capacitance*1e15,self.num_fingers,self.finger_length,
                self.finger_width,self.finger_gap,self.pinw,self.gapw,self.taper_length
                )

    def draw(self,structure):
        s=structure
        pinw=self.pinw
        if self.gapw is None: self.gapw=self.pinw*s.defaults['gapw']/s.defaults['pinw']
        gapw=self.gapw
        
        CPWLinearTaper(structure,length=self.taper_length,start_pinw=s.defaults['pinw'],
                       start_gapw=s.defaults['gapw'],stop_pinw=pinw,stop_gapw=gapw)
        
        start=structure.last
        
        
        center_width=self.num_fingers*self.finger_width+ (self.num_fingers-1)*self.finger_gap
        length=self.finger_length+self.finger_gap
        
        if s.chip.two_layer:
            width = (center_width/2.) + gapw
            gap_pts = [(0, -width),
                       (length, -width),
                       (length, +width),
                       (0, +width),
                       (0, -width)]
            gap_pts = orient_pts(gap_pts, s.last_direction, start)
            if s.chip.solid:
                s.gap_layer.append(sdxf.Solid(gap_pts[:-1]))
            else:
                s.gap_layer.append(sdxf.PolyLine(gap_pts))
            # TODO: Pin Layer for TLS                    
        else:    
            gap1=[  (start[0],start[1]-center_width/2.),
                    (start[0]+length,start[1]-center_width/2.),
                    (start[0]+length,start[1]-center_width/2.-gapw),
                    (start[0],start[1]-center_width/2.-gapw),
                    (start[0],start[1]-center_width/2.)
                ]
    
            gap2=[  (start[0],start[1]+center_width/2.),
                    (start[0]+length,start[1]+center_width/2.),
                    (start[0]+length,start[1]+center_width/2.+gapw),
                    (start[0],start[1]+center_width/2.+gapw),
                    (start[0],start[1]+center_width/2.)
                ]
    
    
            gap1=rotate_pts(gap1,s.last_direction,start)
            gap2=rotate_pts(gap2,s.last_direction,start)

            
            
            if s.chip.solid:
                s.append(sdxf.Solid(gap1[:-1]))
                s.append(sdxf.Solid(gap2[:-1]))
            else:
                s.append(sdxf.PolyLine(gap1))
                s.append(sdxf.PolyLine(gap2))
        
        stop=rotate_pt((start[0]+length,start[1]),s.last_direction,start)
        s.last=stop
        
        #draw finger gaps
        if s.chip.two_layer:
            finger = [(0,0), (0,self.finger_width),
                      (self.finger_length, self.finger_width),
                      (self.finger_length, 0), (0,0)]
            finger = translate_pts(finger, start)
            for ii in range(self.num_fingers):
                if ii % 2 == 0: # Right side
                    offset = (self.finger_gap, 
                              (ii *(self.finger_width + self.finger_gap)) - (center_width/2.))
                else: # Left side
                    offset = (0,
                              (ii *(self.finger_width + self.finger_gap)) - (center_width/2.))
                new_pts = orient_pts(finger, s.last_direction, offset)
                if s.chip.solid:
                    s.pin_layer.append(sdxf.Solid(new_pts[:-1]))
                else:
                    s.pin_layer.append(sdxf.PolyLine(new_pts))
        else:
            for ii in range(self.num_fingers-1):
                if ii%2==0:
                    pts=self.left_finger_points(self.finger_width,self.finger_length,self.finger_gap)
                else:
                    pts=self.right_finger_points(self.finger_width,self.finger_length,self.finger_gap)
                pts=translate_pts(pts,start)
                pts=translate_pts(pts,(0,ii*(self.finger_width+self.finger_gap)-self.pinw/2.))
                pts=rotate_pts(pts,s.last_direction,start)
                if s.chip.solid:
                    s.append(sdxf.Solid(pts[:-1]))
                else:
                    s.append(sdxf.PolyLine(pts))

            #draw last little box to separate sides
            pts = [ (0,0),(0,self.finger_width),(self.finger_gap,self.finger_width),(self.finger_gap,0),(0,0)]
            pts=translate_pts(pts,start)
            #if odd number of fingers add box on left otherwise on right
            pts=translate_pts(pts,( ((self.num_fingers+1) %2)*(length-self.finger_gap),(self.num_fingers-1)*(self.finger_width+self.finger_gap)-self.pinw/2.))
            pts=rotate_pts(pts,s.last_direction,start)
            s.append(sdxf.PolyLine(pts))
            
        CPWLinearTaper(s,length=self.taper_length,start_pinw=pinw,start_gapw=gapw,stop_pinw=s.defaults['pinw'],stop_gapw=s.defaults['gapw'])
   
        
    def left_finger_points(self,finger_width,finger_length,finger_gap):      
        pts= [  (0,0),
                (0,finger_width+finger_gap),
                (finger_length+finger_gap,finger_width+finger_gap),
                (finger_length+finger_gap,finger_width),
                (finger_gap,finger_width),
                (finger_gap,0),
                (0,0)
            ]
                
        return pts
        
    def right_finger_points(self,finger_width,finger_length,finger_gap):         
        pts = [ (finger_length+finger_gap,0),
                (finger_length+finger_gap,finger_width+finger_gap),
                (0,finger_width+finger_gap),
                (0,finger_width),
                (finger_length,finger_width),
                (finger_length,0),
                (finger_length+finger_gap,0)
                ]
        return pts
    
    def ext_Q(self,frequency,impedance=50,resonator_type=0.5):
        if self.capacitance==0: 
            return 0
        frequency=frequency*1e9
        q=2.*pi*frequency*self.capacitance*impedance
        Q=0
        if q!=0:
            Q=1/(resonator_type*pi) *1/ (q**2)
        return Q
    
#Outside and Inside versions are for two layer capacitors

class CPWFingerCapInside:
    """A CPW finger capacitor"""
    def __init__(self,num_fingers,finger_length,finger_width,finger_gap,taper_length = 0, gapw=None,capacitance=0.0):
        self.type='finger'
        self.capacitance=capacitance        #simulated capacitance
        self.num_fingers=num_fingers        #number of fingers
        if num_fingers<2:
            raise MaskError, "CPWFingerCap must have at least 2 fingers!"
        self.finger_length=finger_length    #length of fingers
        self.finger_width=finger_width      #width of each finger
        self.finger_gap=finger_gap
        self.gapw = gapw                    #gap between "center pin" and gnd planes        
        self.pinw = num_fingers*finger_width+ (num_fingers-1)*finger_gap    #effective center pin width sum of finger gaps and widths
        self.length=finger_length+finger_gap
        self.taper_length=taper_length
        self.total_length=finger_length+finger_gap+2.*taper_length
    
    def description(self):
        return "type:\t%s\tAssumed Capacitance:\t%f\t# of fingers:\t%d\tFinger Length:\t%f\tFinger Width:\t%f\tFinger Gap:\t%f\tTotal Pin Width:\t%f\tGap Width:\t%f\tTaper Length:\t%f" % (
                self.type,self.capacitance*1e15,self.num_fingers,self.finger_length,self.finger_width,self.finger_gap,self.pinw,self.gapw,self.taper_length
                )

    def draw(self,structure):
        s=structure
        pinw=self.pinw
        if self.gapw is None: self.gapw=self.pinw*s.defaults['gapw']/s.defaults['pinw']
        gapw=self.gapw
        
        CPWLinearTaperInside(structure,length=self.taper_length,start_pinw=s.defaults['pinw'],start_gapw=s.defaults['gapw'],stop_pinw=pinw,stop_gapw=gapw)
        
        start=structure.last
        
        
        center_width=self.num_fingers*self.finger_width+ (self.num_fingers-1)*self.finger_gap
        length=self.finger_length+self.finger_gap
        
        gap1=[  (start[0],start[1]-center_width/2.),
                (start[0]+length,start[1]-center_width/2.)
            ]

        gap2=[  (start[0],start[1]+center_width/2.),
                (start[0]+length,start[1]+center_width/2.)
            ]

        gap1=rotate_pts(gap1,s.last_direction,start)
        gap2=rotate_pts(gap2,s.last_direction,start)
        stop=rotate_pt((start[0]+length,start[1]),s.last_direction,start)
        s.last=stop

        s.append(sdxf.PolyLine(gap1))
        s.append(sdxf.PolyLine(gap2))

        #draw finger gaps
        for ii in range(self.num_fingers-1):
            if ii%2==0:
                pts=self.left_finger_points(self.finger_width,self.finger_length,self.finger_gap)
            else:
                pts=self.right_finger_points(self.finger_width,self.finger_length,self.finger_gap)
            pts=translate_pts(pts,start)
            pts=translate_pts(pts,(0,ii*(self.finger_width+self.finger_gap)-self.pinw/2.))
            pts=rotate_pts(pts,s.last_direction,start)
            s.append(sdxf.PolyLine(pts))

        #draw last little box to separate sides
        pts = [ (0,0),(0,self.finger_width),(self.finger_gap,self.finger_width),(self.finger_gap,0),(0,0)]
        pts=translate_pts(pts,start)
        #if odd number of fingers add box on left otherwise on right
        pts=translate_pts(pts,( ((self.num_fingers+1) %2)*(length-self.finger_gap),(self.num_fingers-1)*(self.finger_width+self.finger_gap)-self.pinw/2.))
        pts=rotate_pts(pts,s.last_direction,start)
        s.append(sdxf.PolyLine(pts))
        
        CPWLinearTaperInside(s,length=self.taper_length,start_pinw=pinw,start_gapw=gapw,stop_pinw=s.defaults['pinw'],stop_gapw=s.defaults['gapw'])
   
        
    def left_finger_points(self,finger_width,finger_length,finger_gap):      
        pts= [  (0,0),
                (0,finger_width+finger_gap),
                (finger_length+finger_gap,finger_width+finger_gap),
                (finger_length+finger_gap,finger_width),
                (finger_gap,finger_width),
                (finger_gap,0),
                (0,0)
            ]
                
        return pts
        
    def right_finger_points(self,finger_width,finger_length,finger_gap):         
        pts = [ (finger_length+finger_gap,0),
                (finger_length+finger_gap,finger_width+finger_gap),
                (0,finger_width+finger_gap),
                (0,finger_width),
                (finger_length,finger_width),
                (finger_length,0),
                (finger_length+finger_gap,0)
                ]
        return pts
    
    def ext_Q(self,frequency,impedance=50,resonator_type=0.5):
        if self.capacitance==0: 
            return 0
        frequency=frequency*1e9
        q=2.*pi*frequency*self.capacitance*impedance
        Q=0
        if q!=0:
            Q=1/(resonator_type*pi) *1/ (q**2)
        return Q
        


class CPWLCoupler:
    """A structure which is coupled to a CPW via an L coupler, used for medium to high Q hangers"""
    def __init__(self,coupler_length,separation,flipped=False,padding_type=None,pad_to_length=None,pinw=None,gapw=None,radius=None,spinw=None,sgapw=None,capacitance=0.0):
        self.type='L'
        self.coupler_length=coupler_length
        self.separation=separation
        self.padding_type=padding_type
        self.pad_to_length=pad_to_length
        self.pinw=pinw
        self.gapw=gapw
        self.radius=radius
        self.spinw=spinw
        self.sgapw=sgapw
        self.capacitance=capacitance
        self.flipped=flipped

    def description(self):
        return "Type:\t%s\tEstimated Capacitance:\t%f\tCoupler Length:\t%f\tCoupler Separation:\t%f\tPin Width:\t%f\tGap Width:\t%f\tRadius:\t%f\tFeedline Pin Width:\t%f\tFeedline Gap Width:\t%f\t" % (
                self.type,self.capacitance,self.coupler_length,self.separation,self.pinw,self.gapw,self.radius,self.spinw,self.sgapw
                )

    def draw(self,structure,padding_type=None, flipped=None, pad_to_length=0):
        """Draws the coupler and creates the new structure (self.coupled_structure) for building onto"""
        s=structure
        if self.pinw is None:    self.pinw=s.defaults['pinw']
        if self.gapw is None:    self.gapw=s.defaults['gapw']
        if self.radius is None:  self.radius=s.defaults['radius']
        self.padding_type=padding_type
        self.pad_to_length=pad_to_length
        self.spinw=s.defaults['pinw']
        self.sgapw=s.defaults['gapw']
        
        start=s.last
        start_dir=s.last_direction
        lstart_dir=start_dir+180
        
        if flipped is None: flipped = self.flipped
        flip_sign=-1 if flipped else 1
        
        offset_length=0
        if padding_type=='center': offset_length=pad_to_length/2.
        lstart=(offset_length+self.coupler_length+self.gapw+self.radius,flip_sign*self.separation)
        if padding_type=='right':  lstart=(pad_to_length-self.gapw,lstart[1])

        lstart=rotate_pt(lstart,start_dir)
        lstart=translate_pt(lstart,start)
        
        self.coupled_structure=Structure(s.chip,start=lstart,direction=lstart_dir,layer=s.layer,color=s.color,defaults=s.defaults)
        cs=self.coupled_structure
        cs.defaults['pinw']=self.pinw
        cs.defaults['gapw']=self.gapw
        cs.defaults['radius']=self.radius

        #Continue the feedline
        self.feed_length=self.coupler_length+self.radius
        if (not self.pad_to_length is None) and (self.pad_to_length > self.feed_length):
            self.feed_length=self.pad_to_length
                    
        CPWStraight(s,self.feed_length,self.spinw,self.sgapw)

        #make the coupler
        CPWGapCap(gap=self.gapw).draw(cs)
        CPWStraight(cs,self.coupler_length)
        CPWBend(cs,-90*flip_sign)
        
        return cs
        
    def ext_Q(self,frequency,impedance=50,resonator_type=0.5):
        if self.capacitance==0: 
            return 0
        frequency=frequency*1e9
        q=2.*pi*frequency*self.capacitance*impedance
        Q=0
        if q!=0:  
            Q=resonator_type*pi*1/(q**2)
        return Q
                
class ChannelCouplerLayer:
    """This is the channel version of the CPWLCoupler for use in two layer masks."""
    """A structure which is coupled to a CPW via an L coupler, used for medium to high Q hangers"""
    def __init__(self,coupler_length,separation,flipped=False,padding_type=None,pad_to_length=None,pinw=None,gapw=None,radius=None,spinw=None,sgapw=None,capacitance=0.0,L2=False):
        self.type='L'
        self.coupler_length=coupler_length
        self.separation=separation
        self.padding_type=padding_type
        self.pad_to_length=pad_to_length
        self.pinw=pinw
        self.gapw=gapw
        self.radius=radius
        self.spinw=spinw
        self.sgapw=sgapw
        self.capacitance=capacitance
        self.flipped=flipped
        self.L2=L2

    def description(self):
        return "Type:\t%s\tEstimated Capacitance:\t%f\tCoupler Length:\t%f\tCoupler Separation:\t%f\tPin Width:\t%f\tGap Width:\t%f\tRadius:\t%f\tFeedline Pin Width:\t%f\tFeedline Gap Width:\t%f\t" % (
                self.type,self.capacitance,self.coupler_length,self.separation,self.pinw,self.gapw,self.radius,self.spinw,self.sgapw
                )

    def draw(self,structure,padding_type=None,pad_to_length=0):
        """Draws the coupler and creates the new structure (self.coupled_structure) for building onto"""
        s=structure
        if self.pinw is None:    self.pinw=s.defaults['pinw']
        if self.gapw is None:    self.gapw=s.defaults['gapw']
        if self.radius is None:  self.radius=s.defaults['radius']
        self.padding_type=padding_type
        self.pad_to_length=pad_to_length
        self.spinw=s.defaults['pinw']
        self.sgapw=s.defaults['gapw']
        if self.L2:
            self.channelw=self.pinw
        else:
            self.channelw=self.pinw+2*self.gapw
        
        start=s.last
        start_dir=s.last_direction
        lstart_dir=start_dir+180
        
        if self.flipped: flip_sign=-1
        else:            flip_sign=1
        
        offset_length=0
        if padding_type=='center': offset_length=pad_to_length/2.
        lstart=(offset_length+self.coupler_length+self.gapw+self.radius,flip_sign*self.separation)
        if padding_type=='right':  lstart=(pad_to_length-self.gapw,lstart[1])

        lstart=rotate_pt(lstart,start_dir)
        lstart=translate_pt(lstart,start)
        
        self.coupled_structure=Structure(s.chip,start=lstart,direction=lstart_dir,layer=s.layer,color=s.color,defaults=s.defaults)
        cs=self.coupled_structure
        cs.defaults['pinw']=self.pinw
        cs.defaults['gapw']=self.gapw
        cs.defaults['radius']=self.radius

        #Continue the feedline
        self.feed_length=self.coupler_length+self.radius
        if (not self.pad_to_length is None) and (self.pad_to_length > self.feed_length):
            self.feed_length=self.pad_to_length
                    
        Channel(s,self.feed_length,channelw=self.channelw)

        #make the coupler
        #CPWGapCap(gap=self.gapw).draw(cs)
        Channel(cs,self.coupler_length+self.gapw,channelw=self.channelw)
        ChannelBendSolid(cs,-90*flip_sign,channelw=self.channelw)
        
    def ext_Q(self,frequency,impedance=50,resonator_type=0.5):
        if self.capacitance==0: 
            return 0
        frequency=frequency*1e9
        q=2.*pi*frequency*self.capacitance*impedance
        Q=0
        if q!=0:  
            Q=resonator_type*pi*1/(q**2)
        return Q

class CPWTee(Structure):
    """CPWTee makes a Tee structure with padding"""
    def __init__(self,structure,stub_length=None,feed_length=None,flipped=False,pinw=None,gapw=None,spinw=None,sgapw=None):
        """
        stub_length is from center
        flipped determines whether stub is on left or right of wrt current direction
        pinw/gapw are the usual for the stub
        spinw/sgapw are the usual for the continuing part
        """
        s=structure
        #print sgapw
        if pinw is None: pinw=s.defaults['pinw']
        if gapw is None: gapw=s.defaults['gapw']
        if spinw is None: spinw=s.defaults['pinw']
        if sgapw is None: sgapw=s.defaults['gapw']
        #print "pinw: %f, gapw: %f, spinw: %f, sgapw: %f" % (pinw,gapw,spinw,sgapw)

        #minimum feed_length is
        if (feed_length is None) or (feed_length < 2*gapw+pinw): 
            feed_length=2*gapw+pinw
        
        #minimum stub_length is 
        if (stub_length is None) or (stub_length < gapw+spinw):  
            stub_length=gapw+spinw/2.
        #print "pinw: %f, gapw: %f, spinw: %f, sgapw: %f" % (pinw,gapw,spinw,sgapw)
            
        start=s.last
        start_dir=s.last_direction
        
        if flipped: 
            lstart_dir=start_dir-90
            angle=start_dir+180
        else:       
            lstart_dir=start_dir+90
            angle=start_dir
        
        #Bottom part of feed_line
        pts1=[ (-feed_length/2.,-spinw/2.), (-feed_length/2.,-sgapw-spinw/2.0),  (feed_length/2.,-sgapw-spinw/2.0),(feed_length/2.,-spinw/2.),  (-feed_length/2.,-spinw/2.)]

        #Top of feed_line
        pts2=[ (-feed_length/2.,spinw/2.), (-pinw/2.-gapw,spinw/2.), (-pinw/2.-gapw,gapw+spinw/2.), (-feed_length/2.,gapw+spinw/2.), (-feed_length/2,spinw/2.) ]
        pts3=[ (feed_length/2.,spinw/2.), (pinw/2.+gapw,spinw/2.), (pinw/2.+gapw,gapw+spinw/2.), (feed_length/2.,gapw+spinw/2.), (feed_length/2,spinw/2.) ]
        #stub
        pts4=[ (-pinw/2.,spinw/2.), (-pinw/2.,stub_length), (-pinw/2.-gapw,stub_length), (-pinw/2.-gapw,spinw/2.), (-pinw/2.,spinw/2.) ]
        pts5=[ (pinw/2.,spinw/2.), (pinw/2.,stub_length), (pinw/2.+gapw,stub_length), (pinw/2.+gapw,spinw/2.), (pinw/2.,spinw/2.) ]

        shapes=[pts1,pts2,pts3,pts4,pts5]

        center=orient_pt((feed_length/2.,0),s.last_direction,s.last)
        for pts in shapes:
            pts=orient_pts(pts,angle,center)
            s.append(sdxf.PolyLine(pts))
            
        s.last=orient_pt((feed_length,0),s.last_direction,s.last)
        lstart=orient_pt((stub_length,0),lstart_dir,center)

        Structure.__init__(self,s.chip,start=lstart,direction=lstart_dir,layer=s.layer,color=s.color,defaults=s.defaults)
        self.defaults['pinw']=pinw
        self.defaults['gapw']=gapw   

class FingerCoupler(Structure):
    """Finger coupler a CPWTee plus finger capacitor...not used yet..."""
    def __init__(self,structure,cap_desc,stub_length=None,padding_length=None,flipped=False,pinw=None,gapw=None,taper_length=0,spinw=None,sgapw=None):
        CPWTee.__init__(structure,stub_length,padding_length,flipped,spinw,sgapw)
        if pinw is None: pinw=structure['pinw']
        if gapw is None: gapw=structure['gapw']
        
        CPWLinearTaper(self,taper_length,self.defaults['pinw'],cap_desc.pinw,self.defaults['gapw'],cap_desc.gapw)
        cap_desc.draw_cap(self)
        CPWLinearTaper(self,taper_length,cap_desc.pinw,pinw,cap_desc.gapw,gapw)
        
class CapStar:
    """A straight section of CPW transmission line"""
    def __init__(self,structure,number,cap,pinw=None,gapw=None):

        self.structure = structure
        self.number = number
        self.cap = cap
        if pinw is None: pinw=structure.defaults['pinw']
        if gapw is None: gapw=structure.defaults['gapw']
        self.pinw = pinw
        self.gapw = gapw
        
        self.start = structure.last
        self.defaults = structure.defaults
        
        
    def draw(self):
    
        s=self.structure
        start=self.start
        
        pinw = self.pinw
        gapw = self.gapw
        cap = self.cap
        number = self.number
        
        cap_width=cap.num_fingers*cap.finger_width+(cap.num_fingers-1)*cap.finger_gap+2*cap.pinw*s.defaults['gapw']/s.defaults['pinw']
        taper_length=cap.taper_length
        #phi is angle capacitor taper makes
        phi = atan((cap_width-pinw-2*gapw)/(2.*taper_length))
        phi_deg = degrees(phi)
        outer_taper_length = taper_length/cos(phi)

        #print cap_width, phi

        angle_deg = 360./number
        angle = radians(angle_deg)
        self.angle=angle
        self.angle_deg = angle_deg
        
        #keeps track of end of cap side to make sure doesnt intersect itself
        length = 0.
        final_x = length*sin(angle/2.)+outer_taper_length*sin(angle/2.-phi)
        
        while final_x < 7:
            final_x = length*sin(angle/2.)+outer_taper_length*sin(angle/2.-phi)
            length = length+5

        length_x = length*sin(angle/2.)
        length_y = length*cos(angle/2.)
        
        
        outside = [ (start[0]-length_x,start[1]+(pinw/2.+gapw)/sin(angle/2.)+length_y),
                   (start[0],start[1]+(pinw/2.+gapw)/sin(angle/2.)),
                    (start[0]+length_x,start[1]+(pinw/2.+gapw)/sin(angle/2.)+length_y)
            ]
              
        length = gapw*cos(angle/2.)/sin(angle/2.)+length
        length_x = length*sin(angle/2.)
        length_y = length*cos(angle/2.)
        self.length = length
        
        
        
        inside = [ (start[0]-length_x,start[1]+(pinw/2.)/sin(angle/2.)+length_y),
                   (start[0],start[1]+(pinw/2.)/sin(angle/2.)),
                    (start[0]+length_x,start[1]+(pinw/2.)/sin(angle/2.)+length_y)
            ]

        for n in range(number):
            s.append(sdxf.PolyLine(rotate_pts(outside,n*360./number,start)))
            s.append(sdxf.PolyLine(rotate_pts(inside,n*360./number,start)))
            
        for n in range(number):
            s.start = (start[0]+(length+pinw*cos(angle/2.)/(2.*sin(angle/2.)))*sin((n+.5)*angle),start[1]+(length+pinw*cos(angle/2.)/(2.*sin(angle/2.)))*cos((n+.5)*angle))
            s.last = s.start
            s.last_direction =  90-(n+.5)*angle_deg
            cap.draw(s)
        
    def cap_ends(self):
    
        cap_ends = []
        
        for n in range(self.number):
            cap_ends.append((self.start[0]+(self.cap.total_length+self.length+self.pinw*cos(self.angle/2.)/(2.*sin(self.angle/2.)))*sin((n+.5)*self.angle),self.start[1]+(self.cap.total_length+self.length+self.pinw*cos(self.angle/2.)/(2.*sin(self.angle/2.)))*cos((n+.5)*self.angle)))
        
        return cap_ends
        
    def cap_directions(self):
    
        cap_directions = []
        
        for n in range(self.number):
            cap_directions.append(90-(n+.5)*self.angle_deg)
        
        return cap_directions
        
        
        #Still working on good way to output end of capacitor positions and directions.
        
#===============================================================================       
# NEW CLASSES FOR CHANNEL STRUCTURES & TWO-LAYER PHOTOLITHOGRAPHY    
#===============================================================================


class LShapeAlignmentMarks:
    def __init__(self,structure,width,armlength):
        """creates an L shaped alignment marker of width and armlength for photolitho"""
        if width==0: return
        if armlength==0: return
        
        s=structure
        start=s.last
        
        box1=[  (start[0]-width/2.,start[1]-width/2.),
                (start[0]+armlength-width/2.,start[1]-width/2.),
                (start[0]+armlength-width/2.,start[1]+width/2.),
                (start[0]-width/2.,start[1]+width/2.),
                (start[0]-width/2.,start[1]-width/2.)
                ]

        box2=[  (start[0]+width/2.,start[1]+width/2.),
                (start[0]+width/2.,start[1]+armlength-width/2.),
                (start[0]-width/2.,start[1]+armlength-width/2.),
                (start[0]-width/2.,start[1]+width/2.),
                (start[0]+width/2.,start[1]+width/2.)
                ]
        
                
        box1=rotate_pts(box1,s.last_direction,start)
        box2=rotate_pts(box2,s.last_direction,start)
        stop=rotate_pt((start[0]+armlength,start[1]),s.last_direction,start)
        s.last=stop

        s.append(sdxf.Solid(box1))
        s.append(sdxf.Solid(box2))
        
#---------------------------------------------------------------------------- 
class ArrowAlignmentMarks_L1:
    def __init__(self,structure,height,width,buffer=30):
        """creates an arrow/triangle of height and base width for alignment"""
        if height==0: return
        if width==0: return
        
        s=structure
        start=s.last
        
        triangle=[(start[0]+buffer,start[1]),(start[0]+buffer,start[1]+width),(start[0]+buffer+height,start[1]+width/2.),(start[0]+buffer,start[1])]
        
        triangle=rotate_pts(triangle,s.last_direction,start)
        stop=rotate_pt((start[0]+height,start[1]),s.last_direction,start)
        s.last=stop

        s.append(sdxf.PolyLine(triangle))

#---------------------------------------------------------------------------- 
class ArrowAlignmentMarks_L2:
    def __init__(self,structure,height,width,buffer=30):
        """creates an arrow/triangle of height and base width for alignment"""
        if height==0: return
        if width==0: return
        
        s=structure
        start=s.last
        
        box=[(start[0],start[1]),(start[0],start[1]+width),(start[0]+buffer+height,start[1]+width),(start[0]+buffer+height,start[1]),(start[0],start[1])]
        triangle=[(start[0]+buffer+height,start[1]+width/2.),(start[0]+buffer+height+height,start[1]),(start[0]+buffer+height+height,start[1]+width),(start[0]+buffer+height,start[1]+width/2.)]
        
        box=rotate_pts(box,s.last_direction,start)
        triangle=rotate_pts(triangle,s.last_direction,start)
        
        stop=rotate_pt((start[0]+height,start[1]),s.last_direction,start)
        s.last=stop
        
        s.append(sdxf.PolyLine(box))
        s.append(sdxf.PolyLine(triangle))
        
#----------------------------------------------------------------------------
class Channel:
    """A simple channel of given width and length"""
    def __init__(self,structure,length,channelw,solid=None):
        """ Adds a channel of width=channelw and of length = length to the structure"""
        if length==0: return
        if channelw==0: return

        s=structure
        start=structure.last
            
        if solid is None:
            if s.defaults.has_key('solid') == True:
                solid = s.defaults['solid']
            else:
                solid = False
        
        ch1=[  (start[0],start[1]-channelw/2.),
                (start[0]+length,start[1]-channelw/2.),
                (start[0]+length,start[1]+channelw/2.),
                (start[0],start[1]+channelw/2.),
                (start[0],start[1]-channelw/2.)
                ]
                
        ch1=rotate_pts(ch1,s.last_direction,start)
        stop=rotate_pt((start[0]+length,start[1]),s.last_direction,start)
        s.last=stop

        if solid:
            s.append(sdxf.Solid(ch1))
        else:
            s.append(sdxf.PolyLine(ch1))

#----------------------------------------------------------------------------
class ChannelLinearTaper:
    """A section of channel which (linearly) tapers from width=start_channelw to stop_channelw over length=length"""
    def __init__(self, structure,length,start_channelw,stop_channelw):
        if length==0: return
        #load attributes
        s=structure
        start=s.last
        
        #define geometry of channel
        ch1= [ 
            (start[0],start[1]-start_channelw/2.),
            (start[0]+length,start[1]-stop_channelw/2.),
            (start[0]+length,start[1]+stop_channelw/2.),
            (start[0],start[1]+start_channelw/2.),
            (start[0],start[1]-start_channelw/2.)
            ]
        
        #rotate structure to proper orientation
        ch1=rotate_pts(ch1,s.last_direction,start)

        #create polylines and append to drawing
        s.append(sdxf.Solid(ch1))
        
        #update last anchor position
        stop=rotate_pt((start[0]+length,start[1]),s.last_direction,start)
        s.last=stop
        
#-------------------------------------------------------------------------------------------------
class ChannelLauncher:
    """creates a channel launcher with a pad of length=pad_length and width=padwidth and a taper of length=taper_length which
        linearly tapers from padwidth to channelwidth"""
    def __init__(self,structure,flipped=False,pad_length=500,taper_length=400,pad_to_length=1000,padwidth=300,channelwidth=None):
        s=structure
            
        padding = pad_to_length-pad_length-taper_length
        if padding <0: 
            padding=0
            self.length=pad_length+taper_length
        else:
            self.length=pad_to_length
        
        if not flipped:
            Channel(s,length=pad_length,channelw=padwidth)
            ChannelLinearTaper(s,length=taper_length,start_channelw=padwidth,stop_channelw=channelwidth)
            Channel(s,length=padding,channelw=channelwidth)
        else:
            Channel(s,length=padding,channelw=channelwidth)
            ChannelLinearTaper(s,length=taper_length,start_channelw=channelwidth,stop_channelw=padwidth)
            Channel(s,length=pad_length,channelw=padwidth)
            
#-------------------------------------------------------------------------------------------------
class ChannelBend:
    """A Channel bend - adapted from CPWBend"""
    def __init__(self,structure,turn_angle,channelw=None,radius=None,polyarc=True,segments=60):
        """creates a channel bend with channelw/radius
            @param turn_angle: turn_angle is in degrees, positive is CCW, negative is CW
        """
        #load default values if necessary
        
        if turn_angle==0: return
        
        s=structure

        if radius is None: radius=s.defaults['radius']
        if channelw is None:   channelw=s.defaults['channelw']
        
        self.structure=structure
        self.turn_angle=turn_angle
        self.channelw=channelw
        self.radius=radius
        self.segments=segments
        self.pinw=0
        self.gapw=channelw/2.

        self.start=s.last
        self.start_angle=s.last_direction
        self.stop_angle=self.start_angle+self.turn_angle
        
        if turn_angle>0: self.asign=1
        else:            self.asign=-1
       
        #DXF uses the angle of the radial vector for its start and stop angles
        #so we have to rotate our angles by 90 degrees to get them right
        #also it only knows about arcs with CCW sense to them, so we have to rotate our angles appropriately
        self.astart_angle=self.start_angle-self.asign*90
        self.astop_angle=self.stop_angle-self.asign*90
        #calculate location of Arc center
        self.center=rotate_pt( (self.start[0],self.start[1]+self.asign*self.radius),self.start_angle,self.start)
        
        if polyarc: self.poly_arc_bend()
        else:       self.arc_bend()

        self.structure.last=rotate_pt(self.start,self.stop_angle-self.start_angle,self.center)
        self.structure.last_direction=self.stop_angle


    def arc_bend(self):    
            
        #print "start: %d, stop: %d" % (start_angle,stop_angle)
        
        if self.turn_angle>0:
            self.astart_angle=self.start_angle-90
            self.astop_angle=self.stop_angle-90
            #calculate location of Arc center
            self.center=rotate_pt( (self.start[0],self.start[1]+self.radius),self.start_angle,self.start)
        else:
            self.astart_angle=self.stop_angle+90
            self.astop_angle=self.start_angle+90
   
        #make endlines for inner arc
        #start first gap
        #points1=[   (self.start[0],self.start[1]+self.pinw/2.),
                 #   (self.start[0],self.start[1]+self.pinw/2.+self.gapw)
                #]
                
        points1=[   (self.start[0],self.start[1]+self.gapw),
                    (self.start[0],self.start[1]-self.gapw)
                ]
        points1=rotate_pts(points1,self.start_angle,self.start)
        points2=rotate_pts(points1,self.stop_angle-self.start_angle,self.center)
        
        #start 2nd gap
        #points3=[   (self.start[0],self.start[1]-self.pinw/2.),
               #     (self.start[0],self.start[1]-self.pinw/2.-self.gapw)
               # ]
        #points3=rotate_pts(points3,self.start_angle,self.start)
        #points4=rotate_pts(points3,self.stop_angle-self.start_angle,self.center)

        
        #make inner arcs
        self.structure.append(sdxf.Line(points1))
        self.structure.append(sdxf.Arc(self.center,self.radius+self.pinw/2.,self.astart_angle,self.astop_angle))
        self.structure.append(sdxf.Arc(self.center,self.radius+self.pinw/2.+self.gapw,self.astart_angle,self.astop_angle))
        self.structure.append(sdxf.Line(points2))
        
        
        #self.structure.append(sdxf.Line(points3))
        #self.structure.append(sdxf.Arc(self.center,self.radius-self.pinw/2.,self.astart_angle,self.astop_angle))
        #self.structure.append(sdxf.Arc(self.center,self.radius-self.pinw/2.-self.gapw,self.astart_angle,self.astop_angle))
        #self.structure.append(sdxf.Line(points4))            
            

    def poly_arc_bend(self):
    
        #lower gap
        '''pts1=arc_pts(self.astart_angle,self.astop_angle,self.radius+self.pinw/2.+self.gapw,self.segments)
        pts1.extend(arc_pts(self.astop_angle,self.astart_angle,self.radius+self.pinw/2.,self.segments))
        pts1.append(pts1[0])
       
        pts2=arc_pts(self.astart_angle,self.astop_angle,self.radius-self.pinw/2.,self.segments)
        pts2.extend(arc_pts(self.astop_angle,self.astart_angle,self.radius-self.pinw/2.-self.gapw,self.segments))
        pts2.append(pts2[0])'''
        
        pts2=arc_pts(self.astart_angle,self.astop_angle,self.radius+self.gapw,self.segments)
        pts2.extend(arc_pts(self.astop_angle,self.astart_angle,self.radius-self.gapw,self.segments))
        pts2.append(pts2[0])
        
        #print pts2
      
        #self.structure.append(sdxf.PolyLine(translate_pts(pts1,self.center)))
        self.structure.append(sdxf.PolyLine(translate_pts(pts2,self.center)))
        
#-------------------------------------------------------------------------------------------------
class ChannelBendSolid:
    """A Channel bend - adapted from CPWBend"""
    def __init__(self,structure,turn_angle,channelw=None,radius=None,polyarc=True,segments=150.):
        """creates a channel bend with channelw/radius
            @param turn_angle: turn_angle is in degrees, positive is CCW, negative is CW
        """
        #load default values if necessary
        
        if turn_angle==0: return
        
        s=structure

        if radius is None: radius=s.defaults['radius']
        if channelw is None:   channelw=s.defaults['channelw']
        
        self.structure=structure
        self.turn_angle=turn_angle
        self.channelw=channelw
        self.radius=radius
        self.segments=segments
        self.pinw=0
        self.gapw=channelw/2.

        self.start=s.last
        self.start_angle=s.last_direction
        self.stop_angle=self.start_angle+self.turn_angle
        
        if turn_angle>0: self.asign=1
        else:            self.asign=-1
       
        #DXF uses the angle of the radial vector for its start and stop angles
        #so we have to rotate our angles by 90 degrees to get them right
        #also it only knows about arcs with CCW sense to them, so we have to rotate our angles appropriately
        self.astart_angle=self.start_angle-self.asign*90
        self.astop_angle=self.stop_angle-self.asign*90
        #calculate location of Arc center
        self.center=rotate_pt( (self.start[0],self.start[1]+self.asign*self.radius),self.start_angle,self.start)
        
        '''if polyarc: self.poly_arc_bend()
        else:       self.arc_bend()'''

        
        
        
        #ri is inner radius, ro is outer        
        ri=radius-channelw/2.
        ro=radius+channelw/2.                
        
        #theta is the "infinitesimal angle"
        #alpha is remaining angle of the isoscoles triangle
        theta_deg = turn_angle/self.segments
        alpha_deg = (180-theta_deg)/2.
        theta = radians(theta_deg)
        alpha = radians(alpha_deg)
        
        #a is inner arc length of small cell. b is outer arc length
        #calculated using law of sines
        a=sin(theta)*ri/sin(alpha)
        b=sin(theta)*ro/sin(alpha)
        
        #c is displacement vector to center of cirlce with respect to structure.last
        #phi is initial direction of structure in degrees
        phi_deg = structure.last_direction
        phi = radians(phi_deg)
        if self.asign > 0:
            c = (-radius*sin(phi),radius*cos(phi))
            basic = [ (ri,0),
                     (ro,0),
                (ro-b*cos(alpha),b*sin(alpha)),
                (ri-a*cos(alpha),a*sin(alpha)),
                (ri,0)
                ]
            basic = rotate_pts(basic,-90,center=(0,0))
        else:            
            alpha = -alpha
            c = (radius*sin(phi),-radius*cos(phi))
            basic = [ (-ri,0),
                     (-ro,0),
                (-ro+b*cos(alpha),b*sin(alpha)),
                (-ri+a*cos(alpha),a*sin(alpha)),
                (-ri,0)
                ]
            basic = rotate_pts(basic,-90,center=(0,0))
            self.segments=self.segments
        
        #center is what to pivot around
        cent = (self.structure.last[0]+c[0],self.structure.last[1]+c[1])
        
            
        for n in range(int(self.segments)):
            self.structure.append(sdxf.Solid(rotate_pts(translate_pts(basic,cent),n*theta_deg+structure.last_direction,cent)))
            
        
        self.structure.last=rotate_pt(self.start,self.stop_angle-self.start_angle,self.center)
        self.structure.last_direction=self.stop_angle
            
        '''self.structure.append(sdxf.Solid(translate_pts(basic,self.center)))
        self.structure.append(sdxf.Solid(rotate_pts(translate_pts(basic,self.center),theta_deg,self.center)))'''


    '''def arc_bend(self):    
            
        #print "start: %d, stop: %d" % (start_angle,stop_angle)
        
        if self.turn_angle>0:
            self.astart_angle=self.start_angle-90
            self.astop_angle=self.stop_angle-90
            #calculate location of Arc center
            self.center=rotate_pt( (self.start[0],self.start[1]+self.radius),self.start_angle,self.start)
        else:
            self.astart_angle=self.stop_angle+90
            self.astop_angle=self.start_angle+90
   
        #make endlines for inner arc
        #start first gap
        #points1=[   (self.start[0],self.start[1]+self.pinw/2.),
                 #   (self.start[0],self.start[1]+self.pinw/2.+self.gapw)
                #]
                
        points1=[   (self.start[0],self.start[1]+self.gapw),
                    (self.start[0],self.start[1]-self.gapw)
                ]
        points1=rotate_pts(points1,self.start_angle,self.start)
        points2=rotate_pts(points1,self.stop_angle-self.start_angle,self.center)
        
        #start 2nd gap
        #points3=[   (self.start[0],self.start[1]-self.pinw/2.),
               #     (self.start[0],self.start[1]-self.pinw/2.-self.gapw)
               # ]
        #points3=rotate_pts(points3,self.start_angle,self.start)
        #points4=rotate_pts(points3,self.stop_angle-self.start_angle,self.center)

        
        #make inner arcs
        self.structure.append(sdxf.Line(points1))
        self.structure.append(sdxf.Arc(self.center,self.radius+self.pinw/2.,self.astart_angle,self.astop_angle))
        self.structure.append(sdxf.Arc(self.center,self.radius+self.pinw/2.+self.gapw,self.astart_angle,self.astop_angle))
        self.structure.append(sdxf.Line(points2))
        
        
        #self.structure.append(sdxf.Line(points3))
        #self.structure.append(sdxf.Arc(self.center,self.radius-self.pinw/2.,self.astart_angle,self.astop_angle))
        #self.structure.append(sdxf.Arc(self.center,self.radius-self.pinw/2.-self.gapw,self.astart_angle,self.astop_angle))
        #self.structure.append(sdxf.Line(points4))            
            

    def poly_arc_bend(self):
    
        #lower gap
        """pts1=arc_pts(self.astart_angle,self.astop_angle,self.radius+self.pinw/2.+self.gapw,self.segments)
        pts1.extend(arc_pts(self.astop_angle,self.astart_angle,self.radius+self.pinw/2.,self.segments))
        pts1.append(pts1[0])
       
        pts2=arc_pts(self.astart_angle,self.astop_angle,self.radius-self.pinw/2.,self.segments)
        pts2.extend(arc_pts(self.astop_angle,self.astart_angle,self.radius-self.pinw/2.-self.gapw,self.segments))
        pts2.append(pts2[0])"""
        
        pts2=arc_pts(self.astart_angle,self.astop_angle,self.radius+self.gapw,self.segments)
        pts2.extend(arc_pts(self.astop_angle,self.astart_angle,self.radius-self.gapw,self.segments))
        pts2.append(pts2[0])
        
        print pts2
      
        #self.structure.append(sdxf.PolyLine(translate_pts(pts1,self.center)))
        self.structure.append(sdxf.PolyLine(translate_pts(pts2,self.center)))'''
        
#-------------------------------------------------------------------------------------------------
class ChannelWiggles:
    """Channel Wiggles (meanders) = adapted from CPWWiggles"""
    def __init__(self,structure,num_wiggles,total_length,start_up=True,radius=None,channelw=None,endbending1=True,endbending2=True,inverted=False):
        """ 
            @param num_wiggles: a wiggle is from the center pin up/down and back
            @param total_length: The total length of the meander
            @param start_up: Start with a CCW 90 degree turn or a CW turn
            @param endbending: gives you the option of wheither or not to have an additional 90 degree bend back to horizontal at the two ends
        """
            
        s=structure
        if channelw is None:   channelw=s.defaults['channelw']
        if radius is None: radius=s.defaults['radius']

        #calculate vertical segment length:
        #total length=number of 180 degree arcs + number of vertical segs + vertical radius spacers
        #total_length=(1+num_wiggles)*(pi*radius)+2*num_wiggles*vlength+2*(num_wiggles-1)*radius
        #end_bends takes into account the length added by putting on channel bends at each end
        end_bends = 0
        if endbending1 == True:
            end_bends += 1/2.
        if endbending2 == True:
            end_bends += 1/2.      
        
        vlength=(total_length-(end_bends+num_wiggles)*pi*radius-2*(num_wiggles-1)*radius)/(2*num_wiggles)

        if vlength<0: print "Warning: length of vertical segments is less than 0, increase total_length or decrease num_wiggles"
        
        if start_up:  asign=1
        else:         asign=-1
        
        if endbending1:
            ChannelBendSolid(s,asign*90,channelw,radius)
        for ii in range(num_wiggles):
            isign=2*(ii%2)-1
            if inverted:
                isign=-(2*(ii%2)-1)
            Channel(s,vlength,channelw)
            ChannelBendSolid(s,isign*asign*180,channelw,radius)
            Channel(s,vlength,channelw)
            if ii<num_wiggles-1:
                Channel(s,2*radius,channelw)
        if endbending2:
            ChannelBendSolid(s,-isign*asign*90,channelw,radius)
        
#-------------------------------------------------------------------------------------------------
class ChannelTee(Structure):
    """ChannelTee makes a Tee structure with padding"""
    def __init__(self,structure,stub_length=None,feed_length=None,flipped=False,channelw=None):
        """
        stub_length is from center
        flipped determines whether stub is on left or right of wrt current direction
        pinw/gapw are the usual for the stub
        spinw/sgapw are the usual for the continuing part
        """
        s=structure
        
        if channelw is None: channelw=s.defaults['channelw']

        #minimum feed_length is
        if (feed_length is None) or (feed_length < channelw): 
            feed_length=channelw
        
        #minimum stub_length is 
        if (stub_length is None) or (stub_length < channelw):  
            stub_length=channelw
        #print "pinw: %f, gapw: %f, spinw: %f, sgapw: %f" % (pinw,gapw,spinw,sgapw)
            
        start=s.last
        start_dir=s.last_direction
        
        if flipped: 
            lstart_dir=start_dir-90
            angle=start_dir+180
        else:       
            lstart_dir=start_dir+90
            angle=start_dir
        
        #feed_line
        pts1=[ (-feed_length/2.,-channelw/2.), (-feed_length/2.,channelw/2.),  (feed_length/2.,channelw/2.),(feed_length/2.,-channelw/2.),  (-feed_length/2.,-channelw/2.)]
        #stub
        pts2=[ (-channelw/2.,channelw/2),(-channelw/2.,stub_length),(channelw/2.,stub_length),(channelw/2.,channelw/2.),(-channelw/2.,channelw/2.) ]

        shapes=[pts1,pts2]

        center=orient_pt((feed_length/2.,0),s.last_direction,s.last)
        for pts in shapes:
            pts=orient_pts(pts,angle,center)
            s.append(sdxf.Solid(pts))
            
        s.last=orient_pt((feed_length,0),s.last_direction,s.last)
        lstart=orient_pt((stub_length,0),lstart_dir,center)

        Structure.__init__(self,s.chip,start=lstart,direction=lstart_dir,layer=s.layer,color=s.color,defaults=s.defaults)
        self.defaults['channelw']=channelw    

#---------------------------------------------------------------------------------- 
class CenterPinTee(Structure):
    """CCDChannelTee makes a Tee structure with microchannels attached"""
    def __init__(self,structure,stub_length=None,feed_length=None,flipped=False,pinw=None,gapw=None,spinw=None,sgapw=None,notchwidth=10,couplinglength=100,channelwidth=8):
        """
        stub_length is from center
        flipped determines whether stub is on left or right of wrt current direction
        pinw/gapw are the usual for the stub
        spinw/sgapw are the usual for the continuing part
        """
        s=structure
        #print sgapw
        if pinw is None: pinw=s.defaults['pinw']
        if gapw is None: gapw=s.defaults['gapw']
        if spinw is None: spinw=s.defaults['pinw']
        if sgapw is None: sgapw=s.defaults['gapw']
        #print "pinw: %f, gapw: %f, spinw: %f, sgapw: %f" % (pinw,gapw,spinw,sgapw)

        #minimum feed_length is
        if (feed_length is None) or (feed_length < 2*gapw+pinw): 
            feed_length=2*gapw+pinw
        
        #minimum stub_length is 
        if (stub_length is None) or (stub_length < gapw+spinw):  
            stub_length=gapw+spinw/2
        #print "pinw: %f, gapw: %f, spinw: %f, sgapw: %f" % (pinw,gapw,spinw,sgapw)
            
        start=s.last
        start_dir=s.last_direction
        
        if flipped: 
            lstart_dir=start_dir-90
            angle=start_dir+180
        else:       
            lstart_dir=start_dir+90
            angle=start_dir
        
        #Bottom part of feed_line
        pts1=[ (-feed_length/2.,-spinw/2.), (-feed_length/2.,-sgapw-spinw/2.0),  (feed_length/2.,-sgapw-spinw/2.0),(feed_length/2.,-spinw/2.),  (-feed_length/2.,-spinw/2.)]

        #Top of feed_line
        pts2=[ (-feed_length/2,spinw/2.), (-pinw/2.-gapw,spinw/2.), (-pinw/2.-gapw,gapw+spinw/2.), (-feed_length/2.,gapw+spinw/2.), (-feed_length/2,spinw/2.) ]
        pts3=[ (feed_length/2,spinw/2.), (pinw/2.+gapw,spinw/2.), (pinw/2.+gapw,gapw+spinw/2.), (feed_length/2.,gapw+spinw/2.), (feed_length/2,spinw/2.) ]
        #stub
        pts4=[ (-pinw/2.,spinw/2.), (-pinw/2.,stub_length), (-pinw/2.-gapw,stub_length), (-pinw/2.-gapw,spinw/2.), (-pinw/2.,spinw/2.) ]
        pts5=[ (pinw/2.,spinw/2.), (pinw/2.,stub_length), (pinw/2.+gapw,stub_length), (pinw/2.+gapw,spinw/2.), (pinw/2.,spinw/2.) ]
        pts6=[ (-pinw/2.,stub_length), (-pinw/2.,stub_length+couplinglength), (-pinw/2.-notchwidth,stub_length+couplinglength), (-pinw/2.-notchwidth,stub_length), (-pinw/2.,stub_length) ]
        pts7=[ (pinw/2.,stub_length), (pinw/2.,stub_length+couplinglength), (pinw/2.+notchwidth,stub_length+couplinglength), (pinw/2.+notchwidth,stub_length), (pinw/2.,stub_length) ]

        shapes=[pts1,pts2,pts3,pts4,pts5,pts6,pts7]
        
        center=orient_pt((0,0),s.last_direction,s.last)
        for pts in shapes:
            pts=orient_pts(pts,angle,center)
            s.append(sdxf.PolyLine(pts))
            
        s.last=orient_pt((feed_length,0),s.last_direction,s.last)
        lstart=orient_pt((stub_length,0),lstart_dir,center)

        Structure.__init__(self,s.chip,start=lstart,direction=lstart_dir,layer=s.layer,color=s.color,defaults=s.defaults)
        self.defaults['pinw']=pinw
        self.defaults['gapw']=gapw   
           
#-------------------------------------------------------------------------------------------------
class CCDChannelTee(Structure):
    """CCDChannelTee makes a tee structure with microchannels attached;
        This is the first layer structure, i.e. everything that's connected
        to the center pin of the cavity, second layer see below"""
    def __init__(self,structure,stub_length=None,feed_length=None,flipped=False,pinw=None,gapw=None,spinw=None,sgapw=None,ccdwidth=100,ccdlength=100,channelwidth=8):
        s=structure
        #print sgapw
        if pinw is None: pinw=s.defaults['pinw']
        if gapw is None: gapw=s.defaults['gapw']
        if spinw is None: spinw=s.defaults['pinw']
        if sgapw is None: sgapw=s.defaults['gapw']

        #minimum feed_length is
        if (feed_length is None) or (feed_length < 2*gapw+pinw): 
            feed_length=2*gapw+pinw
        
        #minimum stub_length is 
        if (stub_length is None) or (stub_length < gapw+spinw):  
            stub_length=gapw+spinw/2
            
        start=s.last
        start_dir=s.last_direction
        
        if flipped: 
            lstart_dir=start_dir-90
            angle=start_dir+180
        else:       
            lstart_dir=start_dir+90
            angle=start_dir
            
        
        #Bottom part of feed_line
        pts1=[ (-feed_length/2.,-spinw/2.), (-feed_length/2.,-sgapw-spinw/2.0),  (feed_length/2.,-sgapw-spinw/2.0),(feed_length/2.,-spinw/2.),  (-feed_length/2.,-spinw/2.)]

        #Top of feed_line
        pts2=[ (-feed_length/2,spinw/2.), (-pinw/2.-gapw,spinw/2.), (-pinw/2.-gapw,gapw+spinw/2.), (-feed_length/2.,gapw+spinw/2.), (-feed_length/2,spinw/2.) ]
        pts3=[ (feed_length/2,spinw/2.), (pinw/2.+gapw,spinw/2.), (pinw/2.+gapw,gapw+spinw/2.), (feed_length/2.,gapw+spinw/2.), (feed_length/2,spinw/2.) ]
        #stub
        pts4=[ (-pinw/2.,spinw/2.), (-pinw/2.,stub_length), (-pinw/2.-gapw,stub_length), (-pinw/2.-gapw,spinw/2.), (-pinw/2.,spinw/2.) ]
        pts5=[ (pinw/2.,spinw/2.), (pinw/2.,stub_length), (pinw/2.+gapw,stub_length), (pinw/2.+gapw,spinw/2.), (pinw/2.,spinw/2.) ]

        #channels/CCD
        pts6=[(-pinw/2.,stub_length),(-pinw/2.,stub_length+gapw),(-pinw/2.-ccdwidth/2.,stub_length+gapw),(-pinw/2.-ccdwidth/2.,stub_length),(-pinw/2.,stub_length)]
        pts7=[(pinw/2.,stub_length),(pinw/2.,stub_length+gapw),(pinw/2.+ccdwidth/2.,stub_length+gapw),(pinw/2.+ccdwidth/2.,stub_length),(pinw/2.,stub_length)]
        pts8=[(-pinw/2.-ccdwidth/2.+gapw,stub_length+gapw),(-pinw/2.-ccdwidth/2.+gapw,stub_length+gapw+ccdlength-gapw),(-pinw/2.-ccdwidth/2.,stub_length+gapw+ccdlength-gapw),(-pinw/2.-ccdwidth/2.,stub_length+gapw),(-pinw/2.-ccdwidth/2.+gapw,stub_length+gapw)]
        pts9=[(pinw/2.+ccdwidth/2.-gapw,stub_length+gapw),(pinw/2.+ccdwidth/2.-gapw,stub_length+gapw+ccdlength-gapw),(pinw/2.+ccdwidth/2.,stub_length+gapw+ccdlength-gapw),(pinw/2.+ccdwidth/2.,stub_length+gapw),(pinw/2.+ccdwidth/2.-gapw,stub_length+gapw)]
        pts10=[(-pinw/2.,stub_length+ccdlength),(-pinw/2.,stub_length+gapw+ccdlength),(-pinw/2.-ccdwidth/2.,stub_length+gapw+ccdlength),(-pinw/2.-ccdwidth/2.,stub_length+ccdlength),(-pinw/2.,stub_length+ccdlength)]
        pts11=[(pinw/2.,stub_length+ccdlength),(pinw/2.,stub_length+gapw+ccdlength),(pinw/2.+ccdwidth/2.,stub_length+gapw+ccdlength),(pinw/2.+ccdwidth/2.,stub_length+ccdlength),(pinw/2.,stub_length+ccdlength)]

        shapes=[pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9,pts10,pts11]
        
        numberofchannels=(ccdwidth-2*gapw+pinw-channelwidth)/(2*channelwidth)
        numberofchannels=int(round(float(numberofchannels)))
        totalchannelwidth=(2*numberofchannels-1)*channelwidth
        padding=((ccdwidth+pinw-2*gapw)-totalchannelwidth)/2.
        innerwidthstart=-pinw/2.-ccdwidth/2.+2*channelwidth+gapw #inner width of structure measured from left
        
        self.numberofchannels=numberofchannels
        self.channelwidth=channelwidth
        
        for j in range(numberofchannels):
            pts_temp=[(innerwidthstart+channelwidth+padding,stub_length+gapw+channelwidth),
                      (innerwidthstart+channelwidth+padding,stub_length+gapw+ccdlength-2*channelwidth-gapw),
                      (innerwidthstart+padding,stub_length+gapw+ccdlength-2*channelwidth-gapw),
                      (innerwidthstart+padding,stub_length+gapw+channelwidth),
                      (innerwidthstart+channelwidth+padding,stub_length+gapw+channelwidth)]
            pts_temp=translate_pts(pts_temp,((j-1)*2*channelwidth,0))
            shapes.append(pts_temp)
        
        pts12=[(-innerwidthstart-padding+2*channelwidth,stub_length+gapw+ccdlength-2*channelwidth-gapw),
               (-innerwidthstart-padding+2*channelwidth,stub_length+gapw+ccdlength-2*channelwidth-gapw+channelwidth),
               (innerwidthstart+padding-2*channelwidth,stub_length+gapw+ccdlength-2*channelwidth-gapw+channelwidth),
               (innerwidthstart+padding-2*channelwidth,stub_length+gapw+ccdlength-2*channelwidth-gapw),
               (-innerwidthstart-padding+2*channelwidth,stub_length+gapw+ccdlength-2*channelwidth-gapw)]
               
        shapes.append(pts12)
        
        center=orient_pt((0,0),s.last_direction,s.last)
        for pts in shapes:
            pts=orient_pts(pts,angle,center)
            s.append(sdxf.PolyLine(pts))
            
        s.last=orient_pt((feed_length,0),s.last_direction,s.last)
        lstart=orient_pt((stub_length,0),lstart_dir,center)

        Structure.__init__(self,s.chip,start=lstart,direction=lstart_dir,layer=s.layer,color=s.color,defaults=s.defaults)
        self.defaults['pinw']=pinw
        self.defaults['gapw']=gapw   
#-------------------------------------------------------------------------------------------------
class CCDChannelTeeL2(Structure):
    """CCDChannelTee makes a tee structure with microchannels attached
        this is the second layer for the thin electrodes"""
    def __init__(self,structure,stub_length=None,feed_length=None,flipped=False,pinw=None,gapw=None,spinw=None,sgapw=None,ccdwidth=100,ccdlength=100,channelwidth=8,electrodewidth=3):
        """
        stub_length is from center
        flipped determines whether stub is on left or right of wrt current direction
        pinw/gapw are the usual for the stub
        spinw/sgapw are the usual for the continuing part
        """
        s=structure
        #print sgapw
        if pinw is None: pinw=s.defaults['pinw']
        if gapw is None: gapw=s.defaults['gapw']
        if spinw is None: spinw=s.defaults['pinw']
        if sgapw is None: sgapw=s.defaults['gapw']
        #print "pinw: %f, gapw: %f, spinw: %f, sgapw: %f" % (pinw,gapw,spinw,sgapw)

        #minimum feed_length is
        if (feed_length is None) or (feed_length < 2*gapw+pinw): 
            feed_length=2*gapw+pinw
        
        #minimum stub_length is 
        if (stub_length is None) or (stub_length < gapw+spinw):  
            stub_length=gapw+spinw/2
        #print "pinw: %f, gapw: %f, spinw: %f, sgapw: %f" % (pinw,gapw,spinw,sgapw)
            
        start=s.last
        start_dir=s.last_direction
        
        if flipped: 
            lstart_dir=start_dir-90
            angle=start_dir+180
        else:       
            lstart_dir=start_dir
            angle=start_dir
        
        #useful definitions
        numberofchannels=(ccdwidth-2*gapw+pinw-channelwidth)/(2*channelwidth)
        numberofchannels=int(round(float(numberofchannels)))
        totalchannelwidth=(2*numberofchannels-1)*channelwidth
        padding=((ccdwidth+pinw-2*gapw)-totalchannelwidth)/2.
        innerwidthstart=-pinw/2.-ccdwidth/2.+2*channelwidth+gapw #inner width of structure measured from left
        
        self.numberofchannels=numberofchannels
        self.channelwidth=channelwidth
        
        shapes=[]
        
        #make the fingers
        for j in range(numberofchannels):
            pts_temp=[(innerwidthstart+channelwidth+padding-electrodewidth,stub_length+gapw+channelwidth+electrodewidth),
                      (innerwidthstart+channelwidth+padding-electrodewidth,stub_length+gapw+ccdlength-2*channelwidth-gapw+electrodewidth),
                      (innerwidthstart+padding+electrodewidth,stub_length+gapw+ccdlength-2*channelwidth-gapw+electrodewidth),
                      (innerwidthstart+padding+electrodewidth,stub_length+gapw+channelwidth+electrodewidth),
                      (innerwidthstart+channelwidth+padding-electrodewidth,stub_length+gapw+channelwidth+electrodewidth)]
            pts_temp=translate_pts(pts_temp,((j-1)*2*channelwidth,0))
            shapes.append(pts_temp)
        
        pts1=[(-innerwidthstart+2*channelwidth-padding-electrodewidth,stub_length+gapw+ccdlength-2*channelwidth-gapw+electrodewidth),
                (-innerwidthstart+2*channelwidth-padding-electrodewidth,stub_length+gapw+ccdlength-2*channelwidth-gapw+channelwidth-electrodewidth),
                (innerwidthstart-2*channelwidth+padding+electrodewidth,stub_length+gapw+ccdlength-2*channelwidth-gapw+channelwidth-electrodewidth),
                (innerwidthstart-2*channelwidth+padding+electrodewidth,stub_length+gapw+ccdlength-2*channelwidth-gapw+electrodewidth),
                (-innerwidthstart+2*channelwidth-padding-electrodewidth,stub_length+gapw+ccdlength-2*channelwidth-gapw+electrodewidth)]
               
        shapes.append(pts1)
        
        center=orient_pt((0,0),s.last_direction,s.last)
        for pts in shapes:
            pts=orient_pts(pts,angle,center)
            s.append(sdxf.PolyLine(pts))
            
        s.last=orient_pt((feed_length,0),s.last_direction,s.last)
        lstart=orient_pt((stub_length,0),lstart_dir,center)

        Structure.__init__(self,s.chip,start=lstart,direction=lstart_dir,layer=s.layer,color=s.color,defaults=s.defaults)
        self.defaults['pinw']=pinw
        self.defaults['gapw']=gapw 

#-------------------------------------------------------------------------------------------------

class ChannelReservoirL1(Structure):
    """ChannelReservoir - first layer
        width=total width of reservoir
        length=total length of reservoir
        channelw=width of individual channels"""
    def __init__(self,structure,flipped=False,width=100,length=100,channelw=8):
        s=structure
        
            
        start=s.last
        start_dir=s.last_direction
        
        if flipped: 
            lstart_dir=start_dir-90
            angle=start_dir+180
        else:       
            lstart_dir=start_dir+90
            angle=start_dir
        
        #note: numberofchannels is twice the true number of channels since
        #it also contains the spacing between the channels
        numberofchannels=length/(2*channelw)
        numberofchannels=int(round(float(numberofchannels)))
        length=numberofchannels*2*channelw-channelw
        
        
        self.numberofchannels=numberofchannels
        
        leftchannel=[(-width/2.,0),(-channelw/2.,0),(-channelw/2.,channelw),(-width/2.,channelw),(-width/2.,0)]
        rightchannel=[(width/2.,0),(channelw/2.,0),(channelw/2.,channelw),(width/2.,channelw),(width/2.,0)]
        
        # add the first channels on lhs and rhs side of center
        shapes=[leftchannel,rightchannel]    
        
        # add the other channels by translation
        for j in range(1,numberofchannels):
            pts_lhs=translate_pts(leftchannel,(0,j*2*channelw))
            pts_rhs=translate_pts(rightchannel,(0,j*2*channelw))
            shapes.append(pts_lhs)
            shapes.append(pts_rhs)
        
        centerbox=[(-channelw/2,0),(channelw/2.,0),(channelw/2.,length),(-channelw/2.,length),(-channelw/2.,0)]
        shapes.append(centerbox)
        
        center=orient_pt((0,0),s.last_direction,s.last)
        for pts in shapes:
            pts=orient_pts(pts,angle,center)
            s.append(sdxf.PolyLine(pts))
            
        s.last=orient_pt((0,length),s.last_direction,s.last)
        lstart=orient_pt((0,0),lstart_dir,center)

        Structure.__init__(self,s.chip,start=lstart,direction=lstart_dir,layer=s.layer,color=s.color,defaults=s.defaults)
        
#-------------------------------------------------------------------------------------------------

class ChannelReservoirL2(Structure):
    """ChannelReservoir - second layer
        width=total width of reservoir
        length=total length of reservoir
        channelw=width of individual channels"""
    def __init__(self,structure,flipped=False,width=100,length=100,channelw=8,electrodewidth=2):
        s=structure
        
            
        start=s.last
        start_dir=s.last_direction
        
        if flipped: 
            lstart_dir=start_dir-90
            angle=start_dir+180
        else:       
            lstart_dir=start_dir+90
            angle=start_dir
        
        #note: numberofchannels is twice the true number of channels since
        #it also contains the spacing between the channels
        numberofchannels=length/(2*channelw)
        numberofchannels=int(round(float(numberofchannels)))
        length=numberofchannels*2*channelw-channelw
        
        
        self.numberofchannels=numberofchannels
        
        delta=(channelw-electrodewidth)/2.
        
        leftchannel=[(-width/2.+delta,delta),(-channelw/2.+delta,delta),(-channelw/2.+delta,delta+electrodewidth),(-width/2.+delta,delta+electrodewidth),(-width/2.+delta,delta)]
        rightchannel=[(width/2.-delta,delta),(channelw/2.-delta,delta),(channelw/2.-delta,delta+electrodewidth),(width/2.-delta,delta+electrodewidth),(width/2.-delta,delta)]
        
        # add the first channels on lhs and rhs side of center
        shapes=[leftchannel,rightchannel]    
        
        # add the other channels by translation
        for j in range(1,numberofchannels):
            pts_lhs=translate_pts(leftchannel,(0,j*2*channelw))
            pts_rhs=translate_pts(rightchannel,(0,j*2*channelw))
            shapes.append(pts_lhs)
            shapes.append(pts_rhs)
        
        centerbox=[(-electrodewidth/2,0),(electrodewidth/2.,0),(electrodewidth/2.,length),(-electrodewidth/2.,length),(-electrodewidth/2.,0)]
        shapes.append(centerbox)
        
        center=orient_pt((0,0),s.last_direction,s.last)
        for pts in shapes:
            pts=orient_pts(pts,angle,center)
            s.append(sdxf.PolyLine(pts))
            
        s.last=orient_pt((0,length),s.last_direction,s.last)
        lstart=orient_pt((0,0),lstart_dir,center)

        Structure.__init__(self,s.chip,start=lstart,direction=lstart_dir,layer=s.layer,color=s.color,defaults=s.defaults)
        
#-------------------------------------------------------------------------------------------------

class ChannelFingerCap:
    """A Channel finger capacitor"""
    def __init__(self,num_fingers,finger_length,finger_width,finger_gap,taper_length=10,channelw=2,capacitance=0.0):
        self.type='Channel finger cap'
        self.capacitance=capacitance        #simulated capacitance
        self.num_fingers=num_fingers        #number of fingers
        if num_fingers<2:
            raise MaskError, "ChannelFingerCap must have at least 2 fingers!"
        self.finger_length=finger_length    #length of fingers
        self.finger_width=finger_width      #width of each finger
        self.finger_gap=finger_gap
        self.capw = num_fingers*finger_width+ (num_fingers-1)*finger_gap    #effective center pin width sum of finger gaps and widths
        self.length=finger_length+finger_gap
        self.taper_length=taper_length
        self.total_length=finger_length+finger_gap+2.*taper_length
        self.gapw=channelw
        
    def description(self):
        return "type:\t%s\tAssumed Capacitance:\t%f\t# of fingers:\t%d\tFinger Length:\t%f\tFinger Width:\t%f\tFinger Gap:\t%f\tTotal Pin Width:\t%f\tTaper Length:\t%f" % (
                self.type,self.capacitance*1e15,self.num_fingers,self.finger_length,self.finger_width,self.finger_gap,self.capw,self.taper_length
                )    
                
    def left_finger_points(self,structure,finger_width,finger_length,finger_gap):      
        '''This is the original finger drawing code
        pts= [  (0,self.capw/2.),
                (finger_length,self.capw/2.),
                (finger_length,self.capw/2.-finger_width),
                (0,self.capw/2.-finger_width),
                (0,self.capw/2.)
            ]'''
        
        #Calculates how many fingers should go on each side of the capacitor. segmentLen is length of a finger plus the gap between fingers on one side. It's how far down we must move each iteration.
        noLeftFingers = self.num_fingers/2 + (self.num_fingers % 2)
        #fingerSpacing = finger_width + 2*finger_gap
        #segmentLen = fingerSpacing + finger_gap
        s = structure
        
        #Draws the first finger at top.    
        pts= [  (0,self.capw/2.),
                (finger_length,self.capw/2.),
                (finger_length,self.capw/2.-finger_width),
                (0,self.capw/2.-finger_width),
                (0,self.capw/2.)
            ]
        pts1=translate_pts(pts,s.last)
        pts1=rotate_pts(pts1,s.last_direction,s.last)
        #pts1=translate_pts(pts1,(self.finger_length+self.finger_gap,0))
        s.append(sdxf.Solid(pts1))
            
        #Now add the rest of the fingers
        for x in range(noLeftFingers-1):
            pts = translate_pts(pts,offset=(0,-2*finger_width-2*finger_gap))
            pts1=translate_pts(pts,s.last)
            pts1=rotate_pts(pts1,s.last_direction,s.last)
            #pts1=translate_pts(pts1,(self.finger_length+self.finger_gap,0))
            s.append(sdxf.Solid(pts1))
           
        #Bring back to beginning to make sure polygons closed
        #pts.append((0,self.capw/2.))                
                
        return pts
        
    def right_finger_points(self,structure,finger_width,finger_length,finger_gap):         
        '''pts= [  (finger_gap,-self.capw/2.),
                (finger_gap+finger_length,-self.capw/2.),
                (finger_gap+finger_length,-self.capw/2.+finger_width),
                (finger_gap,-self.capw/2.+finger_width),
                (finger_gap,-self.capw/2.)
            ] '''
            
        #Geometric Parameters. A segment is a finger plus the gap between fingers on one side.
        noRightFingers = self.num_fingers/2
        fingerSpacing = finger_width + 2*finger_gap
        segmentLen = fingerSpacing + finger_gap
        s=structure
        
        #Draw first finger at bottom        
        pts= [  (finger_gap+finger_length,-self.capw/2.),
                (finger_gap,-self.capw/2.),
                (finger_gap,-self.capw/2.+finger_width),
                (finger_gap+finger_length,-self.capw/2.+finger_width),
                (finger_gap+finger_length,-self.capw/2.)
            ]
        pts1=translate_pts(pts,s.last)
        pts1=rotate_pts(pts1,s.last_direction,s.last)
        #pts1=translate_pts(pts1,(self.finger_length+self.finger_gap,0))
        s.append(sdxf.Solid(pts1))
            
        #Now add the rest of the fingers
        for x in range(noRightFingers-1):
            pts= translate_pts(pts,offset=(0,2*finger_width+2*finger_gap))
            pts1=translate_pts(pts,s.last)
            pts1=rotate_pts(pts1,s.last_direction,s.last)
            #pts1=translate_pts(pts1,(self.finger_length+self.finger_gap,0))
            s.append(sdxf.Solid(pts1))
           
        #Bring back to beginning to make sure polygons closed
        pts.append((finger_gap+finger_length,-self.capw/2.))                
                
        return pts

    
    def draw(self,structure):
        s=structure
        capw=self.capw
        
        ChannelLinearTaper(s,length=self.taper_length,start_channelw=self.gapw,stop_channelw=self.capw)
        
        start=s.last
        
        center_width=self.num_fingers*self.finger_width+ (self.num_fingers-1)*self.finger_gap
        length=self.finger_length+self.finger_gap

        #draw finger gaps
        
        pts1=self.left_finger_points(s,self.finger_width,self.finger_length,self.finger_gap)
        '''pts1=translate_pts(pts1,start)
        pts1=rotate_pts(pts1,s.last_direction,start)
        #pts1=translate_pts(pts1,(self.finger_length+self.finger_gap,0))
        s.append(sdxf.PolyLine(pts1))'''
        
        pts2=self.right_finger_points(s,self.finger_width,self.finger_length,self.finger_gap)
        '''pts2=translate_pts(pts2,start)
        pts2=rotate_pts(pts2,s.last_direction,start)
        #pts2=translate_pts(pts2,(self.finger_length+self.finger_gap,0))
        s.append(sdxf.PolyLine(pts2))'''
        
        stop=rotate_pt((start[0]+self.finger_length+self.finger_gap,start[1]),s.last_direction,start)
        s.last=stop
        ChannelLinearTaper(s,length=self.taper_length,start_channelw=self.capw,stop_channelw=self.gapw+2.5)
        
    
    def ext_Q(self,frequency,impedance=50,resonator_type=0.5):
        if self.capacitance==0: 
            return 0
        frequency=frequency*1e9
        q=2.*pi*frequency*self.capacitance*impedance
        Q=0
        if q!=0:
            Q=1/(resonator_type*pi) *1/ (q**2)
        return Q
        
#Second version of ChannelFingerCap with symmetric ends                
        
class ChannelFingerCapSym:
    """A Channel finger capacitor"""
    def __init__(self,num_fingers,finger_length,finger_width,finger_gap,taper_length=10,channelw=2,capacitance=0.0):
        self.type='Channel finger cap'
        self.capacitance=capacitance        #simulated capacitance
        self.num_fingers=num_fingers        #number of fingers
        if num_fingers<2:
            raise MaskError, "ChannelFingerCap must have at least 2 fingers!"
        self.finger_length=finger_length    #length of fingers
        self.finger_width=finger_width      #width of each finger
        self.finger_gap=finger_gap
        self.capw = num_fingers*finger_width+ (num_fingers-1)*finger_gap    #effective center pin width sum of finger gaps and widths
        self.length=finger_length+finger_gap
        self.taper_length=taper_length
        self.total_length=finger_length+finger_gap+2.*taper_length
        self.pinw=channelw
    
    #right now draws only one finger on each side of capacitor. Change the following two methods to get larger caps.    
    
    def left_finger_points(self,structure,finger_width,finger_length,finger_gap):      
        '''This is the original finger drawing code
        pts= [  (0,self.capw/2.),
                (finger_length,self.capw/2.),
                (finger_length,self.capw/2.-finger_width),
                (0,self.capw/2.-finger_width),
                (0,self.capw/2.)
            ]'''
        
        #Calculates how many fingers should go on each side of the capacitor. segmentLen is length of a finger plus the gap between fingers on one side. It's how far down we must move each iteration.
        noLeftFingers = self.num_fingers/2 + (self.num_fingers % 2)
        #fingerSpacing = finger_width + 2*finger_gap
        #segmentLen = fingerSpacing + finger_gap
        s = structure
        
        #Draws the first finger at top.    
        pts= [  (0,self.capw/2.),
                (finger_length,self.capw/2.),
                (finger_length,self.capw/2.-finger_width),
                (0,self.capw/2.-finger_width),
                (0,self.capw/2.)
            ]
        pts1=translate_pts(pts,s.last)
        pts1=rotate_pts(pts1,s.last_direction,s.last)
        #pts1=translate_pts(pts1,(self.finger_length+self.finger_gap,0))
        s.append(sdxf.Solid(pts1))
            
        #Now add the rest of the fingers
        for x in range(noLeftFingers-1):
            pts = translate_pts(pts,offset=(0,-2*finger_width-2*finger_gap))
            pts1=translate_pts(pts,s.last)
            pts1=rotate_pts(pts1,s.last_direction,s.last)
            #pts1=translate_pts(pts1,(self.finger_length+self.finger_gap,0))
            s.append(sdxf.Solid(pts1))
           
        #Bring back to beginning to make sure polygons closed
        #pts.append((0,self.capw/2.))                
                
        return pts
        
    def right_finger_points(self,structure,finger_width,finger_length,finger_gap):         
        '''pts= [  (finger_gap,-self.capw/2.),
                (finger_gap+finger_length,-self.capw/2.),
                (finger_gap+finger_length,-self.capw/2.+finger_width),
                (finger_gap,-self.capw/2.+finger_width),
                (finger_gap,-self.capw/2.)
            ] '''
            
        #Geometric Parameters. A segment is a finger plus the gap between fingers on one side.
        noRightFingers = self.num_fingers/2
        fingerSpacing = finger_width + 2*finger_gap
        segmentLen = fingerSpacing + finger_gap
        s=structure
        
        #Draw first finger at bottom        
        pts= [  (finger_gap+finger_length,-self.capw/2.),
                (finger_gap,-self.capw/2.),
                (finger_gap,-self.capw/2.+finger_width),
                (finger_gap+finger_length,-self.capw/2.+finger_width),
                (finger_gap+finger_length,-self.capw/2.)
            ]
        pts1=translate_pts(pts,s.last)
        pts1=rotate_pts(pts1,s.last_direction,s.last)
        #pts1=translate_pts(pts1,(self.finger_length+self.finger_gap,0))
        s.append(sdxf.Solid(pts1))
            
        #Now add the rest of the fingers
        for x in range(noRightFingers-1):
            pts= translate_pts(pts,offset=(0,2*finger_width+2*finger_gap))
            pts1=translate_pts(pts,s.last)
            pts1=rotate_pts(pts1,s.last_direction,s.last)
            #pts1=translate_pts(pts1,(self.finger_length+self.finger_gap,0))
            s.append(sdxf.Solid(pts1))
           
        #Bring back to beginning to make sure polygons closed
        pts.append((finger_gap+finger_length,-self.capw/2.))                
                
        return pts
    
    def ext_Q(self,frequency,impedance=50,resonator_type=0.5):
        if self.capacitance==0: 
            return 0
        frequency=frequency*1e9
        q=2.*pi*frequency*self.capacitance*impedance
        Q=0
        if q!=0:
            Q=1/(resonator_type*pi) *1/ (q**2)
        return Q
    
    def draw(self,structure):
        s=structure
        
        ChannelLinearTaper(s,length=self.taper_length,start_channelw=self.pinw,stop_channelw=self.capw)
        
        start=s.last
        
        center_width=self.num_fingers*self.finger_width+ (self.num_fingers-1)*self.finger_gap
        length=self.finger_length+self.finger_gap

        #draw finger gaps
        
        pts1=self.left_finger_points(s,self.finger_width,self.finger_length,self.finger_gap)
        '''pts1=translate_pts(pts1,start)
        pts1=rotate_pts(pts1,s.last_direction,start)
        #pts1=translate_pts(pts1,(self.finger_length+self.finger_gap,0))
        s.append(sdxf.PolyLine(pts1))'''
        
        pts2=self.right_finger_points(s,self.finger_width,self.finger_length,self.finger_gap)
        '''pts2=translate_pts(pts2,start)
        pts2=rotate_pts(pts2,s.last_direction,start)
        #pts2=translate_pts(pts2,(self.finger_length+self.finger_gap,0))
        s.append(sdxf.PolyLine(pts2))'''
        
        stop=rotate_pt((start[0]+self.finger_length+self.finger_gap,start[1]),s.last_direction,start)
        s.last=stop
        ChannelLinearTaper(s,length=self.taper_length,start_channelw=self.capw,stop_channelw=self.pinw)
                
    def description(self):
        return "type:\t%s\tAssumed Capacitance:\t%f\t# of fingers:\t%d\tFinger Length:\t%f\tFinger Width:\t%f\tFinger Gap:\t%f\tTotal Pin Width:\t%f\tTaper Length:\t%f" % (
                self.type,self.capacitance*1e15,self.num_fingers,self.finger_length,self.finger_width,self.finger_gap,self.capw,self.taper_length
                )
                
#Draws capacitor sheath from dimensions of interior cap

class ChannelFingerCapSheath:
    """A Channel finger capacitor"""
    def __init__(self,structure,cap,channelw=10,pinw=2):
        self.type = 'Channel finger cap sheath'
        self.channelw = channelw
        self.pinw = pinw
        self.taper_length = cap.taper_length

        self.gapw = (channelw - pinw)/2.
        gap_ratio = self.gapw/pinw
        self.width = (1+2*gap_ratio)*cap.capw
        
        s=structure
        
        ChannelLinearTaper(s,length=self.taper_length,start_channelw=self.channelw,stop_channelw=self.width)
        Channel(s,length=cap.length,channelw=self.width)
        ChannelLinearTaper(s,length=self.taper_length,start_channelw=self.width,stop_channelw=self.channelw)
        
#-------------------------------------------------------------------------------------------------    

class ForkCoupler(Structure):
    
    """makes a fork-shaped structure of electrodes
        fork_width is the total width of the fork"""
    def __init__(self,structure,fork_width=None,fork_length=None,flipped=False,finger_width=None,channelw=None):
        """
        """
        s=structure
        start=s.last
        start_dir=s.last_direction
        
        if channelw is None: channelw=s.defaults['channelw']

        #minimum fork_width is
        if (fork_width is None) or (fork_width < channelw): 
            fork_width=channelw
            
        if (fork_length is None) or (fork_length < channelw): 
            fork_length=channelw
        
        if finger_width is None:
            finger_width=channelw/2.
            
            
        if flipped: 
            lstart_dir=start_dir-90
            angle=start_dir+180
        else:       
            lstart_dir=start_dir
            angle=start_dir-90
            
        
        #fork vertical
        pts1=[ (-fork_width/2.,0), (-fork_width/2.,finger_width),  (fork_width/2.,finger_width),(fork_width/2.,0),  (-fork_width/2.,0)]
        #fork finger one
        pts2=[ (-fork_width/2.,finger_width),(-fork_width/2.,finger_width+fork_length),(-fork_width/2.+finger_width,finger_width+fork_length),(-fork_width/2.+finger_width,finger_width),(-fork_width/2.,finger_width)]
        #fork finger two
        pts3=[ (fork_width/2.,finger_width),(fork_width/2.,finger_width+fork_length),(fork_width/2.-finger_width,finger_width+fork_length),(fork_width/2.-finger_width,finger_width),(fork_width/2.,finger_width)]
 
        shapes=[pts1,pts2,pts3]

        center=orient_pt((0,0),s.last_direction,s.last)
        for pts in shapes:
            pts=orient_pts(pts,angle,center)
            s.append(sdxf.PolyLine(pts))
            
        s.last=orient_pt((fork_length,0),s.last_direction,s.last)
        lstart=orient_pt((0,0),lstart_dir,center)

        Structure.__init__(self,s.chip,start=lstart,direction=lstart_dir,layer=s.layer,color=s.color,defaults=s.defaults)
        
        #s.last=orient_pt((0,0),s.last_direction,s.last)
        #lstart=orient_pt((0,0),s.last_direction,s.last)

        #Structure.__init__(self,s.chip,start=lstart,direction=0,layer=s.layer,color=s.color,defaults=s.defaults)
        #self.defaults['channelw']=channelw    


#=======================================================================
# MISC COMPONENTS/CLASSES
#=======================================================================


class CapDesc:
    """Description of a capacitor, including physical geometry and simulated capacitance
       valid types are ('gap','finger','L') 
       !deprecated!
    """
    def __init__(self,capacitance,cap_gap,gapw,num_fingers=0,finger_length=0,finger_width=0,type='gap'):
        self.capacitance=capacitance        #simulated capacitance
        self.num_fingers=num_fingers        #number of fingers (0 means gap cap)
        self.finger_length=finger_length    #length of fingers
        self.finger_width=finger_width      #width of each finger
        self.cap_gap = cap_gap              #gap between fingers or center pins
        self.finger_gap=cap_gap             #for convenience set this to finger_gap
        self.gapw = gapw                    #gap between "center pin" and gnd planes
        
        
        self.pinw = num_fingers*finger_width+ (num_fingers-1)*cap_gap    #effective center pin width sum of finger gaps and widths
        
    def draw_cap(self,structure):
        if self.num_fingers>0:
            CPWFingerCap(structure,self.num_fingers,self.finger_length,self.finger_width,self.cap_gap,self.gapw)
        else:
            CPWGapCap(structure,self.cap_gap)

class AlphaNum:
    """A polyline representation of an alphanumeric character, does not use structures"""
    def __init__(self,drawing,letter,size,point,direction=0):

        if (letter=='') or (letter==' '):
            return
        #s=structure
        scaled_size=(size[0]/16.,size[1]/16.)
        for pts in alphanum_dict[letter.lower()]:
            mpts = scale_pts(pts,scaled_size)
            mpts = orient_pts(mpts,direction,point)
            drawing.append(sdxf.PolyLine(mpts))
        #s.last=orient_pt( (size[0],0),s.last_direction,s.last)

class AlphaNumText:
    """Renders a text string in polylines, does not use structures"""
    def __init__(self,drawing,text,size,point,centered=False,direction=0):
        self.text=text
        if text is None:
            return
        if centered:
            offset=(-size[0]*text.__len__()/2.,0)
            point=orient_pt(offset,direction,point)
        for letter in text:
            AlphaNum(drawing,letter,size,point,direction)
            point=orient_pt( (size[0],0),direction,point)
            
class AlignmentCross:
    def __init__(self,drawing,linewidth,size,point):
        lw=linewidth/2.
        w=size[0]/2.
        h=size[1]/2.
        pts=[ (-lw,-h), (lw,-h), (lw,-lw),(w,-lw),(w,lw),(lw,lw),(lw,h),(-lw,h),(-lw,lw),(-w,lw),(-w,-lw),(-lw,-lw),(-lw,-h)]
        
        pts=translate_pts(pts,point)
        
        drawing.append(sdxf.PolyLine(pts))
        
class FineAlign:
    def __init__(self,chip,buffer=60,al=60,wid=2):
        '''Draws 4 L shaped alignment marks in the corners of the chip
        wid is width of the L's
        buffer is distance from center of L's to edge of chip
        length is lenght of outer side of the L's leg.
        '''
        
        s1=Structure(chip,start=(buffer,buffer),color=3,direction=0)
        LShapeAlignmentMarks(s1,width=wid,armlength=al)
        
        s2=Structure(chip,start=(buffer,chip.size[1]-buffer),color=3,direction=270)
        LShapeAlignmentMarks(s2,width=wid,armlength=al)
        
        s3=Structure(chip,start=(chip.size[0]-buffer,chip.size[1]-buffer),color=3,direction=180)
        LShapeAlignmentMarks(s3,width=wid,armlength=al)
        
        s4=Structure(chip,start=(chip.size[0]-buffer,buffer),color=3,direction=90)
        LShapeAlignmentMarks(s4,width=wid,armlength=al)

class SolidNotch:
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
        
        notch_pts= [ (-self.notch_length/2.,-self.pinw/2.-self.gapw), (-self.notch_length/2.,-self.pinw/2.-self.gapw-self.notch_depth), (self.notch_length/2.,-self.pinw/2.-self.gapw-self.notch_depth),(self.notch_length/2.,-self.pinw/2.-self.gapw), (-self.notch_length/2.,-self.pinw/2.-self.gapw)]
        notch2_pts= mirror_pts(notch_pts,0,(0,0))
        
        notch_pts=orient_pts(notch_pts,s.last_direction,s.last)
        notch2_pts=orient_pts(notch2_pts,s.last_direction,s.last)
        
        box_pts= [ (-1,-1),(1,-1),(1,1),(-1,1),(-1,-1)]

        #Superfine
        sf_pts=scale_pts(box_pts,(self.superfine_size/2.,self.superfine_size/2.))
        sf1_pts=translate_pts(sf_pts,(self.superfine_spacing,-self.superfine_spacing))
        sf2_pts=translate_pts(sf_pts,(-self.superfine_spacing,-self.superfine_spacing))
        sf3_pts=translate_pts(sf_pts,(self.superfine_spacing,self.superfine_spacing))
        sf4_pts=translate_pts(sf_pts,(-self.superfine_spacing,self.superfine_spacing))
        
        sf1_pts=orient_pts(sf1_pts,s.last_direction,s.last)
        sf2_pts=orient_pts(sf2_pts,s.last_direction,s.last)
        sf3_pts=orient_pts(sf3_pts,s.last_direction,s.last)
        sf4_pts=orient_pts(sf4_pts,s.last_direction,s.last)
        
        #Fine
        
        f_pts=scale_pts(box_pts,(self.fine_size/2.,self.fine_size/2.))
        f2_pts = translate_pts(f_pts,(0,-self.fine_offset))
        f_pts=translate_pts(f_pts,(0,self.fine_offset))

        f2_pts=orient_pts(f2_pts,s.last_direction,s.last)
        f_pts=orient_pts(f_pts,s.last_direction,s.last)
        
        #Rough
        chip_size=s.chip.size
        
        dir_pt=(cos(s.last_direction*pi/180.),sin(s.last_direction*pi/180.))
        dir_pt=rotate_pt(dir_pt,90)
        
        r_cpt=orient_pt((0,0),s.last_direction,s.last)
        r_cpt=(r_cpt[0]*abs(dir_pt[1])+(1-dir_pt[1])*(chip_size[0]-self.rough_size/2.),r_cpt[1]*abs(dir_pt[0])+(1-dir_pt[0])*(chip_size[1]-self.rough_size/2.))
        
        r_pts=scale_pts(box_pts,(self.rough_size/2.,self.rough_size/2.))
        r_pts=orient_pts(r_pts,s.last_direction,r_cpt)
        r2_pts=mirror_pts(r_pts,s.last_direction,s.last)
                 
        s.append(sdxf.Solid(notch_pts))
        s.append(sdxf.Solid(notch2_pts))
        s.append(sdxf.Solid(sf1_pts))
        s.append(sdxf.Solid(sf2_pts))
        s.append(sdxf.Solid(sf3_pts))
        s.append(sdxf.Solid(sf4_pts))
        s.append(sdxf.Solid(f_pts))
        s.append(sdxf.Solid(f2_pts))
        s.append(sdxf.Solid(r_pts))
        s.append(sdxf.Solid(r2_pts))

def arc_pts(start_angle,stop_angle,radius,segments=360):
    pts=[]
    for ii in range(segments):
        theta=(start_angle+ii/(segments-1.)*(stop_angle-start_angle))*pi/180.
        p=(radius*cos(theta),radius*sin(theta))
        pts.append(p)
    return pts
    
    


            
            
            
            
            
            
            
            
            
            
            
            
