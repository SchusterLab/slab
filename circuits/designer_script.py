# -*- coding: utf-8 -*-
"""
Created on Mon Aug 06 11:30:46 2012

@author: phil
"""
from slab.circuits import orient_pts, calculate_gap_width, calculate_interior_length
from math import pi
from dsobj import *
import numpy as np
import logging
logging.basicConfig(filename="dstest.log")

class DesignerScript(object):
    def __init__(self, fname, unit="um"):
        if not fname.endswith(".vbs"):
            fname += ".vbs"
        self.designname = fname[:-4]
        self.fname = fname
        self.file = open(fname, 'w')
        self.unit = unit
        self.stackup_height = 0
        self.file.write(script_header % {"designname":self.designname})
        self._uid = 0 # Just a unique number for arbitrary names
        self.setup_num = 1
        self.sweep_num = 1
        self.opti_num = 1
        self.properties = []
        #DString.script = self
        DSObj.script = self

    def write(self, string):
        self.file.write(string)

    def save(self):
        self.file.close()

    def uid(self):
        self._uid += 1
        return str(self._uid)
    
    #def add_unit(self, v):
    #    if isinstance(v, DString):
    #        v = v._val
    #    if isinstance(v, (float, int)) or v.isdigit():
    #        return str(v)+self.unit
    #    else:
    #        return v
    def write_array(self, arr):
        if isinstance(arr, list):
            im = len(arr) - 1
            self.write("Array(")    
            for i, a in enumerate(arr):
                self.write_array(a)
                if i != im:
                    self.write(", ")
            self.write(")")
        else:
            if isinstance(arr, (DSObj, str)):
                self.write("\""+str(arr)+"\"")
            elif isinstance(arr, bool):
                self.write(str(arr).lower())
            else:
                self.write(str(arr))
    
    def insert_design(self, design_name):
        self.write("""oProject.InsertDesign "EM Design", "%s", "", ""
Set oDesign = oProject.SetActiveDesign("%s")
Set oEditor = oDesign.SetActiveEditor("Layout")
""" % (design_name, design_name))
    
    def add_layer(self, name, layer_type, material, thickness, main=False):
        self.write("oEditor.AddStackupLayer ")
        if "main_layer" not in dir(self) or main:
            self.main_layer = name
        arg = [
              "NAME:layer",
              "Name:=", name,
              "Type:=", layer_type, 
              "Top Bottom:=" , "neither", 
              "Color:=", 65280,
              "Pattern:=", 5,
              "Visible:=", True,
              "Selectable:=", True, 
              "Locked:=", False,
              "ElevationEditMode:=", "none",
                [
                 "NAME:Sublayer",
                 "Thickness:=", str(thickness) + self.unit,
                 "LowerElevation:=", str(self.stackup_height) + self.unit,
                 "Roughness:=", 0,
                 "Material:=", material
                ]
              ]
        self.stackup_height += thickness
        self.write_array(arg)
        self.write("\n")
    
    def add_property(self, name, value, unit="", optimize=False):
        self.properties.append(name)
        #if add_unit:
        #    value = str(value)+self.unit
        self.write("oDesign.ChangeProperty ")
        arg = [
                "NAME:AllTabs",
                [
                  "NAME:LocalVariableTab",
                  [
                    "NAME:PropServers",
                    "Instance:0;" + self.designname
                  ],
                  [
                    "NAME:NewProps",
                    [
                      "NAME:"+name,
                      "PropType:=", "VariableProp",
                      "UserDef:=", True,
                      "Value:=", value,
                      ["NAME:Optimization", "Included:=", optimize]
                    ] 
                  ]                  
                ]
              ]
        self.write_array(arg)
        self.write("\n")
        
    def add_properties(self, prop_dict):
        for key, value in prop_dict.items():
            self.add_property(key, value)
    
    def set_module(self, module):
        self.write('Set oModule = oDesign.GetModule("%s")\n' % module)
    
    def add_planar_setup(self, freq):
        self.set_module("SolveSetups")
        name = "PlanarEM Setup " + str(self.setup_num)
        self.write(SetupCommand %
          {"name":name, "freq":freq}
        )
        self.setup_num += 1
        self.last_setup = name
        return name
    
    def add_sweep(self, data, fastsweep, setup=None):
        setup = setup if setup else self.last_setup
        name = "Sweep " + str(self.sweep_num)
        self.sweep_num += 1
        self.write(SweepCommand %
          {"setup":setup, "name":name, "data":data, "fastsweep":fastsweep})
        self.last_sweep = name
        return name
    
    def run_sweep(self, setup=None, sweep=None):
        assert (setup and sweep) or not (setup or sweep)
        if not setup:
            setup = self.last_setup
        if not sweep:
            sweep = self.last_sweep
        self.write('oDesign.Analyze "%(setup)s : %(sweep)s' % (locals()))
    
    def add_lincount(self, start, stop, count, setup=None):
        self.add_sweep(" ".join(["LINC"]+map(str,[start, stop, count])), "true", setup)
    
    def add_point_calc(self, point, setup=None):
        self.add_sweep(str(point), "false", setup)
        
    def add_optimization(self, targets, optimizer="Quasi Newton", setup=None, sweep=None):
        setup = setup if setup else self.last_setup
        sweep = sweep if sweep else self.last_sweep
        self.set_module("Optimetrics")
        name = "Optimization"+str(self.opti_num)
        self.opti_num += 1
        self.write(OptimizationCommandHead % {"name":name, "optimizer":optimizer})
        for ii, (formula, xtype, xval, yval) in enumerate(targets):
            self.write(OptimizationGoal % 
              {"setup":setup, "sweep":sweep, 
               "formula":formula, "xtype":xtype, 
               "xval":str(xval), "yval":str(yval)})
            if ii is not (len(targets)-1):
                self.write(", ")
        self.write(OptimizationCommandTail)
        self.last_opt = name
        return name
    
    def run_optimization(self, name=None):
        name = name if name else self.last_opt
        self.set_module("Optimetrics")
        self.write('oModule.SolveSetup "%s"\n' % name)
        
    def add_report(self, yform, xform="F", name=None, setup=None, sweep=None):
        name = name if name else "XY Plot " + self.uid()
        setup = setup if setup else self.last_setup
        sweep = sweep if sweep else self.last_sweep
        yform = yform if isinstance(yform, list) else [yform]
        self.set_module("ReportSetup")
        self.write("oModule.CreateReport ")
        arg = \
        [
          name, "Standard", "Rectangular Plot",
          setup + " : " + sweep,
          [
            "NAME:Context", "SimValueContext:=",
            [
              3, 0, 2, 0, False, False, -1, 1, 0, 1, 1, "", 0, 0, 
              "EnsDiffPairKey", False, "0", "IDIID", False, "1"
            ]
          ],
          [
            "%s:=" % xform, ["All"],
          ],
          [
            "X Component:=", xform,
            "Y Component:=", yform,
          ], []
        ]
        self.write_array(arg)
        self.write("\n")
        self.last_report = name
        return name
        
    def export_report(self, fname, report=None):
        report = report if report else self.last_report
        self.set_module("ReportSetup")
        self.write('oModule.ExportToFile "' + report + '", "' + fname + '"\n"')
        
    def import_dxf(self, fname):
        stupid_fname = fname.replace('\\','/')
        self.write(ImportCommand % {"filename":stupid_fname, "dest_layer":self.main_layer})
    
    def zoom_to_fit(self):
        self.write("oEditor.ZoomToFit\n")
    
    def draw_rectangle_pts(self, pt1, pt2, angle=0, name=None, layer=None):
        self.write("oEditor.CreateRectangle ")
        if not name:
            name = "rect"+self.uid()
        if not layer:
            layer = self.main_layer
            
        arg = [
                "NAME:Contents",
                "rectGeometry:=",
                [
                  "Name:=", name,
                  "LayerName:=", layer,
                  "lw:=", 0,
                  "Ax:=", pt1[0],#self.add_unit(pt1[0]),
                  "Ay:=", pt1[1],#self.add_unit(pt1[1]),
                  "Bx:=", pt2[0],#self.add_unit(pt2[0]),
                  "By:=", pt2[1],#self.add_unit(pt2[1]),
                  "ang:=", angle
                ]
              ]
        self.write_array(arg)
        self.write("\n")
        return name
    
    def draw_polygon(self, pts, name=None, layer=None):
        self.write("oEditor.CreatePolygon ")
        if not name:
            name = "poly"+self.uid()
        if not layer:
            layer = self.main_layer
            
        arg = [
                "NAME:Contents",
                "polyGeometry:=",
                [
                  "Name:=", name,
                  "LayerName:=", layer,
                  "lw:=", 0,
                  "n:=", len(pts)
                ] + \
                flatten([["x%d:=" % i, p[0], "y%d:=" % i, p[1]] for i, p in enumerate(pts)])
              ]
        self.write_array(arg)
        self.write("\n")
        return name

          
    def draw_line_pts(self, pts, width, name=None, layer=None):
        self.write("oEditor.CreateLine ")
        if not name:
            name = "line"+self.uid()
        if not layer:
            layer = self.main_layer
        arg = [
                "NAME:Contents",
                "lineGeometry:=",
                [
                  "Name:=", name,
                  "LayerName:=", layer,
                  "lw:=", width,
                  "endstyle:=", 1, #It turns out, endstyle actually sets bendtype, joinstyle sets endstyle...
                  "joinstyle:=", 0,
                  "n:=", len(pts)
                ] + flatten([["x%d:=" % i, p[0], "y%d:=" % i, p[1]] for i, p in enumerate(pts)])
              ]
        self.write_array(arg)
        self.write("\n")
        return name
    
    def draw_arc(self, start, delta, radius, width, orientation, name=None, layer=None):
        #arc_id = self.uid()
        R = radius
        dx, dy = delta
        #dx.cache_result("arc_dx_"+arc_id)
        #dy.cache_result("arc_dy_"+arc_id)        
        osign = {'CW':1, 'CCW':-1}[orientation]
        sagitta = R - sqrt(R*R - ((dx*dx + dy*dy)/4))
        #sagitta.cache_result("arc_sagitta_"+arc_id)
        end = vadd(start, delta)
        return self.draw_line_pts([start, (osign*sagitta, "1E+200"), end], width, name, layer)
        
    def draw_arc_angle(self, start, start_angle, bend_angle, radius, width, orientation, name=None, layer=None):
        osign = {'CW':-1, 'CCW':1}[orientation]
        delta = radius*sin(bend_angle), osign*radius*(1-cos(bend_angle))
        delta = rotate_pt(delta, start_angle)
        return self.draw_arc(start, delta, radius, width, orientation, name, layer)
        
    def CPWStraight(self, structure, length, pinw=None, gapw=None):
        pinw = pinw if pinw else structure.pinw
        gapw = gapw if gapw else structure.gapw

        start, angle = structure.start, structure.angle
        length, pinw, gapw = map(DSObjLen, [length, pinw, gapw])
        delta = (pinw+gapw)/2
        names = []
        for sign in [-1, +1]:
            name = self.draw_line_pts(structure.orient_pts([("0um", sign*delta),(length, sign*delta)]), gapw)
            names.append(name)
        end_pt_x, end_pt_y = vadd(start, structure.rotate_pt((length, "0um")))
        line_id = self.uid()
        end_pt_x.cache_result("end_pt_x"+line_id)
        end_pt_y.cache_result("end_pt_y"+line_id)
        structure.start = end_pt_x, end_pt_y
        return names
    
    def CPWGroundCap(self, s, n_fingers, finger_len, finger_width=None, pinw=None, gapw=None):
        pinw = pinw if pinw else s.pinw
        gapw = gapw if gapw else s.gapw
        finger_width = finger_width if finger_width else gapw
        finger_len, finger_width, pinw, gapw = \
          map(DSObjLen, [finger_len, finger_width, pinw, gapw])
        for i in range(n_fingers):
            self.CPWStraight(s, gapw, gapw=gapw+finger_len)
            self.CPWStraight(s, finger_width, pinw=pinw+(2*finger_len))
            self.CPWStraight(s, gapw, gapw=gapw+finger_len)
            if i is not (n_fingers - 1):
                self.CPWStraight(s, finger_width)
       
    def CPWBend(self, structure, bend_angle, radius, orientation, pinw=None, gapw=None, name=None):
        "Orientation should be either 'CW' or 'CCW', bend_angle should be positive!"
        assert(orientation in ['CW', 'CCW'])
        pinw = pinw if pinw else structure.pinw
        gapw = gapw if gapw else structure.gapw
        start, start_angle = structure.start, structure.angle
        bend_angle = DSObj(bend_angle, "deg")
        radius, pinw, gapw = map(DSObjLen, [radius, pinw, gapw])
        
        delta = (pinw+gapw)/2
        osign = {'CW':-1, 'CCW':1}[orientation]
        names = []
        for sign in [-1, 1]:
            offset = structure.rotate_pt((0, sign*delta))            
            start_gap = vadd(start, offset)
            gap_radius = radius - (osign*sign)*delta
            #gap_radius.cache_result("arc_radius_"+self.uid())
            name = self.draw_arc_angle(start_gap, start_angle, bend_angle, gap_radius, gapw, orientation)
            names.append(name)
        
        start_delta = radius*sin(bend_angle), osign*radius*(1-cos(bend_angle))
        end_pt_x, end_pt_y = vadd(start, structure.rotate_pt(start_delta))
        bend_id = self.uid()
        end_pt_x.cache_result("endpt_x_"+bend_id)
        end_pt_y.cache_result("endpt_y_"+bend_id)
        structure.start = end_pt_x, end_pt_y
        structure.angle += osign * bend_angle
        return names
        
    def CPWTaper(self, structure, length, start_pinw, start_gapw, end_pinw, end_gapw):
        start, start_angle = structure.start, structure.angle
        length, start_pinw, start_gapw, end_pinw, end_gapw =\
          map(DSObjLen, [length, start_pinw, start_gapw, end_pinw, end_gapw])
        names = []
        for sign in [-1, 1]:
            h0 = sign * (start_pinw/2)
            h1 = sign * (end_pinw/2)
            h2 = sign * (end_gapw + (end_pinw/2))
            h3 = sign * (start_gapw + (start_pinw/2))
            pts = [(0, h0), (length, h1), (length, h2), (0, h3)]
            pts = structure.orient_pts(pts)
            names.append(self.draw_polygon(pts))
        structure.start = vadd(start, structure.rotate_pt((length, 0)))
        return names
    
    def CPWWiggles(self, structure, total_length, num_wiggles, radius, pinw=None, gapw=None):
        pinw = pinw if pinw else structure.pinw
        gapw = gapw if gapw else structure.gapw
        total_length, radius, pinw, gapw = \
          map(DSObjLen, [total_length, radius, pinw, gapw])
        s = structure
        vlength=(total_length-((1+num_wiggles)*(pi*radius)+2*(num_wiggles-1)*radius))/(2*num_wiggles)
        
        self.CPWBend(s,90,radius,"CCW")
        for ii in range(num_wiggles):
            orientation = "CW" if ii % 2 == 0 else "CCW"
            self.CPWStraight(s, vlength, pinw, gapw)
            self.CPWBend(s,180,radius, orientation, pinw, gapw)
            self.CPWStraight(s, vlength, pinw, gapw)
            if ii<num_wiggles-1:
                self.CPWStraight(s, 2*radius, pinw, gapw)
        final_bend_orientation = "CW" if num_wiggles % 2 == 0 else "CCW"        
        self.CPWBend(s, 90, radius, final_bend_orientation)
        
    def CPWFingerCap(self, structure, num_fingers, finger_length, finger_width, finger_gap, taper_length="50um"):
        pinw, gapw = map(DSObjLen, [structure.pinw, structure.gapw])
        finger_length, finger_width, finger_gap, taper_length =\
          map(DSObjLen, [finger_length, finger_width, finger_gap, taper_length])
        center_width = num_fingers*finger_width + (num_fingers-1)*finger_gap
        center_gap = center_width * (gapw / pinw)
        length = finger_length + finger_gap
        
        self.CPWTaper(structure, taper_length, pinw, gapw, center_width, center_gap)

        left_finger_points =\
          [(0,0),
           (0,finger_width+finger_gap),
           (finger_length+finger_gap,finger_width+finger_gap),
           (finger_length+finger_gap,finger_width),
           (finger_gap,finger_width),
           (finger_gap,0)]
        right_finger_points =\
          [(finger_length+finger_gap,0),
           (finger_length+finger_gap,finger_width+finger_gap),
           (0,finger_width+finger_gap),
           (0,finger_width),
           (finger_length,finger_width),
           (finger_length,0)]
        for ii in range(num_fingers-1):
            if ii%2==0:
                pts=left_finger_points
            else:
                pts=right_finger_points
            
            pts = translate_pts(pts, (0,ii*(finger_width+finger_gap)-center_width/2.))
            pts = structure.orient_pts(pts)
            self.draw_polygon(pts)

        #draw last little box to separate sides
        pts = [ (0,0),(0,finger_width),(finger_gap,finger_width),(finger_gap,0)]
        pts = translate_pts(pts,(((num_fingers+1) %2)*(length-finger_gap),(num_fingers-1)*(finger_width+finger_gap)-center_width/2.))
        pts = structure.orient_pts(pts)
        self.draw_polygon(pts)
        
        self.CPWStraight(structure, length, center_width, center_gap)
        self.CPWTaper(structure, taper_length, center_width, center_gap, pinw, gapw)
    
    def create_port(self, name1, edge1, name2, edge2):
        self.write("oEditor.CreateEdgePort ")
        arg = ["NAME:Contents", 
               "edge:=", [name1, edge1],
               "edge:=", [name2, edge2],
               "external:=", True]
        self.write_array(arg)
        self.write("\n")
    
    def CPWLauncher(self, structure, pad_length, taper_length, 
                    start_pin, start_gap, make_port=False, flipped=False):
        if flipped:
            self.CPWTaper(structure, taper_length, structure.pinw, 
                          structure.gapw, start_pin, start_gap)
            left_box, right_box = self.CPWStraight(structure, pad_length, 
                                                   pinw=start_pin, gapw=start_gap)
            edge_no = 1
        else:
            left_box, right_box = self.CPWStraight(structure, pad_length, 
                                                   pinw=start_pin, gapw=start_gap)
            self.CPWTaper(structure, taper_length, start_pin, start_gap,
                          structure.pinw, structure.gapw)
            edge_no = 0
        if make_port:
            self.create_port(left_box, edge_no, right_box, edge_no)
    
    def CPWGapCap(self, s, gap, pinw, gapw):
        self.draw_line_pts(s.orient_pts([(0,0), (gap, 0)]), pinw+(2*gapw))
        s.start = vadd(s.start, s.rotate_pt((gap, 0)))

    def DoubleCPW(self, s, length, inner_pin, inner_gap, outer_pin, outer_gap):
        start = s.start
        self.CPWStraight(s, length, pinw=inner_pin, gapw=inner_gap)
        s.start = start
        self.CPWStraight(s, length, pinw=inner_pin+(2*(inner_gap+outer_pin)), gapw=outer_gap)
        
    def CPWInnerOuterFingerIsland(self, s, c_gap, n_fingers, inner_finger_length,
                                  outer_finger_length, flipped=False):
        pinw = s.pinw
        gapw = s.gapw
        # Initial Part
        if flipped:
            #CPWGapCap(c_gap, pinw, inner_finger_length+outer_finger_length+(5*c_gap)).draw(s)
            self.CPWStraight(s, c_gap, c_gap, ((pinw+c_gap)/2.)+inner_finger_length+outer_finger_length+(4*c_gap))
            self.CPWStraight(s, c_gap, pinw+(2*(inner_finger_length+outer_finger_length+(4*c_gap))), c_gap)
            start = s.start
            self.DoubleCPW(s, c_gap, pinw, inner_finger_length+(2*c_gap), c_gap, 
                      outer_finger_length+(2*c_gap))
            s.start = start
            self.CPWGapCap(s, c_gap, pinw, 0)
        else:
            self.CPWStraight(s, c_gap, pinw, inner_finger_length+outer_finger_length+(5*c_gap))
            self.DoubleCPW(s, c_gap, pinw, c_gap, 
                      inner_finger_length+outer_finger_length+(3*c_gap), c_gap)
            self.DoubleCPW(s, c_gap, pinw, inner_finger_length+(2*c_gap), c_gap, 
                      outer_finger_length+(2*c_gap))
        # Middle Fingers
        for i in range(n_fingers-2):
            # gap bit
            self.DoubleCPW(s, c_gap, pinw+(2*(inner_finger_length+c_gap)), c_gap, c_gap, c_gap)
            # first bit
            self.DoubleCPW(s, c_gap, pinw, inner_finger_length+(2*c_gap), c_gap, 
                      outer_finger_length+(2*c_gap))
            # middle bit
            self.DoubleCPW(s, c_gap, pinw, c_gap, 
                      inner_finger_length+outer_finger_length+(3*c_gap), c_gap)
            # last bit == first bit
            self.DoubleCPW(s, c_gap, pinw, inner_finger_length+(2*c_gap), c_gap, 
                      outer_finger_length+(2*c_gap))

        # Last gap bit
        self.DoubleCPW(s, c_gap, pinw+(2*(inner_finger_length+c_gap)), c_gap, c_gap, c_gap)
        # Final part
        if flipped:
            self.DoubleCPW(s, c_gap, pinw, inner_finger_length+(2*c_gap), c_gap, 
                      outer_finger_length+(2*c_gap))            
            self.DoubleCPW(s, c_gap, pinw, c_gap, 
                      inner_finger_length+outer_finger_length+(3*c_gap), c_gap)
            self.CPWStraight(s, c_gap, pinw, inner_finger_length+outer_finger_length+(5*c_gap))
        else:            
            start = s.start
            self.DoubleCPW(s, c_gap, pinw, inner_finger_length+(2*c_gap), c_gap, 
                      outer_finger_length+(2*c_gap))
            s.start = start
            self.CPWGapCap(s, c_gap, pinw, 0)
            self.CPWStraight(s, c_gap, pinw+(2*(inner_finger_length+outer_finger_length+(4*c_gap))), c_gap)
            self.CPWStraight(s, c_gap, c_gap, ((pinw+c_gap)/2.)+inner_finger_length+outer_finger_length+(4*c_gap))
            
            
            
    def CPWQubitBox(self, s, c_gap, finger_no_left, finger_no_right,
                 outer_finger_len_left, outer_finger_len_right,
                 inner_finger_len_left, inner_finger_len_right,
                 taper_len=0, int_len=30, pinw=None, gapw=None, align=False, flipped=False):
        c_gap = DSObjLen(c_gap)
        outer_finger_len_left = DSObjLen(outer_finger_len_left)
        outer_finger_len_right = DSObjLen(outer_finger_len_right)
        inner_finger_len_left = DSObjLen(inner_finger_len_left)
        inner_finger_len_right = DSObjLen(inner_finger_len_right)
        finger_gapw = c_gap
        fingerw = 3 * c_gap        

        gapw = gapw if gapw else s.gapw
        pinw = pinw if pinw else s.pinw
        
        taper_len, int_len, pinw, gapw = map(DSObjLen, [taper_len, int_len, pinw, gapw])
        finger_gapw = c_gap
        center_gapw = fingerw = 3 * c_gap
        center_pinw_left = 2 * inner_finger_len_left + pinw
        center_pinw_right = 2 * inner_finger_len_right + pinw
        center_width = dmax(center_pinw_left, center_pinw_right) + (2*center_gapw)
        outer_finger_len_left, outer_finger_len_right = outer_finger_len_left, outer_finger_len_right
        inner_finger_len_left, inner_finger_len_right = inner_finger_len_left, inner_finger_len_right
        if flipped:
            center_pinw_left, center_pinw_right = center_pinw_right, center_pinw_left
            finger_no_left, finger_no_right = finger_no_right, finger_no_left
            outer_finger_len_left, outer_finger_len_right = outer_finger_len_right, outer_finger_len_left
            inner_finger_len_left, inner_finger_len_right = inner_finger_len_right, inner_finger_len_left
        
        self.CPWTaper(s, taper_len, pinw, center_pinw_left, gapw, center_gapw)

        self.CPWInnerOuterFingerIsland(s, c_gap, finger_no_left,
                                  inner_finger_len_left, outer_finger_len_left)
        self.CPWStraight(s, int_len/2., c_gap, (center_width-c_gap)/2.)
        self.CPWGapCap(s, c_gap, center_width, 0)
        self.CPWStraight(s, int_len/2., c_gap, (center_width-c_gap)/2.)
        self.CPWInnerOuterFingerIsland(s, c_gap, finger_no_right,
                                  inner_finger_len_right, outer_finger_len_right, flipped=True)    

        self.CPWTaper(s, taper_len, center_pinw_right, pinw, center_gapw, gapw)

def rotate_pt(pt, angle):
    x = pt[0]*cos(angle) - pt[1]*sin(angle)
    y = pt[0]*sin(angle) + pt[1]*cos(angle)
    return (x, y)
    
class DStructure(object):
    def __init__(self, x="0um", y="0um", angle="0deg", pinw="10um", gapw="10um", unit="um"):
        self.start = DSObj(x), DSObj(y)
        self.angle = DSObj(angle)
        self.pinw = DSObj(pinw)
        self.gapw = DSObj(gapw)
    def rotate_pt(self, pt):
        return rotate_pt(pt, self.angle)
        x = pt[0]*cos(self.angle) - pt[1]*sin(self.angle)
        y = pt[0]*sin(self.angle) + pt[1]*cos(self.angle)
        return (x, y)
    def translate_pt(self, pt):
        return vadd(pt, self.start)
    def translate_pts(self, pts):
        return translate_pts(pts, self.start)
    def orient_pt(self, pt):
        return self.translate_pt(self.rotate_pt(pt))
    def orient_pts(self, pts):
        return [self.orient_pt(p) for p in pts]

def vadd(a, b):
    return a[0]+b[0],a[1]+b[1]
def translate_pts(pts, offset):
    return [vadd(p, offset) for p in pts]
def vsub(a, b):
    return a[0]-b[0],a[1]-b[1]

def flatten(list_of_lists):
    res = []
    for l in list_of_lists:
        res += l
    return res

script_header = \
"""
Dim oAnsoftApp
Dim oDesktop
Dim oProject
Dim oDesign
Dim oEditor
Dim oModule
Set oAnsoftApp = CreateObject("AnsoftDesigner.DesignerScript")
Set oDesktop = oAnsoftApp.GetAppDesktop()
oDesktop.RestoreWindow
Set oProject = oDesktop.NewProject
"""
""" -- Removed for now, use insert_design
oProject.InsertDesign "EM Design", "%(designname)s", "", ""
Set oDesign = oProject.SetActiveDesign("%(designname)s")
Set oEditor = oDesign.SetActiveEditor("Layout")
"""

OptimizationCommandHead = \
"""
oModule.InsertSetup "OptiOptimization", Array("NAME:%(name)s", Array("NAME:StartingPoint"), "Optimizer:=",  _
  "%(optimizer)s", Array("NAME:AnalysisStopOptions", "StopForNumIteration:=", true, "StopForElapsTime:=",  _
  false, "StopForSlowImprovement:=", false, "StopForGrdTolerance:=", false, "MaxNumIteration:=",  _
  1000, "MaxSolTimeInSec:=", 3600, "RelGradientTolerance:=", 0), "CostFuncNormType:=",  _
  "L2", "PriorPSetup:=", "", "PreSolvePSetup:=", true, Array("NAME:Variables"), Array("NAME:LCS"), Array("NAME:Goals", """

OptimizationGoal = \
"""Array("NAME:Goal", "ReportType:=",  _
  "Standard", "Solution:=", "%(setup)s : %(sweep)s", Array("NAME:SimValueContext", "SimValueContext:=", Array( _
  3, 0, 2, 0, false, false, -1, 1, 0, 1, 1, "", 0, 0, "EnsDiffPairKey", false, "0",  _
  "IDIID", false, "1")), "Calculation:=", "%(formula)s", "Name:=",  _
  "%(formula)s", Array("NAME:Ranges", "Range:=", Array("Var:=", "%(xtype)s", "Type:=",  _
  "d", "DiscreteValues:=", "%(xval)s")), "Condition:=", "==", Array("NAME:GoalValue", "GoalValueType:=",  _
  "Independent", "Format:=", "Real/Imag", "bG:=", Array("v:=", "[%(yval)s;]")), "Weight:=",  _
  "[1;]") """
  
OptimizationCommandTail = \
"""), "Acceptable_Cost:=", 0, "Noise:=", 0.0001, "defaults["pinw"]        UpdateDesign:=", false, "UpdateIteration:=",  _
  5, "KeepReportAxis:=", true, "UpdateDesignWhenDone:=", true)
"""

SetupCommand = \
"""
oModule.Add Array("NAME:%(name)s", Array("NAME:Properties", "Enable:=", "true"), "PercentRefinementPerPass:=",  _
  25, "AdaptiveFrequency:=", "%(freq)s", "NumberOfRequestedPasses:=", 10, "TargetMaximumDeltaNorm:=",  _
  0.05, "MinNumberOfPasses:=", 1, "MinNumberOfConvergedPasses:=", 1, "UseDefaultLambda:=",  _
  true, "UseMaxRefinement:=", true, "MaxRefinement:=", 100000, "SaveAdaptiveCurrents:=",  _
  false,"Refine:=", false, "Frequency:=", "%(freq)s", "LambdaRefine:=", true, "MeshSizeFactor:=",  _
  12, "QualityRefine:=", true, "MinAngle:=", "15deg", "UniformityRefine:=",  _
  false, "MaxRatio:=", 2, "Smooth:=", false, "SmoothingPasses:=", 5, "UseEdgeMesh:=",  _
  false, "UseEdgeMeshAbsLength:=", false, "EdgeMeshRatio:=", 0.1, "EdgeMeshAbsLength:=",  _
  "1000mm", "LayerProjectThickness:=", "0meter", "UseDefeature:=", true, "UseDefeatureAbsLength:=",  _
  false, "DefeatureRatio:=", 1E-006, "DefeatureAbsLength:=", "0mm", "InfArrayDimX:=",  _
  0, "InfArrayDimY:=", 0, "InfArrayOrigX:=", 0, "InfArrayOrigY:=", 0, "InfArraySkew:=",  _
  0, "ViaNumSides:=", 1, "ViaMaterial:=", "", "Style25DVia:=", "Wirebond", "Replace3DTriangles:=",  _
  true, "ViaDensity:=", 0, "HfssMesh:=", false, "UnitFactor:=", 1000, "Verbose:=",  _
  false, Array("NAME:AuxBlock"), "DoAdaptive:=", false, "Color:=", Array("R:=", 0, "G:=",  _
  0, "B:=", 0), Array("NAME:AdvancedSettings", "AccuracyLevel:=", 2, "GapPortCalibration:=",  _
  true, "ReferenceLengthRatio:=", 0.25, "RefineAreaRatio:=", 4, "DRCOn:=", false, "FastSolverOn:=",  _
  false, "StartFastSolverAt:=", 4000, "StartIterativeSolverAt:=", 3000, "LoopTreeOn:=",  _
  true, "SingularElementsOn:=", false, "UseStaticPortSolver:=", false, "UseThinMetalPortSolver:=",  _
  false, "ComputeBothEvenAndOddCPWModes:=", false, "ZeroMetalLayerThickness:=",  _
  4E-005, "ThinDielectric:=", 0, "SVDHighCompression:=", false, "NumProcessors:=",  _
  1, "UseHfssIterativeSolver:=", false, "RelativeResidual:=", 0.0001, "OrderBasis:=",  _
  -1, "MaxDeltaZo:=", 2, "UseRadBoundaryOnPorts:=", false, "SetTrianglesForWavePort:=",  _
  false, "MinTrianglesForWavePort:=", 100, "MaxTrianglesForWavePort:=", 500, "numprocessorsdistrib:=",  _
  1, "usehpcformp:=", false, "hpclicensetype:=", 1, "DesignType:=", "Generic"), Array("NAME:CurveApproximation", "ArcAngle:=",  _
  "30deg", "StartAzimuth:=", "0deg", "UseError:=", false, "Error:=", "0meter", "MaxPoints:=",  _
  8, "UnionPolys:=", true, "Replace3DTriangles:=", true))
"""

SweepCommand = \
"""
oModule.AddSweep "%(setup)s", Array("NAME:%(name)s", Array("NAME:Properties", "Enable:=",  _
  "true"), "GenerateSurfaceCurrent:=", false, "FastSweep:=", %(fastsweep)s, "ZoSelected:=",  _
  false, "SAbsError:=", 0.005, "ZoPercentError:=", 1, Array("NAME:Sweeps", "Variable:=",  _
  "%(name)s", "Data:=", "%(data)s", "OffsetF1:=", false, "Synchronize:=",  _
  0))
"""

ImportCommand = \
"""
oEditor.ImportDXF Array("NAME:options", "FileName:=",  _
  "%(filename)s", "Scale:=", 1E-006, "AutoDetectClosed:=", true, "SelfStitch:=",  _
  true, "DefeatureGeometry:=", false, "DefeatureDistance:=", 0, "RoundCoordinates:=",  _
  false, "RoundNumDigits:=", 4, "WritePolyWithWidthAsFilledPoly:=", false, "ImportMethod:=",  _
  1, "2DSheetBodies:=", false, Array("NAME:LayerInfo", Array("NAME:0", "source:=", "0", "display_source:=",  _
  "0", "import:=", false, "dest:=", "%(dest_layer)s", "dest_selected:=", true, "layer_type:=",  _
  "metalizedsignal"), Array("NAME:PYDXF", "source:=", "PYDXF", "display_source:=",  _
  "PYDXF", "import:=", false, "dest:=", "PYDXF", "dest_selected:=", false, "layer_type:=",  _
  "signal")))
"""

# TODO: Check that properties have been added when they are used    
if __name__ == "__main__":
    ilen = calculate_interior_length(5, 3e8/np.sqrt(5.5), 50)   
    
    d = DesignerScript("test_dscript")
    directory = director
    for n_fingers in [1, 5, 10]:
        for n_meanders in [1, 5, 10]:
            for length in [100, 250, 400]:
                design_name = "_".join(map(str,[n_fingers, n_meanders, length]))
                d.insert_design(design_name)
                d.import_dxf()
    d.insert_design()
    
    d.add_layer("substrate", "dielectric", "sapphire", 430)
    d.add_layer("main", "ground", "perfect conductor", 0, main=True)
    
    d.add_property("wiggles_length", str(ilen)+"um", optimize=True)
    d.add_property("inner_finger_length", "20 um", optimize=True)
    d.add_property("delta", "0um")
    d.add_property("left_finger_len", "83um")
    d.add_property("right_finger_len", "87um")
    
    
    gapw = calculate_gap_width(5.5, 50, 10)
    s = DStructure(angle="start_angle:=0deg", pinw="pinw:=10um", gapw="gapw:=%.3fum" % gapw)
    #d.CPWLauncher(s, "50um", "100um", "50um", "25um", True)    
    #d.CPWStraight(s, "150um")
    #d.CPWFingerCap(s, 4, "finger_length:=20um", "finger_width:=5um", "finger_gap:=5um")
    #d.CPWWiggles(s, "wiggles_length", 4, "bend_radius:=25um")
    
    d.CPWQubitBox("c_gap:=5um", 6, 6, "left_finger_len + delta", "right_finger_len + delta", "inner_left_len:=20um", "inner_right_len:=3um")
    #d.CPWStraight(s, "150um")
    #d.CPWLauncher(s, "50um", "100um", "50um", "25um", True, True)

    
    
    #d.CPWWiggles(s, "wiggles_length", 4, "bend_radius")
    #d.CPWFingerCap(s, 4, "finger_length", "finger_width", "finger_gap")
    #d.CPWStraight(s, "150um")
    
    #d.add_planar_setup("5GHz")
    #d.add_point_calc("5GHz 5.1GHz")
    #d.add_optimization([("im(Y(Port1,Port1))", "F", "5GHz", 0),
    #                    ("im(Y(Port1,Port1))", "F", "5.1GHz", 0)])
                        
    #d.run_optimization()
        
    d.save()
    print "Done!"