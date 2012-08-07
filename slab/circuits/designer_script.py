# -*- coding: utf-8 -*-
"""
Created on Mon Aug 06 11:30:46 2012

@author: phil
"""
from slab.circuits import orient_pts

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
oProject.InsertDesign "EM Design", "%(designname)s", "", ""
Set oDesign = oProject.SetActiveDesign("%(designname)s")
Set oEditor = oDesign.SetActiveEditor("Layout")
"""

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

    def write(self, s):
        self.file.write(s)

    def save(self):
        self.file.close()

    def uid(self):
        self._uid += 1
        return str(self._uid)
    
    def add_unit(self, v):
        if isinstance(v, (float, int)) or v.isdigit():
            return str(v)+self.unit
        else:
            return v
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
            if isinstance(arr, str):
                self.write("\""+str(arr)+"\"")
            elif isinstance(arr, bool):
                self.write(str(arr).lower())
            else:
                self.write(str(arr))
                
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
                 "Thickness:=", self.add_unit(thickness),
                 "LowerElevation:=", self.add_unit(self.stackup_height),
                 "Roughness:=", 0,
                 "Material:=", material
                ]
              ]
        self.stackup_height += thickness
        self.write_array(arg)
        self.write("\n")
    
    def add_property(self, name, value):
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
                      "Value:=", value
                    ]
                  ]                  
                ]
              ]
        self.write_array(arg)
        self.write("\n")
    
    def draw_rectangle_pts(self, pt1, pt2, angle=0, name=None, layer=None,):
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
                  "Ax:=", self.add_unit(pt1[0]),
                  "Ay:=", self.add_unit(pt1[1]),
                  "Bx:=", self.add_unit(pt2[0]),
                  "By:=", self.add_unit(pt2[1]),
                  "ang:=", angle
                ]
              ]
        self.write_array(arg)
        self.write("\n")
              
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
                  "lw:=", str(width)+self.unit if isinstance(width, (int, float)) else width,
                  "endstyle:=", 0,
                  "joinstyle:=", 1,
                  "n:=", len(pts)
                ] + flatten([["x:=", p[0], "y:=", p[1]] for p in pts])
              ]
        self.write_array(arg)
        self.write("\n")
              
    def CPWStraight(self, start, angle, length, pinw, gapw):
        start = map(DString, start)
        angle, length, pinw, gapw =\
          map(DString, [angle, length, pinw, gapw])
        h0 = start[1] + pinw/2.
        h1 = start[1] + (pinw/2.) + gapw
        w0, w1 = start[0], start[0] + length
        gap1 = [(w0, h0), (w1, h1)]
        gap2 = [(w0, -h0), (w1, -h1)]
        self.draw_rectangle_pts(gap1[0], gap1[1], angle)
        self.draw_rectangle_pts(gap2[0], gap2[1], angle)
    
    def CPWBend(self, start, start_angle, bend_angle, radius, pinw, gapw, 
                segments=6, name=None, ):
        "Segment number is fixed. Allowed parametric: start, start_angle, pinw, gapw?"
        start, start_angle, bend_angle, radius, pinw, gapw =\
          map(DString, [start, start_angle, bend_angle, radius, pinw, gapw])
        
        #segment_angle_prop = "_" + name + "segang"
        #self.add_property(segment_angle_prop, bend_angle_prop+"/"+str(segments))
        comp_angle = 180 - bend_angle - start_angle
        circle_offset = -1*radius*("cos("+comp_angle+")"), radius*("sin("+comp_angle+")")
        def x_form(n, r):
            return r*("cos("+(n*bend_angle)+")") + start[0] + circle_offset[0]
        def y_form(n, r):
            return r*("sin("+(n*bend_angle)+")") + start[1] + circle_offset[1]
        def points(r):
            return [(x_form(n, r), y_form(n, r)) for n in map(str, range(segments))]
        
        delta = (pinw+gapw)/2
        gap1 = points(radius + delta)
        gap2 = points(radius - delta)
        self.draw_line_pts(self, gap1, gapw)
        self.draw_line_pts(self, gap2, gapw)

class DStructure(object):
    def __init__(self, start=("0","0"), direction="0"):

class DString(str):
    def __add__(self, other):
        return DString("("+str(self)+"+"+str(other)+")")
    def __radd__(self, other):
        return DString("("+str(other)+"+"+str(self)+")")
    def __sub__(self, other):
        return DString("("+str(self)+"-"+str(other)+")")
    def __rsub__(self, other):
        return DString("("+str(other)+"-"+str(self)+")")
    def __mul__(self, other):
        return DString("("+str(self)+"*"+str(other)+")")
    def __rmul__(self, other):
        return DString("("+str(other)+"*"+str(self)+")")
    def __div__(self, other):
        return DString("("+str(self)+"/"+str(other)+")")
    def __rdiv__(self, other):
        return DString("("+str(other)+"/"+str(self)+")")
    def __neg__(self):
        return DString("(-"+str(self)+")")

def cos(a):
  return DString("cos("+str(a)+")")
def sin(a):
  return DString("sin("+str(a)+")")

def flatten(list_of_lists):
    res = []
    for l in list_of_lists:
        res += l
    return res

#class DesignerStructure(object):
#    def __init__(self, start, direction):
#        self.start = start
    
if __name__ == "__main__":
    d = DesignerScript("test_dscript")
    d.add_layer("main", "ground", "perfect conductor", 0, main=True)
    d.add_property("straight_length", "200um")
    d.add_property("gapw", "10um")
    d.CPWStraight((0,0), 0, "straight_length", "10um", "gapw")
    d.save()
    print "Done!"