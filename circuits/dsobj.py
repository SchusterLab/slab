# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:32:56 2012

@author: slab
"""

import re
property_re = re.compile('[a-zA-Z_]+')
unit_re = re.compile('(-?[0-9]+(?:\\.[0-9]*(?:e-?[0-9]+)?)?)\s*([a-zA-Z]*)')
import logging
logging.basicConfig(filename='dsobj.log')

class DSObj(object):
    script = None
    def __init__(self, value, unit=""):
        if isinstance(value, DSObj):
            self.val = value.val
            self.unit = value.unit
            self.type = value.type
        else:
            if isinstance(value, str):
                res = unit_re.match(value)
                if res:
                    sval, unit = res.groups()
                    print "value match", sval, "unit match", unit
                    value = float(sval)
                    self.type = 'literal'
                else:
                    assert value in self.script.properties
                    self.type = "property"
            elif isinstance(value, tuple):
                assert len(value) in [2,3] # only unary or ternary
                self.type = "computation"
            elif isinstance(value, (int,float)):
                self.type = "literal"
                self.val = float(value)
            else:
                raise Exception("unknown type"+str(type(value)))
            self.val = value
            self.unit = unit
            if self.val == -25. and not self.unit:
                assert False
        
    def cache_result(self, name):
        self.script.add_property(name, str(self))
        self.val = name
        self.type = 'property'
        print self, self.unit
        if not self.unit: assert False
    def __repr__(self):
        if self.type is 'property':
            return self.val
        elif self.type is 'literal':
            return str(self.val) + self.unit
        else:
            if self.val[0] in ['+', '*', '/']:
                return "("+str(self.val[1])+")"+self.val[0]+"("+str(self.val[2])+")"
            else:
                return self.val[0]+"("+",".join(map(str, self.val[1:]))+")"
    def __add__(self, other):
        if not isinstance(other, DSObj):
            other = DSObj(other)
        if self.val == 0:
            return other
        if other.val == 0:
            return self        
        if not self.unit == other.unit:
            print self.unit, other.unit, self, other
            assert False
        unit = self.unit
        if self.type is "literal" and other.type is "literal":
            return DSObj(self.val+other.val, unit)
        else:
            return DSObj(('+', self, other), unit)
    def __radd__(self, other):
        return self + other
    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return (-self) + other
    def __mul__(self, other):
        if not isinstance(other, DSObj):
            other = DSObj(other)
        unit = self.unit + other.unit
        if self.val is 0 or other.val is 0: return DSObj(0, unit)
        if self.val is 1: return other
        if other.val is 1: return self
        if self.type is 'literal' and other.type is 'literal':
            return DSObj(self.val * other.val, unit)
        else:
            return DSObj(('*', self, other), unit)
    def __rmul__(self, other):
        return self * other
    def __div__(self, other):        
        if not isinstance(other, DSObj):
            other = DSObj(other)
        unit = self.unit.replace(other.unit, "", 1)
        if other.val is 1: return self
        if self.type is 'literal' and other.type is 'literal':
            return DSObj(self.val / other.val, unit)
        else:
            return DSObj(('/', self, other), unit)
    def __rdiv__(self, other):
        if self.val is 1: return other
        if not isinstance(other, DSObj):
            other = DSObj(other)
        unit = other.unit.replace(self.unit, "", 1)
        if self.type is 'literal' and other.type is 'literal':
            return DSObj(other.val / self.val, unit)
        else:
            return DSObj(('/', other, self), unit)
    def __neg__(self):
        if self.type is 'literal':
            return DSObj(-self.val, self.unit)
        else:
            return DSObj(('-', self), self.unit)

import math
def sqrt(dso):
    unit = dso.unit[:len(dso.unit)/2]
    if dso.type == "literal":
        return DSObj(math.sqrt(dso.val), unit)
    else:
        return DSObj(("sqrt", dso.val), unit)
def sin(dso):
    unit = ""
    if dso.type == "literal":
        if dso.unit == 'deg':
            v = math.radians(dso.val)
        else:
            v = dso.val
        return DSObj(math.sin(v), unit)
    else:
        return DSObj(("sin", dso), unit)
def cos(dso):
    unit = ""
    if dso.type == "literal":
        if dso.unit == 'deg':
            v = math.radians(dso.val)
        else:
            v = dso.val
        return DSObj(math.cos(v), unit)
    else:
        return DSObj(("cos", dso), unit)
def dmax(*args):
    unit = args[0].unit
    assert all([a.unit == unit for a in args])
    if all([a.type == 'literal' for a in args]):
        return DSObj(max([a.val for a in args]), unit)
    else:
        return DSObj(("max",)+args, unit)
def DSObjLen(v):
    return DSObj(v, unit="um")
        
        