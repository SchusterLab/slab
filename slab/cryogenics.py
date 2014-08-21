__author__ = 'dave'

"""This package is intended to provide useful functions for calculating cryogenic properties"""


class CryoMaterial:

    def __init__(self,name, description):
        self.name=name
        self.description=description

    def specific_heat(self, T):
        pass

    def thermal_conductivity(self,T):
        pass

    def vapor_pressure(self,T):
        pass


