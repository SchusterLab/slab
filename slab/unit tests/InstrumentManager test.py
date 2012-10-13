# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 10:41:58 2012

@author: Dave
"""

"""
To do this test you have to first start the nameserver 
and an InstrumentManager server

#To start a nameserver
    python -m Pyro4.naming
    
#To start InstrumentManager Server
    im = InstrumentManager(r'c:\_Lib\python\slab\instruments\instrument.cfg')
    im.serve_instruments()
    
"""

from slab.instruments import InstrumentManager

im=InstrumentManager()

print im.keys()
print im['echo'].echo('This is a test')
print im['random'].random()