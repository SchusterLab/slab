# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 12:31:34 2012

@author: slab
"""

# Instrument server

import Pyro4

def serve_instruments(inst_dict):
    'inst_dict is in form {name:instrument_instance}'
    daemon = Pyro4.Daemon()
    ns = Pyro4.locateNS()
    for name, instrument_instance in inst_dict.items():
        uri = daemon.register(instrument_instance)
        ns.register(name, uri)
    daemon.requestLoop()

def load_instrument(name):
    ns = Pyro4.locateNS()
    return Pyro4.Proxy(ns.lookup(name))

"""
import numpy as np
import matplotlib.pyplot as plt
class TestPyro:
    def hello(self):
        return "hello world"
    def npsin(self, n):
        return np.sin(np.linspace(0, 4*np.pi, n))

def test_server():
    ns = Pyro4.locateNS()
    daemon = Pyro4.Daemon()
    ns.register("testpyro", daemon.register(TestPyro()))
    daemon.requestLoop()

def test_client():
    ns = Pyro4.locateNS()
    proxy = Pyro4.Proxy(ns.lookup("testpyro"))
    print proxy.hello()    
    return proxy.npsin(100)
"""