# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 14:50:09 2011

@author: Phil
"""
import slab.instruments
import os
try:
    import Pyro4
    Pyro4Loaded=True
except:
    print "Warning: Pyro4 package is not present, Instrument Servers will not work."
    Pyro4Loaded=False

class InstrumentManager(dict):
    """InstrumentManager class reads configuration files and 
    keeps track of listed instruments and their settings
    """
    def __init__(self,config_path=None,server=False,ns_address=None):
        """Initializes InstrumentManager using config_path if available"""
        dict.__init__(self)
        self.config_path=config_path
        self.config=None
        self.ns_address=ns_address
        #self.instruments={}
        if config_path is None: 
            if Pyro4Loaded:
                self.connect_proxies()
        else:
            self.load_config_file(config_path)
            if server:
                self.serve_instruments()
            elif Pyro4Loaded:
                self.connect_proxies()
        
    def load_config_file(self,config_path):
        """Loads configuration file"""
        print "Loaded Instruments: ",
        f=open(config_path,'r')
        for line in f.readlines():
            if line[0]!='#' and line[0]!='':
                name=line.split()[0]
                print name,
                self[name]=self.load_instrument(line)
        print "!"
    
    def load_instrument(self,config_string):
        """Loads instrument based on config_string (Name\tAddress\tType)"""
        #print config_string
        params=config_string.split()
        return getattr(slab.instruments,params[1])(name=params[0],address=params[2])

    def serve_instruments(self):
        """inst_dict is in form {name:instrument_instance}"""
        daemon = Pyro4.Daemon()
        ns = Pyro4.locateNS(self.ns_address)
        for name, instrument_instance in self.items():
            uri = daemon.register(instrument_instance)
            ns.register(name, uri)
            print "Registered: %s\t%s" %(name,uri)
        daemon.requestLoop()
        
    def connect_proxies(self):
        self.ns=Pyro4.locateNS()
        for name,uri in self.ns.list().items()[1:]:
            self[name]=Pyro4.Proxy(uri)
        
    def get_settings(self):
        """Get settings from all instruments"""
        settings=[]
        for k,inst in self.iteritems():
            settings.append(inst.get_settings())
        return settings
        
    def save_settings(self,path,prefix=None,params={}):
        """Get settings from all instruments and save to a .cfg file"""
        settings=self.get_settings()
        settings.append(params)
        if prefix:
            fname=os.path.join(path,prefix)
        else:
            #print "hey"
            fname = path
        if ".cfg" not in fname.lower():
            fname+='.cfg'
        f=open(fname,'w')
        for s in settings:
            f.write(repr(s))
            f.write('\n')
        f.close()
        
if __name__=="__main__":
    im = InstrumentManager(r'c:\_Lib\python\slab\instruments\instrument.cfg')
    im.serve_instruments()