# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 14:50:09 2011

@author: Phil
"""
import slab.instruments
import os

class InstrumentManager(dict):
    """InstrumentManager class reads configuration files and 
    keeps track of listed instruments and their settings
    """
    def __init__(self,config_path=None):
        """Initializes InstrumentManager using config_path if available"""
        dict.__init__(self)
        self.config_path=config_path
        self.config=None
        #self.instruments={}
        if config_path is not None: self.load_config_file(config_path)
        
    def load_config_file(self,config_path):
        """Loads configuration file"""
        print "Loaded Instruments: ",
        f=open(config_path,'r')
        for line in f.readlines():
            if line[0]!='#':
                name=line.split()[0]
                print name,
                self[name]=self.load_instrument(line)
        print "!"
    
    def load_instrument(self,config_string):
        """Loads instrument based on config_string (Name\tAddress\tType)"""
        #print config_string
        params=config_string.split()
        return getattr(slab.instruments,params[1])(name=params[0],address=params[2])
        
    def get_settings(self):
        settings=[]
        for k,inst in self.iteritems():
            settings.append(inst.get_settings())
        return settings
        
    def save_settings(self,path,prefix=None,params={}):
        settings=self.get_settings()
        settings.append(params)
        if prefix:
            fname=os.path.join(path,prefix)
        else:
            print "hey"
            fname = path
        if ".cfg" not in fname.lower():
            fname+='.cfg'
        f=open(fname,'w')
        for s in settings:
            f.write(repr(s))
            f.write('\n')
        f.close()
        
if __name__=="__main__":
    im = InstrumentManager(r'D:\Dropbox\UofC\_Lib\python\slab\instruments\instrument.cfg')