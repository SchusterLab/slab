# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 14:50:09 2011

@author: David Schuster
"""
import slab
import os
import sys
import socket
from optparse import OptionParser

try:
    import Pyro4
    Pyro4Loaded = True
    # Block calls from running simultaneously
    Pyro4.config.SERVERTYPE = 'multiplex'
    Pyro4.config.REQUIRE_EXPOSE = False
    # Pyro4.config.HMAC_KEY = b'6551d449b0564585a9d39c0bd327dcf1'
    Pyro4.config.SERIALIZER = "pickle"
    Pyro4.config.SERIALIZERS_ACCEPTED=set(['json', 'marshal', 'serpent','pickle'])
except ImportError:
    print("Warning: Pyro4 package is not present")
    print("Instrument Servers will not work.")
    Pyro4Loaded = False


class InstrumentManager(dict):
    """InstrumentManager class reads configuration files and
    keeps track of listed instruments and their settings
    :param config_path: Path to configuration file
    """
    def __init__(self, config_path=None, server=False, ns_address=None):
        """Initializes InstrumentManager using config_path if available"""
        dict.__init__(self)
        self.config_path = config_path
        self.config = None
        self.ns_address = ns_address
        #self.instruments={}
        if not server and Pyro4Loaded:
                try:
                    #self.clean_nameserver()
                    self.connect_proxies()
                except Exception as e:
                    print("Warning: Could not connect proxies!")
                    print(e)
        if config_path is not None:
            instruments=self.load_config_file(config_path)
        else:
            instruments=[]
        if server and Pyro4Loaded:
            self.serve_instruments(instruments)
        else:
            for instrument in instruments:
                self[instrument.name] = instrument



    def line_is_comment_or_empty(self, line=""):
        _line = line.strip();
        if len(_line) == 0 or _line[0] == "#":
            return True
        else : return False

    def parse_config_string(self, line):
        params = line.split();
        name, instrument_class, address = params;
        return name, instrument_class, address;

    def load_config_file(self, config_path):
        """Loads configuration file"""
        print("Loaded Instruments: ", end=' ')
        f = open(config_path, 'r')
        instruments=[]
        for line in f.readlines():
            isComment = self.line_is_comment_or_empty(line);
            if not isComment:
                name = self.parse_config_string(line)[0]
                instruments.append(self.load_instrument(line))
        print("!")
        return instruments

    def load_instrument(self, config_string):
        """Loads instrument based on config_string (Name\tAddress\tType)"""
        #print config_string
        name, in_class, addr = self.parse_config_string(config_string);
        fn = getattr(slab.instruments, in_class)
        return fn(name=name, address=addr)

    def __getattr__(self, item):
        """Maps values to attributes.
        Only called if there *isn't* an attribute with this name
        """
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)
            
    def set_alias(self,name,alias):
        """Sets an alias for an instrument"""
        self[alias]=self[name]

    def serve_instruments(self, instruments=None):
        """inst_dict is in form {name:instrument_instance}"""
        Pyro4.config.SERVERTYPE = "multiplex"
        daemon = Pyro4.Daemon(host=socket.gethostbyname(socket.gethostname()))
        ns = Pyro4.locateNS(self.ns_address)
     
        for instrument in instruments:
            uri = daemon.register(instrument)
            ns.register(instrument.name, uri)

            # register all the objects we expose as properties
            # https://pyro4.readthedocs.io/en/stable/servercode.html#autoproxying
            # https://github.com/irmen/Pyro4/blob/master/examples/autoproxy/server.py
            if hasattr(instrument,"autoproxy"):
                for obj in instrument.autoproxy:
                    daemon.register(obj)

            print("Registered: %s\t%s" % (instrument.name, uri))
        daemon.requestLoop()

    def connect_proxies(self):
        ns = Pyro4.locateNS(self.ns_address)
        for name, uri in list(ns.list().items()):
            self[name] = Pyro4.Proxy(uri)

    def get_settings(self):
        """Get settings from all instruments"""
        settings = []
        for k, inst in self.items():
            try:
                settings.append(inst.get_settings())
            except:
                print("Warning! Could not get settings for instrument: %s" % k)
        return settings

    def save_settings(self, path, prefix=None, params={}):
        """Get settings from all instruments and save to a .cfg file"""
        settings = self.get_settings()
        settings.append(params)
        if prefix:
            fname = os.path.join(path, prefix)
        else:
            #print "hey"
            fname = path
        if ".cfg" not in fname.lower():
            fname += '.cfg'
        f = open(fname, 'w')
        for s in settings:
            f.write(repr(s))
            f.write('\n')
        f.close()

    def clean_nameserver(self):
        """Checks to make sure all of the names listed
        in server are really there"""
        ns = Pyro4.locateNS(self.ns_address)
        for name, uri in list(ns.list().items()):
            try:
                proxy=Pyro4.Proxy(uri)
                proxy._pyroTimeout = 0.1
                proxy.get_id()
            except:
                ns.remove(name)

def main(args):
    parser=OptionParser()
    parser.add_option("-f","--file",dest="filename",
                      help="Config file to load",metavar="FILE")
    parser.add_option("-s","--server",action="store_true",dest="server",
                      default=False,help="Act as instrument server")
    parser.add_option("-n","--nameserver","--ns_address",action="store",
                      type="string",dest="ns_address",
                      help="Address of name server (default auto-lookup)" )
    parser.add_option("-g","--gui", action="store_true",dest="gui",default=False,
                      help="Run Instrument Manager in gui mode")
    parser.add_option("-i", action="store_true",dest="interact",default=False,
                      help="interactive option not used.")
    options,args=parser.parse_args(args)

    if options.gui:
        sys.exit(slab.gui.runWin(InstrumentManagerWindow,filename=options.filename,nameserver=options.ns_address))
    else:
        im=InstrumentManager(config_path=options.filename,server=options.server,
                             ns_address=options.ns_address)
        globals().update(im)
        globals()['im']=im
        try:
            globals()['plotter']=liveplot.LivePlotClient()
        except:
            print("Warning: Couldn't load liveplotter")

if __name__ == "__main__":
    try:
        import slab.gui
        from slab.instruments import InstrumentManagerWindow
    except:
        print("Warning: Could not import slab.gui or InstrumentManagerWindow!")
    try:
        import liveplot
    except:
        print("Warning: Could not load liveplot")


    main(sys.argv[1:])
