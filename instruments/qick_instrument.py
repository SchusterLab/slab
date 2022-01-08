from slab.instruments import Instrument
from qick import QickSoc
import Pyro4
import socket

class QickInstrument(Instrument, QickSoc):
    
    def serve_instrument(inst, ns_address="192.168.14.1", ns_port=None):
        Pyro4.config.SERVERTYPE = "multiplex"
        daemon = Pyro4.Daemon(host=socket.gethostbyname(socket.gethostname()))
        ns = Pyro4.locateNS(ns_address,port=ns_port)
        uri = daemon.register(inst)
        ns.register(inst.name, uri)
        print("Registered: %s\t%s" % (inst.name, uri))
        daemon.requestLoop()

    def __init__(self, name, address='', enabled=True, timeout=1, query_sleep=0, **kwargs):
        Instrument.__init__(self, name=name, address=address, enabled=enabled, timeout=timeout, query_sleep=query_sleep, **kwargs)
        QickSoc.__init__(self)
        #self.reset()

    def reset(self, bitfile=None, force_init_clks=False,ignore_version=True, **kwargs):
        QickSoc.__init__(self)
        #self.soc = QickSoc(bitfile, force_init_clks,ignore_version, **kwargs)
    
    """
    def single_write(self, addr, data):
        self.soc.single_write(addr,data)
        
    def single_read(self, addr):
        return self.soc.single_read(addr)

    def read_dmem(self, addr=0, length=100):
        return self.soc.read_dmem(self, addr, length)

    def load_dmem(self, buff_in, addr=0):
        return self.soc.read_dmem(self, buff_in, addr)

    def start(self):
        return self.soc.start()

    def stop(self):
        return self.soc.stop()
        
    def get_accumulated(self, ch, address=0, length=None):
        if length is None: 
            length=self.soc.AxisAvgBuffer.AVG_MAX_LENGTH
        return self.soc.get_accumulated(ch,address,length)
    
    def get_decimated(self, ch, address=0, length=None):
        if length is None:
            length=self.soc.AxisAvgBuffer.BUF_MAX_LENGTH
        return self.soc.get_decimated(ch,address,length)

    def configure_readout(self, ch, output, frequency):
        return self.soc.configure_readout(ch,output,frequency)

    def config_avg(self, ch, address=0, length=1, enable=True):
        return self.soc.config_avg(ch,address,length,enable)
    
    def enable_avg(self, ch):
        return self.soc.enable_avg(ch)

    def config_buf(self, ch, address=0, length=1, enable=True):
        return self.soc.config_buf(ch,address,length,enable)

    def enable_buf(self, ch):
        return self.soc.enable_buf(ch)
    
    def set_nyquist(self, ch, nqz):
        return self.soc.set_nyquist(ch,nqz)

    def load_bin_program(self, binprog):
        return self.soc.load_bin_program(binprog)

    def load_pulse_data(self, ch, idata, qdata, addr):
        return self.soc.load_pulse_data (ch,idata,qdata,addr)

    def get_avg_max_length(self, ch=0):
        return self.soc.get_avg_max_length(ch)
    """

