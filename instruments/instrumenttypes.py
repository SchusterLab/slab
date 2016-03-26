try:
    import visa
    if 'Instrument' not in dir(visa):
        visa.Instrument=visa.instrument

except Exception as e:
    print e
    print "Warning VISA library import failed"
import telnetlib
import socket
import time

try:
    import serial
except ImportError:
    print "Warning serial library import failed."


class Instrument(object):
    """
    A subclass of Instrument is an instrument which communicates over a certain
    channel. The subclass must define the methods write and read, for
    communication over that channel
    """
    address = ''  # Address of instrument
    name = ''  # Instrument Name
    enabled = False  # If enabled=False commands should not be sent
    instrument_type = ''  # Instrument type
    protocol = ''  # Protocol
    id_string = ''  # id string
    query_sleep = 0  # seconds to wait between write and read
    term_char = '\n'  # character to be appended to all writes
    # operation_range={}        #map to hold the operation range

    def __init__(self, name, address='', enabled=True, query_timeout=1000):
        """
        :param name:
        :param address:
        :param enabled:
        :param query_timeout: timeout for low-level queries in milliseconds
        :return:
        """
        self.name = name
        self.address = address
        self.enabled = enabled
        self.query_timeout = query_timeout # timeout for connection, different from timeout for query

    def get_name(self):
        return self.name

    def get_id(self):
        return "Default Instrument %s" % (self.name)

    def query(self, cmd, timeout=None):
        # Note: the implementation here is synchronous. The query timeout is required to be implemented in the
        # self.read() command, and this should provide a simple and nice synchronous interface that's useful
        # for most experiments. In the future we might want to implement a asynchronous IO interface
        #                                                         --- Ge Yang on fixing #1 hanging query.
        self.write(cmd)
        time.sleep(self.query_sleep)
        return self.read(timeout)

    def set_query_timeout(self, timeout):
        self.query_timeout = timeout

    def get_query_timeout(self):
        return self.query_timeout

    def get_settings(self):
        settings = {}
        settings['name'] = self.name
        settings['address'] = self.address
        settings['instrument_type'] = self.instrument_type
        settings['protocol'] = self.protocol
        return settings

    def set_settings(self, settings):
        print settings

    def attr(self, name):
        "re-naming of __getattr__ which is unavailable when proxied"
        return getattr(self, name)

        # def set_operation_range(self, operation_range):
        #    self.operation_range = operation_range

        # def get_operation_range(self):
        #    return self.operation_range


class VisaInstrument(Instrument):
    def __init__(self, name, address='', enabled=True, query_timeout=1.0, **kwargs):
        Instrument.__init__(self, name, address, enabled, query_timeout, **kwargs)
        if self.enabled:
            self.protocol = 'VISA'
            self.query_timeout = query_timeout
            address = address.upper()
            print address
        self.instrument = visa.Instrument(address, timeout=query_timeout, **kwargs)

    def write(self, s):
        if self.enabled: self.instrument.write(s + self.term_char)

    def read(self, timeout=None):
        # todo: implement timeout, reference SocketInstrument.read
        if self.enabled: return self.instrument.read()

    def close(self):
        if self.enabled: self.instrument.close()


# def __del__(self):
#        if self.enabled: self.close()

class TelnetInstrument(Instrument):
    def __init__(self, name, address='', enabled=True, query_timeout=10):
        Instrument.__init__(self, name, address, enabled, query_timeout, **kwargs)
        self.protocol = 'Telnet'
        if len(address.split(':')) > 1:
            self.port = int(address.split(':')[1])
        if self.enabled:
            self.tn = telnetlib.Telnet(address.split(':')[0], self.port)

    def write(self, s):
        if self.enabled: self.tn.write(s + self.term_char)

    def read(self, timeout=None):
        # todo: implement timeout, reference SocketInstrument.read
        if self.enabled: return self.tn.read_some()

    def close(self):
        if self.enabled: self.tn.close()

    #    def __del__(self):


# if self.enabled: self.tn.close()

import select
class SocketInstrument(Instrument):
    default_port = 23

    def __init__(self, name, address='', enabled=True, recv_length=1024, query_timeout=1000, **kwargs):
        Instrument.__init__(self, name, address, enabled, query_timeout, **kwargs)
        self.protocol = 'socket'
        self.recv_length = recv_length
        if len(address.split(':')) > 1:
            self.port = int(address.split(':')[1])
            self.ip = address.split(':')[0]
        else:
            self.ip = address
            self.port = self.default_port
        self.on_enable()

    def on_enable(self):
        if self.enabled:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.ip, self.port))
            self.set_query_timeout(self.query_timeout)
            self.socket.setblocking(0)

    def set_enable(self, enable=True):
        self.enabled = enable
        self.on_enable()

    def write(self, s):
        if self.enabled: self.socket.send(s + self.term_char)

    def read(self, timeout=None):
        # Note: make sure you use query_timeout instead of timeout. self.timeout is for
        # timeout/wait before connection only.
        #
        # Note: timeout is in milliseconds, not in seconds.
        #
        if timeout == None: timeout = self.query_timeout
        ready = select.select([self.socket], [], [], timeout/1000.0)
        if (ready[0] and self.enabled):
            return self.socket.recv(self.recv_length)

class SerialInstrument(Instrument):
    # todo: the `baudrate` and `querysleep` need to be updated to band_rate and query_sleep
    def __init__(self, name, address, enabled=True, query_timeout=1.0,
                 recv_length=1024, baudrate=9600, querysleep=1):
        Instrument.__init__(self, name, address, enabled)
        self.protocol = 'serial'
        self.enabled = enabled
        if self.enabled:
            try:
                self.ser = serial.Serial(address, baudrate)
                # try: self.ser = serial.Serial(int(address.upper().split('COM')[1])-1, baudrate)
                # except:
                #    try: self.ser = serial.Serial(int(address), baudrate)
                #    except: raise ValueError
            except serial.SerialException:
                print 'Cannot create a connection to port ' + str(address) + '.\n'
        self.set_query_timeout(query_timeout)
        self.recv_length = recv_length
        self.query_sleep = querysleep

    # todo: `set_timeout` is deprecated because of conflicting meanings from different libraries. A `set_query_timeout` should be implemented
    # def set_query_timeout(self, timeout):
    #     Instrument.set_query_timeout(self, timeout)
    #     if self.enabled: self.ser.setTimeout(self.query_timeout)

    def test(self):
        self.ser.setTimeout(self.timeout)

    def set_query_sleep(self, querysleep):
        self.query_sleep = querysleep

    def write(self, s):
        if self.enabled: self.ser.write(s + self.term_char)

    def read(self, timeout=None):
        # todo: implement timeout, reference SocketInstrument.read
        if self.enabled: return self.ser.read(self.recv_length)

    def reset_connection(self):
        self.ser.close()
        time.sleep(self.query_sleep)
        self.ser.open()

    def __del__(self):
        try:
            self.ser.close()
        except Exception as e:
            print e
            print 'cannot properly close the serial connection.'


class WebInstrument(Instrument):
    def __init__(self, name, address='', enabled=True):
        Instrument.__init__(self, name, address, enabled)
        self.protocol = 'http'
        self.enabled = enabled
