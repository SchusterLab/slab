__author__ = 'ge'

from slab.instruments import WebInstrument
import urllib2 as ur


class HeaterSwitch(WebInstrument):
    def __init__(self, name="HeaterSwitch", address='', enabled=True):
        WebInstrument.__init__(self, name, address, enabled)

    def get_switch(self, num):
        if num in (0, 1, 2, 3, 4):
            return ur.urlopen(self.address + '/Q' + str(num)).read().split('\r\n')[1]
        else:
            print"ERROR: Invalid switch number"

    def get_switches(self):
        return [self.get_switch(ii) for ii in range(1, 5)]

    def set_switch(self, num, state):
        if num in (1, 2, 3, 4):
            if state.lower() == 'on':
                return ur.urlopen(self.address + '/S' + str(num) + '0').read().split('\r\n')[1]
            elif state.lower() == 'off':
                return ur.urlopen(self.address + '/S' + str(num) + '1').read().split('\r\n')[1]
            else:
                print "ERROR: Invalid command (ON\OFF) only"
        else:
            print "ERROR: Invalid switch number"

    def reset_switch(self):
        return ur.urlopen(self.address + '/R').read().split('\r\n')[1]
