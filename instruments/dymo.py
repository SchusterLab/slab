import usb.core
import usb.util

from slab.instruments import Instrument

VENDOR_ID = 0x0922
PRODUCT_ID = 0x8009

class Dymo(Instrument):

    def __init__(self,name='Dymo', address='', enabled=True ):
        Instrument.__init__(self,name,address,enabled)
        dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)


        interface = 0
        if dev.is_kernel_driver_active(interface) is True:
            #print "but we need to detach kernel driver"
            dev.detach_kernel_driver(interface)

        # use the first/default configuration
        dev.set_configuration()
        #print "claiming device"
        usb.util.claim_interface(dev, interface)
        self.dev=dev

    def get_id(self):
        devmanufacturer = usb.util.get_string(self.dev, 256, 1)
        devname = usb.util.get_string(self.dev, 256, 2)
        return devmanufacturer + " " + devname

    def twos_comp(self,val, bits):
        """compute the 2's compliment of int value val"""
        if (val & (1 << (bits - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
            val = val - (1 << bits)        # compute negative value
        return val 

    def get_weight(self):

        try:
            # first endpoint
            endpoint = self.dev[0][(0,0)][0]

            # read a data packet
            attempts = 10
            data = None
            while data is None and attempts > 0:
                try:
                    data = self.dev.read(endpoint.bEndpointAddress,endpoint.wMaxPacketSize)
                except usb.core.USBError as e:
                    data = None
                    if e.args == ('Operation timed out',):
                        attempts -= 1
                        print "timed out... trying again"
                        continue
            scaling_factor=10.0**self.twos_comp(data[3],8)
            if data[2] == 12:
		lbs_factor = 0.453592
            else:
                lbs_factor = 1
            raw_weight=(data[4] + data[5] * 256)*scaling_factor
            return raw_weight*lbs_factor
        except usb.core.USBError as e:
            print "USBError: " + str(e.args)
        except IndexError as e:
            print "IndexError: " + str(e.args)


if __name__ =="__main__":
    d=Dymo()
    print d.get_id()
    print d.get_weight()

