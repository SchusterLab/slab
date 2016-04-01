__author__ = 'Dave'

import usb.core
import usb.util
from liveplot import LivePlotClient
import time
import math

def grab():
    try:
        # first endpoint
        endpoint = dev[0][(0,0)][0]

        # read a data packet
        attempts = 10
        data = None
        while data is None and attempts > 0:
            try:
                data = dev.read(endpoint.bEndpointAddress,
                                   endpoint.wMaxPacketSize)
            except usb.core.USBError as e:
                data = None
                if e.args == ('Operation timed out',):
                    attempts -= 1
                    print "timed out... trying again"
                    continue

        return data
    except usb.core.USBError as e:
        print "USBError: " + str(e.args)
    except IndexError as e:
        print "IndexError: " + str(e.args)




def listen():
    DATA_MODE_GRAMS = 2
    DATA_MODE_OUNCES = 11

    last_raw_weight = 0
    last_raw_weight_stable = 4

    print "listening for weight..."

    while True:
        time.sleep(.5)

        weight = 0
        print_weight = ""

        data = grab()
        if data != None:
            raw_weight = data[4] + data[5] * 256

        # +/- 2g
        if raw_weight > 0 and abs(raw_weight-last_raw_weight) > 0 and raw_weight != last_raw_weight:
            last_raw_weight_stable = 4
            last_raw_weight = raw_weight

        if raw_weight > 0 and last_raw_weight_stable >= 0:
            last_raw_weight_stable -= 1

        if raw_weight > 0 and last_raw_weight_stable == 0:
            if data[2] == DATA_MODE_OUNCES:
                ounces = raw_weight * 0.1
                weight = math.ceil(ounces)
                print_weight = "%s oz" % ounces
            elif data[2] == DATA_MODE_GRAMS:
                grams = raw_weight
                weight = math.ceil(grams)
                print_weight = "%s g" % grams

        #print "stable weight: " + print_weight
        lp.append_y("scale",raw_weight)


lp=LivePlotClient()

VENDOR_ID = 0x0922
PRODUCT_ID = 0x8009

#print usb.core.find()
# find the USB device
dev = usb.core.find(idVendor=VENDOR_ID,
                       idProduct=PRODUCT_ID)

# use the first/default configuration
dev.set_configuration()
# first endpoint

listen()

