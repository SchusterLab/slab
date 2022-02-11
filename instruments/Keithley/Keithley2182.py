# -*- coding: utf-8 -*-
'''
Keithley2182 nanovoltmeter (Keithley199.py)
===========================================
:Author: Kan-Heng Lee
'''
from slab.instruments import VisaInstrument
from slab.datamanagement import SlabFile
import time


class Keithley2182(VisaInstrument):
    '''
    Important note:
    If 2182 is directly connect to the pc through gpib, it is set to 'GPIB0::7::INSTR'
    by defalut. If 2182 is connected to the 6221 through RS232, the address should be
    set to the address of 6221, which by default is .'GPIB0::12::INSTR'.
    '''
    def __init__(self,name="Keithley2182",address='GPIB0::7::INSTR',enabled=True,timeout=1):
        #if ':' not in address: address+=':22518'
        # VisaInstrument.__init__(self,name,address,enabled, term_chars='\r')
        VisaInstrument.__init__(self, name, address, enabled)
        self.query_sleep=0.01
        self.recv_length=65536
        self.term_char='\r'
        self.comm = 'RS232' ###Change this when you switch connection.
        self.data_transfer_time = 0.1
        
    def write_thru(self,command):
        '''
        "GPIB" when 2182a is connected to gpib, pass the command as it is.
        "RS232" when 2182a is connected to RS232 "through" 6221. Note it is not
        connected to PC at all in this config.
        '''
        if self.comm == 'GPIB':
            self.write(command)
        elif self.comm == 'RS232':
            self.write(f''':SYST:COMM:SER:SEND "{command}"''')
    
    def queryb_thru(self,command):
        if self.comm == 'GPIB':
            return self.queryb(command)
        elif self.comm == 'RS232':
            self.write(f''':SYST:COMM:SER:SEND "{command}"''')
            time.sleep(self.query_sleep)
            return self.queryb(':SYST:COMM:SER:ENT?')
    
    def readb_thru(self):
        if self.comm == 'GPIB':
            return self.readb()
        elif self.comm == 'RS232':
            return self.readb(':SYST:COMM:SER:ENT?')
        
    # def wait_for_complete(self):
    #     opc=self.queryb_thru('*OPC?').strip('\n')
    #     while opc != '1':
    #         time.sleep(self.query_sleep)
    #         opc=self.readb_thru().strip('\n')
    # readb does not work.

    def get_id(self):
        return self.queryb_thru('*IDN?')

    def init(self):
        self.write_thru('ABOR')
        time.sleep(1)
        self.write_thru('*rst')
        time.sleep(1)

    def beeper(self,onoff):
        if onoff == 'on':
            bool = 1
        elif onoff == 'off':
            bool = 0
        else:
            print('''input is string 'on' or 'off' ''')
        self.write_thru(f':SYSTem:BEEPer:STATe {bool}')

    def buffer_reset(self, n=1024):
        """
        n is number of measurement points that will be stored in buffer.
        1. Clear whatever in the buffer
        2. Set buffer type to 'Sense[1]'
        3. Set buffer size to max (1024) as defalt.
        4. Select buffer control mode 'Next'
        """
        self.write_thru(':trac:cle')
        self.write_thru(':trac:feed sens1')
        self.write_thru(':trac:poin ' + str(n))
        self.write_thru('trac:feed:control NEXT')

    def set_para(self, channel, v_range = 'auto', integrate_time = 0.05, digit = 6):
        """
        Basic setting of the meters:
        Set measurement channel (1 or 2)
        v_range=expected voltage range to measure. Default is auto. Range can be 0 to 120V.
        integrate_time = measuring speed (in secs), best SNR between 0.01667 to 0.1 seconds.
        digit = displayed digit on the meter screen
        """
        if v_range == 'auto':
            self.write_thru(':sens:volt:rang:auto on')
        else:
            self.write_thru(':sens:volt:rang ' + str(v_range))
        self.write_thru(':sens:chan ' + str(channel))
        self.write_thru(':sens:volt:aper ' + str(integrate_time))
        self.write_thru(':sens:volt:dig ' + str(digit))
        info = f'Keithley 2182 param: ch {channel}, range={v_range}, integrate time={integrate_time}'
        return info

    def get_volt(self, integrate_time=0.05):
        """Returns the last measured voltage in the buffer"""
        self.write_thru(':INIT') #restart measurement cycle.
        time.sleep(integrate_time)
        time.sleep(self.data_transfer_time)
        opc = self.queryb_thru('SENS:DATA:FRES?').strip('\n\n')
        while opc == '':
            time.sleep(self.data_transfer_time)
            opc = self.queryb_thru('SENS:DATA:FRES?').strip('\n\n')
        return float(opc)



#    def integrate_voltage(self, dt):
#        """
#        :param time: time in seconds to integrate the signal
#        :return: mean and standard deviation of the voltage
#        """
#        t0 = time.time()
#        v = []
#        while time.time() < t0+dt:
#            V = self.get_volt()
#            time.sleep(0.017)
#            v.append(V)
#
#        return np.mean(np.array(v)), np.std(np.array(v))



    # def set_mode(self, mode):
    #     """
    #     :param mode: may be one of the following: "VDC", "VAC", "Ohms", "IDC", "IAC"
    #     :return:
    #     """
    #     conversion_table = {'VDC' : 'F0', 'VAC' : 'F1', 'Ohms' : 'F2', 'IDC' : 'F3', 'IAC' : 'F4'}
    #     self.write(conversion_table[mode]+'X')

#if __name__ == '__main__':
#    print("HERE")
#    # #magnet=IPSMagnet(address='COM1')
#    # V=Keithley199(address='GPIB0::26::INSTR')
#    # print V.get_id()
#    # V.set_range_auto()
#    # print V.get_volt()
#    dmm=HP34401A()