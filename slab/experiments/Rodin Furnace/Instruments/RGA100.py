"""
Author: Andrew Oriani
May 2019

This is a driver for the Stanford Research Systems RGA100 residual gas analyzer

UPDATE: 11/2020

Parallel and async functionality added using threading to allow background updating and operation. locks seperate function calls
to the serial handler of the RGA. Would caution it's use inside of the instrument manager server, however. Has not been thoroughy tested
in that context.
"""

import threading
from threading import RLock, Lock 
import slab
from slab.instruments import SerialInstrument
import time
import numpy as np
import warnings

class SerialHandler(SerialInstrument):
    def __init__(self, name="", address='COM3',enabled=True, timeout=.5, query_timeout=20):
        SerialInstrument.__init__(self, name, address, enabled, timeout, baudrate=28800, query_sleep=0.05)
        self.term_char='\r'
        self.lock=RLock()
        self.query_timeout=query_timeout
        self.comm_error=list()

    def f_query(self, cmd, bit_len=8):
        while False:
            self.lock.acquire(blocking=False)
        with self.lock:
            self.f_write(cmd)
            len_stat = 0
            et = 0
            st=time.time()
            while len_stat == 0 and et < self.query_timeout:
                stat = self.ser.read(bit_len).decode()
                len_stat = len(stat)
                et=time.time()-st

            if et>self.query_timeout:
                self.comm_error.append('Query timeout met, check serial RGA connection....')
                raise RuntimeError('Query timeout met, check serial RGA connection....')
               

        return stat

    def f_queryb(self, cmd):
        while False:
            self.lock.acquire(blocking=False)
        with self.lock:
            self.f_write(cmd)
            len_stat=0
            et = 0
            st=time.time()
            while len_stat == 0 and et < self.query_timeout:
                stat = self.ser.read(4)
                len_stat = len(stat)
                et=time.time()-st

            if et>self.query_timeout:
                self.comm_error.append('Query timeout met, check serial RGA connection....')
                raise RuntimeError('Query timeout met, check serial RGA connection....')   

        return stat

    def f_write(self, cmd):
        while False:
            self.lock.acquire(blocking=False)
        with self.lock:
            self.ser.write(str(cmd+self.term_char).encode())


class RGA100functions(SerialHandler):
    def __init__(self, name="", address='COM3',enabled=True, timeout=.5, query_timeout=20):
        super(RGA100functions, self).__init__(name, address, enabled, timeout, query_timeout)
        self.fil_admission=0
        self.error_status=''

    def set_filament_output(self, state=False, current=1.0):
        # check RGA com status

        if state == True:
            # check emission current in mA
            adj_emission = .02 * round(current / .02)
            FL_off_check = float(self.f_query('FL?', bit_len=8).split('\n\r')[0])
            if FL_off_check == 0.0:
                FL_on = self.f_query('FL0.02', bit_len=3)
            else:
                pass
            if adj_emission >= 0.0 and adj_emission <= 3.5:
                FL_stat = self.f_query('FL%.2f' % adj_emission, bit_len=3).split('\n\r')[0]
                self.fil_admission=adj_emission
            else:
                raise Exception('Filament emission current must be between 0mA and 3.5mA, %.2fmA entered.' % adj_emission)
        elif state == False:
            FL_off_check = float(self.f_query('FL?', bit_len=8).split('\n\r')[0])
            FL_stat = '0'
            if FL_off_check == 0.0:
                pass
            else:
                FL_stat = self.f_query('FL0', bit_len=3).split('\n\r')[0]
        else:
            raise Exception('Filament on state must be boolean')
        if len(FL_stat) == 0:
            stat = self.f_query('IN0', bit_len=3)
            if len(stat.split('\n\r')[:-1]) >= 2:
                pass
            elif stat.split('\n\r')[0] == '0':
                pass
            else:
                self.err_stat()
                raise Exception('Status error in setting electron filament emission.')
        else:
            if int(FL_stat) != 0:
                self.err_stat()
                raise Exception('Status error in setting filament emission current.')
            else:
                pass
        FL_val = float(self.f_query('FL?').split('\n\r')[0])
        return FL_val

    def _set_filament_on(self):
        stat=self.f_query('FL%.2f' % self.fil_admission, bit_len=3)

    def _set_filament_off(self):
        stat=self.f_query('FL0', bit_len=3)

    def ionizer_set(self, e_energy=70, ion_energy='high', foc_volts=90, fil_emission=.5, filament_on=False):

        # check if focus voltage is set correctly, must ve between 0V and 150V
        if np.around(foc_volts) >= 0 and np.around(foc_volts) <= 150:
            FV_stat = self.f_query('VF%i' % np.around(foc_volts), bit_len=3).split('\n\r')[0]
            if len(FV_stat) == 0:
                stat = self.f_query('IN0', bit_len=3)
                if len(stat.split('\n\r')[:-1]) == 2:
                    print('...checking focus voltage setting status: ok')
                    pass
                else:
                    self.err_stat()
                    raise Exception('Status error in setting focus voltage')
            else:
                if int(FV_stat) != 0:
                    self.err_stat()
                    raise Exception('Status error in setting focus voltage')
                else:
                    pass
        else:
            raise Exception('Focus volts must be between 0V and 150V, %.2fV entered' % foc_volts)
        FV_val = float(self.f_query('VF?').split('\n\r')[0])
        print('Focus Volts set to: %.2fV' % FV_val)

        # check electron energy
        if e_energy >= 25 and e_energy <= 105:
            # set electron energy
            EE_stat = self.f_query('EE%i' % np.around(e_energy), bit_len=3).split('\n\r')[0]
            EE_val = self.f_query('EE?')
            if len(EE_stat) == 0:
                stat = self.f_query('IN0', bit_len=3)
                if len(stat.split('\n\r')[:-1]) == 2:
                    print('...checking electron acceleration energy status: ok')
                    pass
                else:
                    self.err_stat()
                    raise Exception('Status error in setting electron acceleration energy.')
            else:
                if int(EE_stat) != 0:
                    self.err_stat()
                    raise Exception('Status error in setting electron acceleration energy.')
                else:
                    pass
        else:
            raise Exception('Electron acceleration energy must be between 25eV and 105eV, %.2feV entered' % e_energy)
        EE_val = float(self.f_query('EE?').split('\n\r')[0])
        print('Electron acceleration energy value: %.2feV' % EE_val)

        # set ion energy scaling
        if ion_energy != 'high' and ion_energy != 'low':
            raise Exception('Invalid ion energy scaling, must be high or low.')
        else:
            if ion_energy == 'high':
                IE_stat = self.f_query('IE1', bit_len=3).split('\n\r')[0]
            elif ion_energy == 'low':
                IE_stat = self.f_query('IE0', bit_len=3).split('\n\r')[0]
            else:
                pass
            if len(IE_stat) == 0:
                stat = self.f_query('IN0', bit_len=3)
                if len(stat.split('\n\r')[:-1]) == 2:
                    print('...checking ion energy scaling: ok')
                    pass
                else:
                    self.err_stat()
                    raise Exception('Status error in setting ion energy scaling.')
            else:
                if int(IE_stat) != 0:
                    self.err_stat()
                    raise Exception('Status error in setting ion energy scaling.')
                else:
                    pass
        IE_val = int(self.f_query('IE?', bit_len=3).split('\n\r')[0])
        if IE_val == 0:
            print('Ion energy scale set to: low')
        else:
            print('Ion energy scale set to: high')

        if filament_on == True:
            FL_val = self.set_filament_output(True, current=fil_emission)
            print('Filament is on with emission current of %.2f' % FL_val)
        elif filament_on == False:
            FL_val = self.set_filament_output(False, current=fil_emission)
            print('Filament is currently off')

        print('Filament parameters successfully set!')
        return

    def ionizer_set_default(self, filament_on=False):
        if int(self.f_query('IN0', bit_len=3).split('\n\r')[0]) == 0:
            pass
        else:
            raise Exception('Check com status of SRS RGA100 head')

        time.sleep(.5)
        EE_stat = self.f_query('EE*', bit_len=3).split('\n\r')[0]
        FV_stat = self.f_query('VF*', bit_len=3).split('\n\r')[0]
        IE_stat = self.f_query('IE*', bit_len=3).split('\n\r')[0]
        if filament_on == True:
            FL_stat = self.f_query('FL*', bit_len=3).split('\n\r')[0]
        if filament_on == False:
            FL_off_check = float(self.f_query('FL?', bit_len=8).split('\n\r')[0])
            if FL_off_check == 0.0:
                pass
            else:
                FL_stat = self.f_query('FL0', bit_len=3).split('\n\r')[0]
        elif filament_on == True:
            FL_stat = self.f_query('FL*', bit_len=3).split('\n\r')
        else:
            raise Exception('Filament on state must be boolean')

        print('Ionizer parameters set to default values successfully!')

        return

    def get_id(self):
        ID = self.f_query('ID?', bit_len=25).split('\n\r')[0]
        return ID

    def err_stat(self):
        self.error_status = self.f_query('IN0', bit_len=3).split('\n\r')[:-1]

    def set_ion_energy(self, energy='high'):
        # set ion energy scaling
        if energy != 'high' and energy != 'low':
            raise Exception('Invalid ion energy scaling, must be high or low.')
        else:
            if energy == 'high':
                IE_stat = self.f_query('IE1', bit_len=3).split('\n\r')[0]
            elif energy == 'low':
                IE_stat = self.f_query('IE0', bit_len=3).split('\n\r')[0]
            else:
                pass
            if len(IE_stat) == 0:
                stat = self.f_query('IN0', bit_len=3)
                if len(stat.split('\n\r')[:-1]) == 2:
                    pass
                else:
                    self.err_stat()
                    raise Exception('Status error in setting ion energy scaling.') 
            else:
                if int(IE_stat) != 0:
                    self.err_stat()
                    raise Exception('Status error in setting ion energy scaling.')
                else:
                    pass
        IE_val = int(self.f_query('IE?', bit_len=3).split('\n\r')[0])
        return IE_val

    def set_focus_voltage(self, voltage=90):

        if np.around(voltage) >= 0 and np.around(voltage) <= 150:
            FV_stat = self.f_query('VF%i' % np.around(voltage), bit_len=3).split('\n\r')[0]
            if len(FV_stat) == 0:
                stat = self.f_query('IN0', bit_len=3)
                if len(stat.split('\n\r')[:-1]) == 2:
                    pass
                else:
                    self.err_stat()
                    raise Exception('Status error in setting focus voltage')
            else:
                if int(FV_stat) != 0:
                    self.err_stat()
                    raise Exception('Status error in setting focus voltage')
                else:
                    pass
        else:
            raise Exception('Focus volts must be between 0V and 150V, %.2fV entered' % voltage)
        FV_val = float(self.f_query('VF?').split('\n\r')[0])
        return FV_val

    def set_electron_energy(self, energy=70):
        if energy >= 25 and energy <= 105:
            # set electron energy
            EE_stat = self.f_query('EE%i' % np.around(energy), bit_len=3).split('\n\r')[0]
            EE_val = self.f_query('EE?')
            if len(EE_stat) == 0:
                stat = self.f_query('IN0', bit_len=3)
                if len(stat.split('\n\r')[:-1]) == 2:
                    print('...checking electron acceleration energy status: ok')
                    pass
                else:
                    self.err_stat()
                    raise Exception('Status error in setting electron acceleration energy.')
            else:
                if int(EE_stat) != 0:
                    self.err_stat()
                    raise Exception('Status error in setting electron acceleration energy.')
                else:
                    pass
        else:
            raise Exception('Electron acceleration energy must be between 25eV and 105eV, %.2feV entered' % energy)
        EE_val = float(self.f_query('EE?').split('\n\r')[0])
        return EE_val

    def get_filament_state(self):
        FL_val = float(self.f_query('FL?', bit_len=8).split('\n\r')[0])
        if FL_val == 0.0:
            state = 0
        elif FL_val != 0.0:
            state = 1
        else:
            pass
        return [state, FL_val]

    def set_scan_rate(self, NF=5):
        if type(NF) == int:
            if NF >= 0 and NF <= 7:
                self.f_write('NF%i' % int(NF))
            else:
                raise Exception('Invalid scan rate value, must be 0-7')
        else:
            raise Exception('Scan rate must be integer value between 0 and 7')
        return

    def reset_TP_state(self):
        self.f_write('TP1')
        return

    def get_ST_scale_factor(self):
        return float(self.f_query('ST?', bit_len=8).split('\n\r')[0])

    def get_SP_scale_factor(self):
        return float(self.f_query('SP?', bit_len=8).split('\n\r')[0])

    def electrometer_cal(self):
        st=time.time()
        stat_val = self.f_query('CL').split('\n\r')[0]
        elapsed_time=time.time()-st
        if int(stat_val) == 0:
            pass
        else:
            raise Exception('Error code %s in electometer calibration'%stat_val)
        return elapsed_time

    def _get_pressure(self, units='torr', TP_scale=.0052):
        meas = self.f_queryb('TP?')
        data = int.from_bytes(meas, byteorder='little', signed=True)
        if units == 'torr':
            factor = 1.
        elif units == 'mbar':
            factor = 1000.00/750.062
        else:
            raise Exception('Units must be in mbar or torr')
        return (data * 1E-16 * factor) / (TP_scale * 1E-3)

    def zero_offset_cal(self):
        st=time.time()
        stat_val = self.f_query('CL').split('\n\r')[0]
        elapsed_time=time.time()-st
        if int(stat_val) == 0:
            pass
        else:
            raise Exception('Error code %s in electometer calibration'%stat_val)
        return

    def _single_mass_scan(self, mass=1, sens_scale=.1218, units='torr'):
        if mass >= 1 and mass <= 100:
            pass
        else:
            raise Exception('Mass oustide of allowable range of 1-100amu')
        if type(mass) == int:
            pass
        else:
            raise Exception('Mass must be integer input between 1-100amu')
        if units == 'torr':
            factor = 1.
        elif units == 'mbar':
            factor = 1000.00/750.062
        else:
            raise Exception('Units must be in mbar or torr')
        raw_data = self.f_queryb('MR%i'%mass)
        data = int.from_bytes(raw_data, byteorder='little', signed=True)
        self.f_write('MR0')
        return (data * 1E-16 * factor) / (sens_scale * 1E-3)

    def _multi_mass_scan(self, mass_vals, sens_scale=.1218, units='torr'):
        if units == 'torr':
            factor = 1.
        elif units == 'mbar':
            factor = 1000.00 / 750.062
        else:
            raise Exception('Units must be in mbar or torr')
        pressure_list = []
        for I, mass in enumerate(mass_vals):
            if mass >= 1 and mass <= 100:
                pass
            else:
                raise Exception('Mass oustide of allowable range of 1-100amu')
            if type(mass) == int:
                pass
            else:
                raise Exception('Mass must be integer input between 1-100amu')
            raw_data = self.f_queryb('MR%i' % mass)
            data = int.from_bytes(raw_data, byteorder='little', signed=True)
            pressure_list.append((data * 1E-16 * factor) / (sens_scale * 1E-3))
        self.f_write('MR0')
        return pressure_list

    def reset_SP(self):
        self.f_write('SP0.1218')
        return

    def reset_ST(self):
        self.f_write('ST0.0052')
        return

    def reset_RGA(self):
        self.f_query('IN1', bit_len=3)
        return

    def close(self):
        self.ser.close()

class RGA100(RGA100functions):
    def __init__(self, name="", address='COM3',enabled=True, timeout=.5, thread_timeout=20, query_timeout=20):
        super(RGA100, self).__init__(name, address, enabled, timeout, query_timeout)
        self.thread_timeout=thread_timeout
        self.max_threads=5
        self.jobs=[]

    def multi_mass_scan(self, mass_vals, sens_scale=.1218, units='torr', parallel=False):
        self.MM_scan_vals=None
        if parallel==True:
            self._thread_check()
            thread=self._set_thread('MMscan', self._par_multi_mass_scan, mass_vals, sens_scale, units)
            thread.start()
            return thread
        else:
            self._par_multi_mass_scan(mass_vals, sens_scale, units)
            return self.MM_scan_vals

    def get_pressure(self, units='torr', TP_scale=.0052, parallel=False):
        self.P_scan_val=None
        if parallel==True:
            self._thread_check()
            thread=self._set_thread('Pscan', self._par_get_pressure, units, TP_scale)
            thread.start()
            return thread
        else:
            self._par_get_pressure(units, TP_scale)
            return self.P_scan_val

    def set_filament_on(self, parallel=False):
        if parallel==True:
            self._thread_check()
            thread=self._set_thread('FilTog', self._par_filament_toggle, True)
            thread.start()
            return thread
        else:
            self._par_filament_toggle(state=True)

    def set_filament_off(self, parallel=False):
        if parallel==True:
            self._thread_check()
            thread=self._set_thread('FilTog', self._par_filament_toggle, False)
            thread.start()
            return thread
        else:
            self._par_filament_toggle(state=False)
        
    def _set_thread(self, name, func, *args):
        thread = threading.Thread(name=self._thread_name_check(name), target=func, args=args)
        self.jobs.append(thread)
        self.num_threads+=1
        return thread

    def _thread_name_check(self, name):
        for jobs in self.jobs:
            if jobs._name==name:
                name_split=name.split('_')
                if len(name_split)==1:
                    name=name+'_%i'%(1)
                else:
                    name=name_split[0]+'_'+str(int(name_split[1])+1)
                name=self._thread_name_check(name)
            else:
                name=name
        return name

    def _del_dead_threads(self):
        if self.jobs==[]:
            pass
        else:
            for I, job in enumerate(self.jobs):
                if job.is_alive()==False:
                    del(self.jobs[I])
        self.num_threads=len(self.jobs)

    def _thread_check(self):
        self._del_dead_threads()
        st=time.time()
        et=0
        while self.num_threads>=self.max_threads and et<self.thread_timeout:
            self._del_dead_threads()
            et=time.time()-st
        if et>self.thread_timeout:
            raise RuntimeError('Timeout met in waiting for threads to close')

    def _par_multi_mass_scan(self, mass_vals, sens_scale, units):
        self.MM_scan_vals=self._multi_mass_scan(mass_vals, sens_scale, units)

    def _par_get_pressure(self, units, TP_scale):
        self.P_scan_val=self._get_pressure(units, TP_scale)

    def _par_filament_toggle(self, state=True):
        if state==True:
            self._set_filament_on()
        else:
            self._set_filament_off()

