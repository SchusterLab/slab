"""
BK Precision XLN10014 Power supply driver

"""
import slab
from slab.instruments import SerialInstrument
import time
import threading
from threading import Lock, RLock 

class SerialHandler(SerialInstrument):
    def __init__(self, name="", address='COM3',enabled=True, timeout=.05, query_timeout=20):
        SerialInstrument.__init__(self, name, address, enabled, timeout, baudrate=57600, query_sleep=0.05)
        self.term_char='\r\n'
        self.serial_lock=RLock()
        self.query_timeout=query_timeout

    def p_query(self, cmd, bit_len=1024):
        while False:
            self.serial_lock.acquire(blocking=False)
        with self.serial_lock:
            self.p_write(cmd)
            et = 0
            st=time.time()
            mes=''
            while mes == '' and et < self.query_timeout:
                mes = self.ser.read(bit_len).decode()
                et=time.time()-st
            
        if et>self.query_timeout:
            raise RuntimeError('Query timeout met, check serial connection....')   

        return mes

    def p_write(self, cmd):
        while False:
            self.serial_lock.acquire(blocking=False)
        with self.serial_lock:
            self.ser.write(str(cmd+self.term_char).encode())

class XLN10014functions(SerialHandler):
    """
    The XLN10014 uses a baudrate of 57600 and a <CR><NL> termination character
    """
    def __init__(self, name="", address='COM3',enabled=True, timeout=.02, query_timeout=20):
        super(XLN10014functions, self).__init__(name, address, enabled, timeout, query_timeout)
        self.max_threads=1
        self.num_threads=0
        self.jobs=[]
        self.voltage_offset=.011
        self.thread_lock=Lock()

    def _set_thread(self, name, func, *args):
        self._del_dead_threads()
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
        while self.num_threads>=self.max_threads:
            self._del_dead_threads()

    def _par_hold(self, hold_time):
        start_time=time.time()
        if self.get_state():
            pass
        else:
            return 
        et=time.time()-start_time
        while et<=hold_time and not self.stop_ramp:
            et=time.time()-start_time


    def _par_volt_ramp(self, start_volt, stop_volt, rate, interval):
        self.set_volt(start_volt)
        if interval<1:
            interval=1
        else:
            pass
        
        volt_inc=rate*interval
        if stop_volt>start_volt:
            start_volt=start_volt+(stop_volt-start_volt)%volt_inc+volt_inc
        else:
            start_volt=start_volt+(stop_volt-start_volt)%volt_inc

        sign=(stop_volt-start_volt)/abs(stop_volt-start_volt)

        if self.get_state():
            pass
        else:
            return 
        volt_set=start_volt
        start_time=time.time()
        while not round(volt_set, 4)==round(stop_volt, 4) and not self.stop_ramp:
            et=time.time()-start_time
            if not self.hold:
                volt_set+=sign*volt_inc
                self.set_volt(volt_set)
            else:
                pass
            time.sleep(.2)
            time.sleep(interval-(time.time()-start_time)%interval)
        self.elapsed_time=time.time()-start_time
        return 

    def _check_for_alive(self):
        while False:
            self.thread_lock.acquire(blocking=False)
        with self.thread_lock:
            current_ident=threading.get_ident()
            alive_jobs=[]
            if self.jobs==[]:
                pass
            else:
                for I, job in enumerate(self.jobs):
                    if job._ident==current_ident:
                        pass
                    else:
                        if job.is_alive()==True:
                            alive_jobs.append(job)
                        elif job.is_alive()==False:
                            pass
            if len(alive_jobs)>0:
                state=True
            else:
                state=False
        return state

    def _get_job_names(self):
        names=[]
        if self.jobs==[]:
            pass
        else:
            for job in self.jobs:
                if job.is_alive():
                    names.append(job._name)
                else:
                    pass
        return names

    def get_id(self):
        if self._check_for_alive():
            return
        self.p_write('*CLS')
        vers=self.p_query('VER?', bit_len=6).split('\r\n')[0]
        ID=self.p_query('*IDN?', bit_len=40).split('\r\n')[0]
        # print('version: %s, ID: %s '%(vers,ID))
        return vers, ID


    def set_volt(self, voltage):
        if self._check_for_alive():
            return 
        self.p_write('VOLT %s'%(voltage+self.voltage_offset))
        return

    def get_meas_volt(self):
        vout=''
        while vout=='':
            vout=self.p_query('VOUT?').split('\r\n')[0]
        return float(vout)

    def set_current(self, current):
        if self._check_for_alive():
            return 
        self.p_write('CURR %s'%current)
        return

    def get_set_current(self):
        cset=self.p_query('CURR?').split('\r\n')[0]
        return float(cset)

    def get_set_volt(self):
        vset = self.p_query('VOLT?').split('\r\n')[0]
        return float(vset)

    def get_meas_current(self):
        cout=''
        while cout=='':
            cout=self.p_query('IOUT?').split('\r\n')[0]
        return float(cout)

    def get_state(self):
        state=self.p_query('OUT:STAT?').split('\r\n')[0]
        if state=='OFF':
            return False
        else:
            return True

    def set_state(self,state):
        if self._check_for_alive():
            return
        self.p_write('OUT %s'%str(int(state)))
        return

    def set_current_limit(self, limit):
        if self._check_for_alive():
            return
        if limit<=14.4:
            self.p_write('OUT:LIM:CURR %s'%str(limit))
        else:
            print('Current limit must be below 14.4 Amps')
        return

    def get_current_limit(self):
        limit=self.p_query('OUT:LIM:CURR?').split('\r\n')[0]
        return limit

    def set_voltage_limit(self, limit):
        if self._check_for_alive():
            return
        if limit<=100.0:
            self.p_write('OUT:LIM:VOLT %s'%str(limit))
        else:
            print('Voltage limit must be below 100 Volts')
        return

    def get_voltage_limit(self):
        limit=self.p_query('OUT:LIM:VOLT?').split('\r\n')[0]
        return limit

    def clear_status(self):
        if self._check_for_alive():
            return
        self.p_write('*CLS')
        return

    def supply_reset(self):
        if self._check_for_alive():
            return
        self.p_write('*RST')
        return

class XLN10014(XLN10014functions):
    def __init__(self, name="", address='COM3',enabled=True, timeout=.02, query_timeout=20):
        super(XLN10014, self).__init__(name, address, enabled, timeout, query_timeout)

    def volt_ramp(self, start_volt, stop_volt, rate, interval, parallel=False):
        self.stop_ramp=False
        self.hold=False
        if [names for names in self._get_job_names() if 'VRamp' in names]==[]:
            if parallel==True:
                thread=self._set_thread('VRamp', self._par_volt_ramp, start_volt, stop_volt, rate, interval)
                thread.start()
                return thread
            else:
                self._par_volt_ramp(start_volt, stop_volt, rate, interval)
        else:
            return

    def ramp_profile(self, state_dict, parallel=False):
        self.stop_ramp=False
        self.hold=False
        if [names for names in self._get_job_names() if 'VRamp' in names]==[]:
            if parallel==True:
                thread=self._set_thread('VRamp', self._par_ramp_profile, state_dict)
                thread.start()
                return thread
            else:
                self._par_ramp_profile(state_dict)
        else:
            return


    def _par_ramp_profile(self, state_dict):
        self.ramp_state=None
        func_dict={'ramp':self._par_volt_ramp,
                   'hold':self._par_hold
                    }

        if state_dict=={}:
            return
        else:
            pass
        for states in state_dict:
            key=[state for state in list(states.keys()) if state!='name'][0]
            if key in list(func_dict.keys()):
                if 'name' in states.keys():
                    self.ramp_state=states['name']
                else:
                    self.ramp_state=key
                args=states[key]        
                func_dict[key](*args)
            else:
                pass
        self.ramp_state='stop'
        return 
