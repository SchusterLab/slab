'''

Furnace automation code written by Andrew Oriani

REVISION 2.0: 12/2020

-Data logging, ramping, and nitrogen treatment processes are done in independent threads instead of a single state machine as before
-Thread handling is done via a seperate class now
-The XLN powersupply has it's own parallelization built into it's driver for ramping
-Improvements to query speed and the ability to query the PLC during nitrogen pulsing has been added
-Improvements to speed and parallelization has been added to the RGA
-An updated pressure server can now do intervals as low as 1.5 seconds for full instrument logging, down from 10 seconds prior
-File and log handling has been modified to automatically generate a Furnace Log file path if not present

'''

import glob
import slab
from slab import *
import numpy as np
import subprocess
import time
import os
from h5py import File, special_dtype
import gc
import yaml
from datetime import datetime, timedelta
from slab import get_next_filename, Experiment
from slab.instruments import RGA100, XLN10014
import warnings
import logging
import threading
from threading import Lock 


logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ThreadHandler():
    def __init__(self, jobs=None, max_threads=1):
        if jobs==None:
            self.jobs=[]
        else:
            self.jobs=jobs
        self.max_threads=max_threads
        self.thread_lock=Lock()

    def _check_for_alive(self, jobs=None):
        if jobs==None:
            jobs=self.jobs

        while False:
            self.thread_lock.acquire(blocking=False)

        with self.thread_lock:
            current_ident=threading.get_ident()
            alive_jobs=[]
            if jobs==[]:
                pass
            else:
                for I, job in enumerate(jobs):
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

    def _get_job_names(self, jobs=None):
        if jobs==None:
            jobs=self.jobs
        
        names=[]
        if jobs==[]:
            pass
        else:
            for job in jobs:
                if job.is_alive():
                    names.append(job._name)
                else:
                    pass
        return names

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

class vacuum(ThreadHandler):
    def __init__(self, syst_config='S:\\_Furnace\\Automation Scripts\\system_config.yml'):

        super(vacuum, self).__init__(jobs=None, max_threads=4)

        try:
            self.im=InstrumentManager()
        except: 
            raise Exception('Unable to open Instrument Manager Server')

        try:
            self.AG850 = self.im.AG850
        except:
            raise Exception('Unable to connect to turbo controller')
        try:
            self.P_monitor = self.im.pressure_gauge
        except:
            raise Exception('Unable to connect to pressure gauge')
        try:
            self.PID = self.im.PID
        except:
            raise Exception('Unable to connect to PID heaters')
        try:
            self.PLC = self.im.PLC
        except:
            raise Exception('Unable to connect to PLC')
        self.config = syst_config

        with open(self.config, 'r') as f:
            self.v_config = yaml.full_load(f)['PLC']['VALVES']

        
        self.VAC_jobs=[]
        self.VAC_lock=Lock()

        self.get_valves()
        self.valves = {}
        for I, chs in enumerate(self.v_config):
            self.valves[chs] = {'CHANNEL': self.v_config[chs]['CHANNEL'], 'STATE': self.curr_pos[I]}

        # self.PID.load_config(syst_config)

    def load_p_monitor_config(self):
        if self._check_for_alive(jobs=self.VAC_jobs):
            return 
        print('System Config: %s' % (self.config))
        try:
            self.P_monitor.load_config(self.config)
            print('________________________________')
            logging.info('Pressure Monitor Parameters set.')

        except:
            logging.error('Unable to Load Pressure Config')

    def load_PID_config(self):
        print('System Config: %s' % (self.config))
        try:
            self.PID.load_config(self.config)
            print('________________________________')
            logging.info('PID Heater Parameters set.')
        except:
            logging.error('Unable to Load PID Config')

    def load_turbo_config(self):
        if self._check_for_alive(jobs=self.VAC_jobs):
            return 
        logging.info('System Config: %s' % (self.config))
        try:
            self.AG850.load_config(self.config)
            print('________________________________')
            logging.info('Turbo Parameters set.')
        except:
            logging.error('Unable to Load Turbo Config')

    def full_start(self, heaters=False):
        if self._check_for_alive(jobs=self.VAC_jobs):
            return 
        pump_stat = self.AG850.get_status()
        if pump_stat == 'stop':
            self.set_open('ROUGHING')
            self.set_open('RGA')
            self.set_close('PULSE')
            self.set_valves()
            print('________________________________')
            logging.info('Valves set, starting backing pump')
            self.AG850.set_purge(state=1)
            timeout = 3600.00
            start_time = time.time()
            elapsed_time = time.time() - start_time
            while self.P_monitor.set_point_status()[4] == 0 and elapsed_time <= timeout:
                elapsed_time = time.time() - start_time
                pass
            if elapsed_time >= timeout:
                logging.error('Process timed out, unable to reach roughing pressure')
                return
            else:
                pass
            self.set_open('BACKING')
            self.set_open('GATE')
            self.set_valves()
            self.AG850.start()
            print('________________________________')
            logging.info('Turbo Started')
            while self.AG850.get_freq() >= 60.0:
                pass
            while self.AG850.get_freq() <= 100.00:
                pass
            self.set_close('ROUGHING')
            self.set_valves()
            print('________________________________')
            logging.info('Initial Pumping Started')

            if heaters == True:
                while self.P_monitor.set_point_status()[5] == 0:
                    pass
                self.PID.start_all()
                print('________________________________')
                logging.info('Bakeout Heaters Turned On')
        else:
            logging.error('Turbo pump must be in "STOP" condition')

    def _par_nitrogen_backfill(self, f_time):
        f_time=int((f_time-(f_time-60)%60)/60)
        if self._check_for_alive(jobs=self.VAC_jobs):
            return 
        self.set_close('RGA')
        self.set_valves()
        self.set_close('GATE')
        self.set_open('ROUGHING')
        self.set_valves()
        if self.PLC.get_state(5)[1]==False:
            self.PLC.set_pulse_time(f_time)
            self.PLC.pulse(self.valves['PULSE']['CHANNEL'])
            et=0
            start_time=time.time()
            while et<=f_time*60 or not self.P_monitor.set_point_status()[5]:
                et=time.time()-start_time
                pass
            self.set_open('GATE')
            self.set_close('ROUGHING')
            self.set_valves()
            while not self.P_monitor.set_point_status()[0]:
                pass
            #TODO: need a small delay to let things settle down otherwise it triggers an interlock error for opening the RGA
            time.sleep(.25)
            self.set_open('RGA')
            self.set_valves()


    def set_open(self, key):
        if self._check_for_alive(jobs=self.VAC_jobs):
            return 
        self.valves[key]['STATE'] = 1

    def set_close(self, key):
        if self._check_for_alive(jobs=self.VAC_jobs):
            return 
        self.valves[key]['STATE'] = 0

    def set_valves(self):
        if self._check_for_alive(jobs=self.VAC_jobs):
            return 
        new_valve_pos = [0] * len(self.curr_pos)
        for key in iter(self.valves.keys()):
            new_valve_pos[self.valves[key]['CHANNEL'] - 1] = self.valves[key]['STATE']
        set_inds = [index for index, elem in enumerate(new_valve_pos) if elem != self.curr_pos[index]]
        for inds in set_inds:
            self.PLC.set_valve(inds + 1, new_valve_pos[inds])

        self.get_valves()

    def get_valves(self):
        while False: 
            self.VAC_lock.acquire(blocking=False)
        with self.VAC_lock:
            try:
                v_state = self.PLC.get_all()
                self.curr_pos = v_state
                return self.curr_pos
            except:
                raise Exception('ERROR occured in PLC, reset PLC')

    def cal_CTR(self):
        if self._check_for_alive(jobs=self.VAC_jobs):
            return 
        pres = self.P_monitor.p_read(1)[0]
        if pres < 2E-6:
            self.P_monitor.cal_ctr()
            print('________________________________')
            logging.info('CTR calibrated')
        else:
            print('________________________________')
            logging.info('Pressure must be below 2E-6 for accurate CTR calibration')

    def heaters_on(self, state=True, load_config=False):
        if load_config == True:
            self.load_PID_config()
        else:
            pass
        if state == True:
            self.PID.start_all()
            print('________________________________')
            logging.info('Bakeout Heaters On')
        elif state == False:
            self.PID.stop_all()
            print('________________________________')
            logging.info('Bakeout Heaters Off')

    def full_stop(self):
        if self._check_for_alive(jobs=self.VAC_jobs):
            return 
        pump_stat = self.AG850.get_status()
        self.PID.stop_all()
        self.get_valves()
        self.set_close('GATE')
        self.set_close('RGA')
        self.set_close('ROUGHING')
        self.set_valves()
        if pump_stat != 'stop':
            if pump_stat == 'normal':
                self.AG850.stop()
                print('Turbo Shut Off')
            elif pump_stat == 'starting':
                self.AG850.stop()
            print('________________________________')
            logging.info('Turbo Shut Off')
        else:
            pass
        self.set_close('BACKING')
        self.set_valves()
        self.AG850.set_purge(0)
        print('________________________________')
        logging.info('Backing Pump Shut Off')

    def get_pressure(self, ch):
        return self.P_monitor.p_read(ch)[0]

    def get_PID_temps(self):
        return self.PID.get_temp(0)

    def reset_plc(self):
        if self._check_for_alive(jobs=self.VAC_jobs):
            return
        self.PLC.reset()

class FurnaceSetup(vacuum):
    def __init__(self, config_path=None, system_config_path=None, kill_server=False, p_server=False):
        local_path=os.getcwd()
        if config_path==None:
            config_path=local_path+'\\furnace_config.yml'
        if system_config_path==None:
            system_config_path=local_path+'\\system_config.yml'

        try:
            super(FurnaceSetup, self).__init__(system_config_path)
        except:
            raise Exception('Unable to load vacuum interface, check system_config.yml is in same path or in specified path')

        try:
            with open(config_path, 'r') as f:
                self.furnace_config = yaml.full_load(f)['Furnace']
            with open(config_path, 'r') as f:
                self.RGA_config=yaml.full_load(f)['RGA']
            with open(config_path, 'r') as f:
                self.nitrogen_config = yaml.full_load(f)['Nitrogen_Doping']
        except:
            raise Exception('Unable to load furnace config parameters, check config param keys')
        try:
            with open(system_config_path, 'r') as f:
                self.supply_config=yaml.full_load(f)['SUPPLY']
        except:
            raise Exception('Unable to load supply config parameters, check config param keys')

        print('Furnace Configuration:')
        print('_________________________')
        for keys in iter(self.furnace_config.keys()):
            print(keys.replace("_", " ")+': '+str(self.furnace_config[keys]))

        if self.nitrogen_config['nitrogen_step']==True:
            for keys in iter(self.nitrogen_config.keys()):
                print(keys.replace("_", " ")+': '+str(self.nitrogen_config[keys]))
        print('_________________________')

        #RGA and logging parameters
        self.RGA_port=self.RGA_config['port']

        #supply safety params
        self.max_current = self.supply_config['max_current']
        self.max_volts = self.supply_config['max_volts']
        self.min_volts = self.supply_config['min_volts']
        self.supply_port = self.supply_config['port']

        #funace ramping and logging parameters
        self.ramp_up_rate = self.furnace_config['Ramp_up_rate']
        self.ramp_down_rate = self.furnace_config['Ramp_down_rate']
        self.P_hold = np.float(self.furnace_config['P_hold'])
        self.Hold_v = self.furnace_config['Hold_v']
        self.Hold_time = self.furnace_config['Hold_time']
        self.log_interval = self.furnace_config['logging_interval']
        self.PH_time=self.furnace_config['Pre_heat_time']
        self.P_start=np.float(self.furnace_config['P_start'])

        #nitrogen doping parameters
        self.nitrogen_state=self.nitrogen_config['nitrogen_step']
        self.nitrogen_start_time=self.nitrogen_config['nitrogen_delay_time']
        self.nitrogen_hold_time=self.nitrogen_config['nitrogen_hold_time']
        self.nitrogen_end_time=self.nitrogen_start_time+self.nitrogen_hold_time

        if self.nitrogen_state==True:
            if self.nitrogen_end_time>self.Hold_time:
                
                logging.error('Nitrogen start and end times must be less than the total hold time. Exiting automation')
                exit()

        if self.min_volts<12.0:
               
            logging.error('Minimum voltage lower than allowable 12 volts, exiting automation')
            exit()

        if self.max_volts<=self.min_volts:
               
            logging.error('Maximum voltage less than minimum voltage, exiting automation')
            exit()

        if self.Hold_v<self.min_volts or self.Hold_v>self.max_volts:
               
            logging.error('Holding voltage not valid, must be less than max voltage and greater than min voltage')
            print('Exiting automation')
            exit()

        if self.P_start>=self.P_hold:
               
            logging.error('Degas pressure hold point must be greater than ramp restart pressure')
            print('Exiting automation')
            exit()

        self.kill_s = kill_server

        #############################################################
        # Server code, comment out if just debugging the automation #
        #############################################################

        self.PS = Server()
        self.PID = self.PS.get_PID()
        self.p_log = self.PS.get_log()

        self.pp_keys = self.PS.pp_keys()

        #############################################################
        #                    End of server code                     #
        #############################################################

        try:
            self.im = InstrumentManager()
        except:
            raise Exception('ERROR in contacting instrument manager server')

        try:
            self.xln = XLN10014(address=self.supply_port)
        except:
            raise Exception('ERROR in contacting XLN power supply')
        
        self.max_threads=4
        self.num_threads=0
        self.log_lock=Lock()
        self.thread_lock=Lock()
        self.jobs=[]

    def _start_furnace_server(self):
        local_time = time.time()
        self.st = datetime.fromtimestamp(local_time).strftime('%Y-%m-%d %H:%M:%S').split(' ')[0]

        prefix = self.st.split(' ')[0] + '_Furnace_log'
        
        self.data_path=self.PS.data_path

        self.flog = Experiment(self.data_path, prefix, liveplot_enabled=False)
        self.set_flog()

        print('_________________________')
        print('Monitoring Furnace at:')
        print('_________________________')
        print(self.flog.fname)
        print('_________________________')

    def set_flog(self):
        with self.flog.datafile() as f:
            f.attrs.create('start_time', self.st, dtype='S20')
            v_string =special_dtype(vlen=bytes)
            f.attrs.create('Pressure_log', self.p_log, dtype=v_string)
            for keys in iter(self.furnace_config.keys()):
                f.attrs[keys] = self.furnace_config[keys]

    def update_log(self):
        self.err_vals=None
        while False:
            self.log_lock.acquire(blocking=False)
        with self.log_lock:
            self.fil_state=self.PS.read('fil_state')
            if self.fil_state==0:
                self.furnace_shut_down()
                self.stop_logging=True
                return
            if self.PS.server_state()==False:
                self.furnace_shut_down()
                self.stop_logging=True
                return
            p_tot=self.PS.read('p_tot')
            curr=self.xln.get_meas_current()
            volts=self.xln.get_meas_volt()
            if self.PS.read('p_tot_compound')>=self.P_hold:
                self.xln.hold=True
            else:
                self.xln.hold=False
            with self.flog.datafile() as f:
                f.append('tpts', time.time())
                f.append('RGA_pres', p_tot)
                f.append('curr', float(curr))
                f.append('volts', float(volts))
                for keys in self.pp_keys:
                    f.append_pt(keys, self.PS.read(keys))


    def furnace_shut_down(self):
        self.xln.set_state(0)
        self.xln.set_volt(0)
        self.xln.set_current(0)
        logging.info('Power supply shut down')
        if self.PS.server_state==True:
            if self.fil_state==1:
                if self.kill_s==True:
                    self.PS.kill()
                    time.sleep(5)
                    logging.info('Pressure Server Shut Down')
                    self.RGA_fil_off()
                else:
                    pass
            elif self.fil_state==0:
                self.PS.kill()  
                logging.info('Pressure Server Shut Down')
                exit()
        elif self.PS.server_state()==False:
            time.sleep(5)
            self.RGA_fil_off()
            exit()
    
    def RGA_fil_off(self):
        try:
                
            logging.info('Shutting Down RGA Filament')
            rga = RGA100(address=self.RGA_port)
            rga.set_filament_output(state=False)
            rga.ser.close()
        except:
            logging.warning('WARNING: Couldn\'t shut RGA filament off')        

    def _par_logger(self, interval=10):
        start_time=time.time()
        while not self.stop_logging:
            self.update_log()
            time.sleep(interval-(time.time()-start_time)%interval)

class Server():
    def __init__(self, config_path=None, launch_at_start=True):
        if config_path==None:
            local_path=os.getcwd()
            config_path=local_path+'\\furnace_config.yml'
        try:
            with open(config_path, 'r') as f:
                self.RGA_config=yaml.full_load(f)['RGA']
            self.data_path = self.RGA_config['data_path']
            if self.data_path==None:
                log_dir='\\Furnace Logs'
                self.data_path=os.path.dirname(config_path)+log_dir
                if not os.path.exists(self.data_path):
                    os.makedirs(self.data_path)
                self.data_path=self.data_path+'\\'
            
        except:
            print('Unable to find or open server, check config_path and config.')
        if launch_at_start:
            self.log, self.PID = self.pressure_server(self.data_path)
        else:
            pass

    def get_log(self):
        return self.log

    def get_PID(self):
        return self.PID

    def last_log_fname(self, path, prefix):
        dirlist = glob.glob(path + '*' + prefix + '*')
        dsort = dirlist.sort()
        i = 1
        max_iter = 100
        if dsort == None and len(dirlist) == 0:
            date = prefix.split('_')[0]
            while dsort == None and len(dirlist) == 0 and i < max_iter:
                prefix = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=i)).strftime('%Y-%m-%d') + '_RGA_log'
                dirlist = glob.glob(path + '*' + prefix + '*')
                dirlist.sort()
                i += 1
            if i == max_iter:
                return None
            else:
                f_name = os.path.split(dirlist[-1])[-1]
        else:
            f_name = os.path.split(dirlist[-1])[-1]
        return f_name

    def pressure_server(self, data_path):
        next_fname = self.next_log_fname(data_path)
        num = next_fname.split('_')[0]
        prefix = next_fname.split(num + '_')[1]
        last_fname = self.last_log_fname(data_path, prefix)
        if last_fname!=None:
            log_time = self.swmr_read(data_path + last_fname, 'tpts', -1)
            et = time.time() - log_time
            if et > 30:
                self.p_log, self.p_log_ID = self.start_pressure_server(data_path)
            else:
                try:
                    PID = self.swmr_attr(data_path + last_fname, 'LOG_PID')
                    p_name = self.process_name(PID)
                    if p_name == 'python.exe':
                        self.p_log = last_fname
                        self.p_log_ID = PID
                        print('Server Running:')
                        print('_________________')
                        print('Server Log: %s' % self.p_log)
                        print('Server Process ID %i' % self.p_log_ID)
                        print('_________________')
                    else:
                        self.p_log, self.p_log_ID = self.start_pressure_server(data_path)
                except:
                    print('ERROR in determining PID from log')
                    return
        else:
            self.p_log, self.p_log_ID = self.start_pressure_server(data_path)
        return self.p_log, self.p_log_ID

    def server_state(self):
        p_name = self.process_name(self.PID)
        if p_name == 'python.exe':
            state = True
        else:
            state = False
        return state

    def read(self, key):
        data = self.swmr_read(self.data_path + self.log, key, -1)
        return data

    def attr_read(self, key):
        data=self.swmr_attr(self.data_path+self.log, key)
        return data

    def pp_keys(self):
        data_file = File(self.data_path + self.log, 'r', libver='latest', swmr=True)
        try:
            keys = [key for key in data_file.keys()]
            pp_key_vals = []
            for vals in keys:
                if vals != 'tpts' and vals != 'p_tot':
                    pp_key_vals.append(vals)
                else:
                    pass
            data_file.close()
            return pp_key_vals
        except:
            self.exit_handler()
            raise Exception('ERROR')

    def start_pressure_server(self, data_path, suffix='h5'):
        '''
        Starts an external pressure datalogger using shell command
        '''
        try:
            conf = subprocess.call('start python -i pressure_server.py', shell=True)
            if conf != 0:
                raise Exception('Error in shell script, check instrument')
            else:
                pass

            new_log_fname = self.next_log_fname(data_path)

            i = 0
            max_iter = 60
            print('Waiting for server to start...')

            while os.path.isfile(data_path + new_log_fname + '.' + suffix) == False and i < max_iter:
                time.sleep(1)
                i += 1
            if i >= max_iter:
                raise Exception('ERROR: Timed out waiting for log to be created')
            else:
                pass
            print('_________________')

            try:
                time.sleep(2)
                PID = self.swmr_attr(data_path + new_log_fname + '.' + suffix, 'LOG_PID')
                print('New log created: %s' % new_log_fname + '.' + suffix)
                print('Server Process ID: %i' % PID)
                return new_log_fname + '.' + suffix, PID
            except:
                raise Exception('ERROR: Server has no PID')

        except RuntimeError:
            print('Error occurred in server initialization')

    def exit_handler(self):
        '''
        Function that finds any instances of open H5 files and safely
        closes them in event of normal interpreter shutdown or error.
        '''
        for obj in gc.get_objects():  # Browse through ALL objects
            if isinstance(obj, File):  # Just HDF5 files
                try:
                    obj.close()
                except:
                    pass  # Was already closed

    def latest_file(self, filepath):
        list_files = glob.glob(filepath + '*')
        latest_file = max(list_files, key=os.path.getctime)
        data_file = latest_file.split('\\')[-1]
        return data_file

    def swmr_read(self, file_path, key, index):
        data_file = File(file_path, 'r', libver='latest', swmr=True)
        try:
            data = np.array(data_file[key])[index]
            data_file.close()
            return data
        except:
            self.exit_handler()

    def swmr_attr(self, file_path, key):
        data_file = File(file_path, 'r', libver='latest', swmr=True)
        try:
            attr_val = data_file.attrs[key]
            data_file.close()
            return attr_val
        except:
            self.exit_handler()
            print('ERROR')

    def kill(self):
        self.kill_server(self.PID)

    def process_name(self, PID):
        '''
        Finds process name for a given process ID using shell commands and parsing return.
        '''
        out = subprocess.check_output('tasklist /fi "pid eq {0}"'.format(PID), shell=True).decode("utf-8")
        if out.split(':')[0] == 'INFO':
            return 'no task'
        else:
            return out.split('\r\n')[3].split(' ')[0]

    def kill_server(self, PID):
        '''
        safely kills the data server via shell. Checks if process is a valid python executable.
        '''
        try:
            p_name = self.process_name(PID)
            if p_name == 'python.exe':
                # needs /t flag for safe process shutdown
                subprocess.Popen('taskkill /PID {0} /t'.format(PID), shell=True)
            elif p_name == 'no task':
                print('No process currently running with that ID')
            else:
                print('Not a valid python interpreter, cannot close process')
        except:
            print('ERROR Occurred in killing process')

    def _get_next_filename(self, datapath,prefix,suffix=''):
        ii = self._next_file_index(datapath, prefix)
        return "%05d_" % (ii) + prefix +suffix

    def _next_file_index(self, datapath,prefix=''):
        """Searches directories for files of the form *_prefix* and returns next number
            in the series"""

        dirlist=glob.glob(os.path.join(datapath,'*_'+prefix+'*'))
        dirlist.sort()
        try:
            ii=int(os.path.split(dirlist[-1])[-1].split('_')[0])+1
        except:
            ii=0
        return ii

    def next_log_fname(self, path):
        local_time = time.time()
        date = datetime.fromtimestamp(local_time).strftime('%Y-%m-%d %H:%M:%S').split(' ')[0]
        prefix = date.split(' ')[0] + '_RGA_log'
        next_file = self._get_next_filename(path, prefix)
        return next_file

class FurnaceAutomation(FurnaceSetup):
        def __init__(self, config_path=None, system_config_path=None, kill_server=False, p_server=False):
            super(FurnaceAutomation, self).__init__(config_path, system_config_path, kill_server, p_server)
            self.ramp_interval=2
            self.nitrogen_on=False

        def start_log(self, parallel=False):

            self.stop_logging=False
            if [names for names in self._get_job_names() if 'Logger' in names]==[]:
                self._start_furnace_server()
                if parallel==True:
                    self._thread_check()
                    thread=self._set_thread('Logger', self._par_logger, self.log_interval)
                    thread.start()
                    if self.jobs[-1].is_alive():
                        logging.info('Furnace logger running')
                        return thread
                    else:
                        raise Exception('Unable to start the logging thread')
                else:
                    self._par_logger(self.log_interval)
            return

        def start_ramping(self):
            if not self.xln.get_state() or self.xln.get_meas_volt()<self.min_volts:
                self.start_supply()

            state_dict=[{'hold':(self.PH_time,), 'name':'pre-heat'},
                        {'ramp':(self.min_volts, self.Hold_v, self.ramp_up_rate, self.ramp_interval), 'name':'ramp-up'},
                        {'hold':(self.Hold_time,), 'name':'soak'},
                        {'ramp':(self.Hold_v, self.min_volts, self.ramp_down_rate, self.ramp_interval), 'name':'ramp-down'}
                        ]
            
            thread=self.xln.ramp_profile(state_dict=state_dict, parallel=True)
            self.jobs.append(self.xln.jobs[-1])

            if thread.is_alive():
                 
                logging.info('Ramping started, in state: %s'%self.xln.ramp_state)
                return thread
            else:
                raise Exception('Unable to start ramping, check instruments...')

        def nitrogen_backfill(self, parallel=False, delay=False):
            self.stop_nitrogen=False
            self.nitrogen_on=False
            if delay==False:
                delay=0
            else:
                delay=self.nitrogen_start_time
            
            if [names for names in self._get_job_names() if 'NitrogenFill' in names]==[]:
                if parallel==True:
                    self._thread_check()
                    thread=self._set_thread('NitrogenFill', self._par_delayed_nitrogen_backfill, delay, self.nitrogen_hold_time, 'soak')
                    self.VAC_jobs.append(thread)
                    thread.start()
                    if self.jobs[-1].is_alive():
                         
                        logging.info('Nitrogen armed and holding')
                    else:
                        raise Exception('Unable to start the nitrogen thread')

                    return thread
                else:
                    self._par_delayed_nitrogen_backfill(delay, self.nitrogen_hold_time, 'soak')

        def _par_delayed_nitrogen_backfill(self, delay, fill_time, start_cond='soak',):
            if [names for names in self._get_job_names() if 'VRamp' in names]==[]:
                return 
            while self.xln.ramp_state!=start_cond and not self.stop_nitrogen:
                pass
            et=0
            start_time=time.time()
            while et<=delay and not self.stop_nitrogen:
                et=time.time()-start_time
                pass
            if self.stop_nitrogen:
                return 
            else:
                self.nitrogen_on=True
                self._par_nitrogen_backfill(fill_time)
                self.nitrogen_on=False

        def start_supply(self):
            try:
                self.xln.get_id()
            except:
                raise Exception('ERROR in communicating with power supply')

            self.xln.set_state(False)
            self.xln.set_voltage_limit(self.max_volts)
            self.xln.set_volt(self.min_volts)
            self.xln.set_current(self.max_current)
            self.xln.set_state(True)
            i=0
            max_iter=20
            while self.xln.get_meas_volt()<11.9 and i<=max_iter:
                self.xln.set_state(False)
                time.sleep(1)
                self.xln.set_state(True)
                i+=1
            if i>=max_iter:
                self.xln.set_state(False)
                 
                logging.error('Power supply unable to reach starting voltage, exiting automation')
                exit()
            else:
                pass
             
            logging.warning('Power supply on, furnace started')


        def run_full_process(self, new_server=True, monitor=True, stop_logger=True):

            if new_server==True:
                log_check=[threading.enumerate()[I] for I, jobs in enumerate(threading.enumerate()) if jobs.name=='Logger']
                if log_check!=[]:
                    self.stop_logging=True
                    while log_check[0].is_alive():
                        pass

            job_names=self._get_job_names(threading.enumerate())
            automation_names=['VRamp', 'Logger', 'NitrogenFill']
            running_jobs=[an for an in automation_names if an in ' '.join(job_names)]
            if running_jobs==[]:
                log_thread=self.start_log(parallel=True)
                ramp_thread=self.start_ramping()
                if self.nitrogen_state==True:
                    self.nitrogen_backfill(parallel=True, delay=True)
            elif len(running_jobs)==1 and [name for name in running_jobs if 'Logger' in name]!=[]:
                ramp_thread=self.start_ramping()
                if self.nitrogen_state==True:
                    self.nitrogen_backfill(parallel=True, delay=True)
            else: 
                non_log_jobs=[name for name in running_jobs if not 'Logger' in name]
                logging.warning('WARNING: %s operations are currently running, cannot start process'%(", ".join(non_log_jobs)))

            if monitor==True:
                r_state=self.xln.ramp_state
                hold_state=self.xln.hold
                nitrogen_running=self.nitrogen_on
                while ramp_thread.is_alive():
                    if r_state==self.xln.ramp_state and hold_state==self.xln.hold and nitrogen_running==self.nitrogen_on:
                        pass
                    else:
                        if r_state!=self.xln.ramp_state:
                            r_state=self.xln.ramp_state
                            logging.info('Current State: %s'%r_state)
                        elif hold_state!=self.xln.hold and self.xln.ramp_state!='soak' and self.xln.ramp_state!='pre-heat':
                            hold_state=self.xln.hold
                            if self.xln.hold==True:
                                logging.info('Holding voltage, overpressure')
                            else:
                                logging.info('Pressure stabilized, restarting')
                        elif nitrogen_running!=self.nitrogen_on:
                            nitrogen_running=self.nitrogen_on
                            if self.nitrogen_on==True:
                                logging.info('Nitrogen treatment running')
                            else:
                                logging.info('Nitrogen treatment finished')
                logging.info('Process finished!')
                self.furnace_shut_down()
                if stop_logger==True:
                    self.stop_logging=True
                    while log_thread.is_alive():
                        pass
                    logging.info('Logger thread shut down, furnace logging file closed')

if __name__=="__main__":
    '''
    Run file and answer prompts to run full automation. Best to run in interactive window
    '''

    cwd=os.getcwd()

    furnace_config="furnace_config_N2"
    # furnace_config="furnace_config_O2_bake"

    furnace=FurnaceAutomation(config_path=cwd+'\\'+furnace_config+'.yml', system_config_path=None)
    user_input=input('Would you like to start furnace automation? [Y/N]:')
    if user_input[0].lower()=='y':
        furnace.run_full_process()
    elif user_input[0].lower()=='n':
        logging.info('Exiting automation selection.')
    else:
        raise TypeError('User input must be Y (yes) or N (no).')


