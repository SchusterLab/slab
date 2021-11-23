'''
Author: Andrew Oriani

RGA Pressure server.

The following script should only be called through a separate python interpreter.
It's basic functionality is to start a pressure logger that can be asynchronously read by an external python
interpreter. This is done using h5py's SWMR functionality. Due to the fact that the SWMR requires the h5 file be open
for the extent of the logging, care should be taken to stop the logger in a safe manner, either by killing the process
through a normal closing of the external window, or using the shell based kill_server function also provided in the
code. Failure to do so will result in a corrupted dataset.

'''


from slab import Experiment
# from slab import *
import numpy as np
import time
from datetime import datetime
import os
import gc
import atexit
import yaml
import h5py
from h5py import File
from slab.instruments import RGA100, InstrumentManager
from sys import exit
from signal import signal, CTRL_C_EVENT

#exit handler searches for open instances of h5 files, using atexit, the function will run and safely close
#all files during normal interpreter shutdown.

def exit_handler():
    for obj in gc.get_objects():  # Browse through ALL objects
        if isinstance(obj, File):  # Just HDF5 files
            try:
                print('hey')
                time.sleep(5)
                obj.close()
            except:
                pass  # Was already closed
    exit()

def rga_shutdown():
    rga.set_filament_output(state=False)


def set_close(path):
    f = File(path, 'w')
    f.attrs['state'] = 'closed'

atexit.register(exit_handler)
#get process ID, to be stored in H5 file as attribute to allow for external server ID
PID=os.getpid()

#RGA parameter keys
RGA_keys=['fil_current','e_energy','ion_energy','foc_volts','scan_rate','units','t_wait']
config_path=os.getcwd()+'\\'
config_name='furnace_config.yml'
system_config='system_config.yml'

with open(config_path+config_name, 'r') as f:
    rga_config=yaml.full_load(f)['RGA']

try:
    with open(config_path+system_config, 'r') as f:
        rga_PLC_chan=yaml.full_load(f)['PLC']['VALVES']['RGA']['CHANNEL']
except:
    print('Unable to open RGA PLC channel for monitoring, shutting down server.')
    time.sleep(10)

port=rga_config['port']
data_path=rga_config['data_path']

if data_path==None:
    log_dir='\\Furnace Logs'
    data_path=os.path.dirname(config_path)+log_dir
    if not os.path.exists(data_path):
        os.makedirs(data_path)

try:
    im=InstrumentManager()
except:
    print('Problem initializing instrument manager')
    time.sleep(10)
    exit()


try:
    p_gauge=im.pressure_gauge
    print('_________Pressure Init_________')
    print('_______________________________')
except:
    print('Problem initializing pressure gauges')
    time.sleep(10)
    exit()

try:
    t_gauge=im.TempScan
    print('___________Temp Init___________')
    print('_______________________________')
except:
    print('Problem initializing temperature scanner')
    time.sleep(10)
    exit()

try:
    PLC=im.PLC
    print('___________PLC Init____________')
    print('_______________________________')
except:
    print('Problem initializing PLC')
    time.sleep(10)
    exit()

try:
    #check if Instrument manager Server is Running
    rga=RGA100(address=port)

    #if nameserver is unavailable uncomment below, comment above

    #im = InstrumentManager(config_path='S:\_Furnace\Furnace scripts\instrument.cfg', server=False)
except:
    print('Problem initializing RGA')
    time.sleep(10)
    exit()

try:
    #check if RGA is connected
    rga.get_id()
except:
    print('RGA not connected, connect device and restart server.')
    time.sleep(10)
    exit()


local_time=time.time()
date = datetime.fromtimestamp(local_time).strftime('%Y-%m-%d %H:%M:%S').split(' ')[0]

prefix=date.split(' ')[0]+'_RGA_log'

expt=Experiment(path=data_path, prefix=prefix, liveplot_enabled=False)

print('__________Setting RGA__________')
print('_______________________________')
#initialize ionizer head
try:
    rga.ionizer_set(rga_config['e_energy'], rga_config['ion_energy'], rga_config['foc_volts'], rga_config['fil_current'], True)
except:
    print('Error initializing RGA, error status is: %s'%str(rga.error_status))
    print('Shutting down server in 30 seconds....')
    time.sleep(30)
    exit()

filament_on=True

#set scan rate
rga.set_scan_rate(rga_config['scan_rate'])

pp_mass=rga_config['scan_masses']

masses=[]
species=[]
spec_str=''
for keys in iter(pp_mass.keys()):
    masses.append(pp_mass[keys])
    species.append(keys)
    spec_str+=keys+', '

start_time=time.time()
st = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
units=rga_config['units']

print('_____________________________')
print('_________Gas Species_________')
print('_____________________________')
print(spec_str[:-2])
print('_____________________________')
print("___Monitoring pressure in____ ")
print(expt.fname)
print('_____________________________')
print('_____Pressure read start_____')
print('_____________________________')




# atexit.register(exit_handler)
# signal.signal(signal.SIGTERM, exit_handler)
# signal.signal(signal.SIGINT, exit_handler)


fil_state=rga.get_filament_state()[0]
with expt.datafile(swmr=True) as f:
    f.attrs.create('start_time', st, dtype='S20')
    f.attrs.create('units', units, dtype='S10')
    f.attrs.create('LOG_PID', PID, dtype=np.integer)

    tpts = f.create_dataset('tpts', shape=(0,), maxshape=(None,), dtype='float64')
    RGA_pres = f.create_dataset('p_tot', shape=(0,), maxshape=(None,), dtype='float64')
    c_pres=f.create_dataset('p_tot_compound', shape=(0,), maxshape=(None,), dtype='float64')
    m_pres=f.create_dataset('p_tot_mano', shape=(0,), maxshape=(None,), dtype='float64')
    temps=f.create_dataset('cav_temp', shape=(0,), maxshape=(None,), dtype='float64')

    fil_state=f.create_dataset('fil_state', shape=(0,), maxshape=(None,), dtype=np.integer)
    pp_dsets=[]
    for names in species:
        pp_dsets.append(f.create_dataset(names, shape=(0,), maxshape=(None,), dtype='float64'))

    f.swmr_mode=True

    logging_interval=rga_config['interval']

    start_time=time.time()
    while True:
        et=time.time()-start_time
        try:
            p_thread=rga.get_pressure(units=units, parallel=True)
            mm_thread=rga.multi_mass_scan(mass_vals=masses, units=units, parallel=True)
            pres_comp=p_gauge.p_read(1)[0]
            pres_mono=p_gauge.p_read(2)[0]
            temp=t_gauge.get_temp(1, sens='TC', t_type='K')[0]
            RGA_valve_state=PLC.get_state(rga_PLC_chan)[1]
        except:
            print('ERROR in logging has Occurred, safely exiting program...closing log')
            print('Window will close automatically in 10 seconds')
            time.sleep(30)
            exit()

        f.append_dset_pt(tpts, time.time())
        f.append_dset_pt(c_pres, pres_comp)
        f.append_dset_pt(m_pres, pres_mono)
        f.append_dset_pt(temps, temp)


        if RGA_valve_state==False and filament_on==True:
            rga.set_filament_off(parallel=True)
            filament_on=False
        elif RGA_valve_state==True and filament_on==False:
            rga.set_filament_on(parallel=True)
            filament_on=True

        if pres_comp > 2E-4 and RGA_valve_state==True and filament_on==True:
            rga.set_filament_output(state=False)
            f.append_dset_pt(fil_state, 0)
            print('Maximum pressure exceeded, shutting off filament')
            print('Window will automatically close in 30 seconds')
            time.sleep(30)
            exit()
        else :
            f.append_dset_pt(fil_state, 1)

        while mm_thread.is_alive() or p_thread.is_alive():
            pass

        if rga.comm_error!=[]:
            raise Exception(rga.comm_error[-1])
        elif rga.MM_scan_vals!=None and rga.P_scan_val!=None:
            pass
        # else:
        #     raise Exception('ERROR in RGA pressure values, failed to obtain correct values...')

        f.append_dset_pt(RGA_pres, rga.P_scan_val)

        for dset, pressures in zip(pp_dsets, rga.MM_scan_vals):
            f.append_dset_pt(dset, pressures)
        
        print('RGA: %.3e, CIG: %.3e, Mano: %.3e, Units: %s, Temp: %.3f C, %.3f sec' % (rga.P_scan_val, pres_comp, pres_mono, units, temp, et))
        time.sleep(logging_interval-((time.time() - start_time) % logging_interval))



