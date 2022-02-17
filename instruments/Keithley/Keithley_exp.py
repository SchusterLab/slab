from slab.datamanagement import SlabFile
from numpy import *
import datetime
import os
import time
from slab.dsfit import fitlinear
from slab.instruments import *
from tqdm import tqdm

def take_IV(filename,curr_source, volt_meter, exp_para, zeroed_at_start=True):
    '''
    Measure simple IV curve given the parameters:
        exp_para={
        'Imax':1e-4, # unit:amp
        'Imin':-1e-4,
        'step':1e-4,
        'v_channel':1,
        'v_range':'auto',
        'integrate_time':0.05,
        'settling_time':0.05 #This is the time for current source to reach setpoint,
        'wait_time': turn off current between points to let heat dissipate. Minimum can be 0.
        }
  
    A few notes:
        1. Integrate time should be somewhere between 16.67 to 100 ms. (i.e., 1/(integer*60Hz))
        2. Settling time is rise time*5 (5*RC), where C is roughly the cabel capacitance and R of DUT.
        Measure R and C if accurate value is needed.
    '''
    Imax = exp_para['Imax']
    Imin = exp_para['Imin']
    step = exp_para['step']
    v_channel = exp_para['v_channel']
    v_range = exp_para['v_range']
    v_preamp = exp_para['v_preamp']
    integrate_time = exp_para['integrate_time']
    settling_time = exp_para['settling_time']
    wait_time = exp_para['wait_time']
    
    curr_points = list(arange(0,Imax,step))+list(arange(Imax,Imin,-1*step))+list(arange(Imin,0,step))+[0]
    print(f'Number of scan points: {len(curr_points)}')

    volt_meter.init()
    volt_meter.buffer_reset(n=len(curr_points))
    volt_meter.set_para(v_channel, v_range = v_range, integrate_time = integrate_time)
    
    curr_source.init(max(curr_points))
    curr_source.curr('on') #current output turns on here.
    time.sleep(1)

    v_points=[]
    volt_meter.beeper('off') # The code generate an error message but it's not a problem.
    for val in tqdm(curr_points):
        curr_source.set_curr(val)
        time.sleep(settling_time) # Rise time is 5*RC, where C is roughly the cabel capacitance and R of DUT. 
        reading=volt_meter.get_volt(integrate_time)/v_preamp
        if zeroed_at_start:
            if len(v_points)==0:
                v_points.append(0)
                v0=reading
            else:
                v_points.append(reading-v0)#read the data taken the latest.
        else:
            v_points.append(reading)#read the data taken the latest.
        if wait_time == 0:
            pass
        else:
            curr_source.curr('off')
            time.sleep(wait_time)
            curr_source.curr('on')

    with SlabFile(filename,'a') as f:
        f.add('v_preamp',v_preamp)
        f.append_line('I_pts', curr_points)
        f.append_line('V_pts', v_points)
        
    volt_meter.init()
    curr_source.init(max(curr_points))
    print(f'finish downloading {filename}.')
    return curr_points, v_points
      

def take_delta(filename,curr_source, curr_val, integrate_time=0.05, points_avg = 1):
    '''
    This function is only functioning when the 2182a is connected to 6221 instead
    of PC. All reading is routed through 6221 in this case. 
    '''
    if int(curr_source.queryb('SOUR:DELT:NVPResent?')):
        curr_source.init(curr_val) #Restores 622x defaults.
        curr_source.write(f'SOUR:DELT:HIGH {curr_val}') #Sets source value.
        curr_source.write(f'SOUR:DELT:COUN {points_avg}') # measure one point at a time??
        curr_source.write('SOUR:DELT:CAB ON') #Enable abort when compliance voltage is reached.
        curr_source.write(f'TRAC:POIN {points_avg}') #Set buffer to store the data points.
        curr_source.write('SOUR:DELT:ARM') #Prepare 6221/2182 for the delta measurement.
        while not curr_source.queryb('SOUR:DELT:ARM?'):
            time.sleep(0.1)
        curr_source.write('INIT:IMM') #Start measurement.
        
        reading=curr_source.queryb('SENS:DATA?')
        
        curr_source.write('SOUR:SWE:ABOR')
        
        return reading #need to add average function.
    
    else:
        print('!!!Aborted!!!: Keithley 2182 is not connected to 6221. Check connection.')
    













if __name__ == '__main__':
    im = InstrumentManager()
    cs = im['Keithley6221']
    nvm = im['Keithley2182']
    
    
    
    # --------------------------------------------- input ---------------------------------------------
    # Folder = '\data'+'\IV20180709' #Name of the folder to store the .txt data.
    Folder = '\data'+'\DC20190503' #Name of the folder to store the .txt data.
    Device = 'wms1089L2_36mK'
    expt_path = os.getcwd() + Folder
    
    # Variables
    Imin = -1e-4
    Imax = 1e-4
    step = 5e-6
    rate = 0.08333 # Voltage measuring speed in delay seconds. Best signal-to-noise ratio is when rate = 0.01667 to 0.08333 (best) seconds.
    
    volt_autorange = False #Turn on/off auto-range on the voltage meter.
    # If autorange is off, set the range below:
    vamp = 100 #voltage preamp
    vrange= 1e-2  #expected maximum voltage range before amplified.
    # ---------------------------------------------- End ---------------------------------------------
    
    
    
    #generate sweeppoints in list form
    #pt = int(2*(Imax-Imin)/step) #total number of scan points
    sp = list(np.arange(0,Imax,step))+list(np.arange(Imax,Imin,-1*step))+list(np.arange(Imin,0,step)) #list for setting up sweep points
    pt = len(sp)
    print('Number of scan points: ',pt)
    # if pt==len(sp): #check whether the steps match the range
    #     print('variable okay')
    # else:
    #     print('!step error',pt,len(sp))
    
    #system initialize
    cs.initialize(Imax,volt_comp=10)
    nvm.initialize()
    
    #system setup
    if volt_autorange:
        nvm.set_range_auto()
    else:
        nvm.set_volt_range(vrange*vamp*10)
    
    nvm.setting(channel=1, rate=rate, digit=6) #basic setting.
    nvm.buffer_reset(n=pt) #Clear the memory
    time.sleep(1)
    
    cs.curr('on')
    time.sleep(1)
    
    #Data taking
    v=[]
    prefix = Device
    fname = get_next_filename(expt_path, prefix, suffix='.h5')
    fname = os.path.join(expt_path, fname)
    
    for ii,i_value in enumerate(sp):
        cs.set_curr(i_value)
        time.sleep(0.05)
        v_value = nvm.get_volt()
        time.sleep(rate*3)#Make a pause for longer than the measurement time above.
    
        complete = nvm.ask_completion()
        while complete != 1:
            print('waiting for measurement')
            time.sleep(rate*3)  # Make a pause for longer than the measurement time above.
            complete = nvm.ask_completion()
    
        if ii == 0:
            rel = v_value
            v.append(0.0)
        else:
            buffer=v_value
            v.append((buffer-rel)/vamp)#Adjusted for voltage preamp and baseline value.
    
        with SlabFile(fname) as f:
            f.append_pt(('I_pts'), i_value)
            f.append_pt(('V_pts'), v[-1])
        time.sleep(0.01)
    
    time.sleep(1)
    # dmm.write('*WAI')
    # nvm.write('*WAI')
    cs.curr('off')
    
    #system initialize
    cs.initialize()
    nvm.initialize()
    
    #plotting
    sp = array(sp)*1e6
    v = array(v)*1e6
    plt.plot(sp,v,'bo') #plot the data
    fits = fitlinear(sp,v,showfit=True)
    print ("Resistance = ",fits[1])
    plt.ylabel('Voltage (uV)')
    plt.xlabel('Current (uA)')
    #plt.ylim(-1000, 1000)
    plt.show()
