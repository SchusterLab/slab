__author__ = 'Nelson'

from slab.experiments.ExpLib.SequentialExperiment import *
from slab.experiments.General.run_seq_experiment import *
from numpy import delete
from slab import *
import os
import json


datapath = os.getcwd() + '\data'
config_file = os.path.join(datapath, "..\\config" + ".json")
with open(config_file, 'r') as fid:
        cfg_str = fid.read()

cfg = AttrDict(json.loads(cfg_str))

def get_data_filename(prefix):
    return  os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))


def multimode_pulse_calibration(seq_exp, mode):
    # pulse_calibration(seq_exp)
    seq_exp.run('multimode_calibrate_offset',{'exp':'multimode_rabi','dc_offset_guess':0,'mode':mode,'update_config':True})


def multimode_dc_offset_calibration(seq_exp, mode):
    # pulse_calibration(seq_exp)
    seq_exp.run('multimode_calibrate_offset',{'exp':'short_multimode_ramsey','dc_offset_guess':0,'mode':mode,'update_config':True})

def multimode_dc_offset_recalibration(seq_exp, mode):
    # pulse_calibration(seq_exp)
    seq_exp.run('multimode_calibrate_offset',{'exp':'long_multimode_ramsey','dc_offset_guess':cfg['multimodes'][mode]['dc_offset_freq'],'mode':mode,'update_config':True})

def multimode_ef_pulse_calibration(seq_exp, mode):
    # pulse_calibration(seq_exp)
    seq_exp.run('multimode_calibrate_ef_sideband',{'exp':'multimode_ef_rabi','dc_offset_guess_ef':0,'mode':mode,'update_config':False})

def multimode_pi_pi_phase_calibration(seq_exp,mode):
    seq_exp.run('multimode_pi_pi_experiment',{'mode':mode,'update_config':True})

def multimode_ef_dc_offset_recalibration(seq_exp, mode):
    # pulse_calibration(seq_exp)
    seq_exp.run('multimode_calibrate_ef_sideband',{'exp':'long_multimode_ef_ramsey','dc_offset_guess_ef':cfg['multimodes'][mode]['dc_offset_freq_ef'],'mode':mode,'update_config':True})

def multimode_ef_dc_offset_calibration(seq_exp, mode):
    # pulse_calibration(seq_exp)
    seq_exp.run('multimode_calibrate_ef_sideband',{'exp':'short_multimode_ef_ramsey','dc_offset_guess_ef':0,'mode':mode,'update_config':True})


def multimode_ge_calibration_all(seq_exp, kwargs):
    frequency_stabilization(seq_exp)
    multimode_pulse_calibration(seq_exp, kwargs['mode'])
    #multimode_dc_offset_calibration(seq_exp, kwargs['mode'])
    multimode_dc_offset_recalibration(seq_exp, kwargs['mode'])
    multimode_pi_pi_phase_calibration(seq_exp, kwargs['mode'])


def multimode_cz_calibration(seq_exp, mode, data_file,data_file_2=None):
    frequency_stabilization(seq_exp)
    #seq_exp.run('EF_Ramsey',{'update_config':True,'data_file':data_file_2})
    seq_exp.run('multimode_qubit_mode_cz_v2_offset_experiment',
                {'mode': mode, 'offset_exp': 0, 'update_config': True, "data_file": data_file})

    frequency_stabilization(seq_exp)
    #seq_exp.run('EF_Ramsey',{'update_config':True,'data_file':data_file_2})

    seq_exp.run('multimode_qubit_mode_cz_v2_offset_experiment',
                {'mode': mode, 'offset_exp': 1, 'update_config': True, "data_file": data_file})

def multimode_mode_mode_cnot_calibration(seq_exp, mode,mode2, data_file=None,data_file_2=None):
    # frequency_stabilization(seq_exp)

    seq_exp.run('multimode_mode_mode_cnot_v2_offset_experiment',
                {'mode': mode,'mode2':mode2, 'offset_exp': 4, 'number':1, 'load_photon':False,'update_config': True})

    seq_exp.run('multimode_mode_mode_cnot_v2_offset_experiment',
                {'mode': mode,'mode2':mode2, 'offset_exp': 4,'number':1, 'load_photon':True,'update_config': True})

    seq_exp.run('multimode_mode_mode_cnot_v2_offset_experiment',
                {'mode': mode,'mode2':mode2, 'offset_exp': 5, 'load_photon':True,'update_config': True })

def multimode_mode_mode_cnot_calibration_v3(seq_exp, mode,mode2, data_file=None,data_file_2=None):
    # frequency_stabilization(seq_exp)
    # ef_frequency_calibration(seq_exp)

    seq_exp.run('multimode_mode_mode_cnot_v3_offset_experiment',
                {'mode': mode,'mode2':mode2, 'offset_exp': 0, 'number':1, 'load_photon':False,'update_config': True,'data_file':data_file})

    seq_exp.run('multimode_mode_mode_cnot_v3_offset_experiment',
                {'mode': mode,'mode2':mode2, 'offset_exp': 0,'number':1, 'load_photon':True,'update_config': True,'data_file':data_file})

    seq_exp.run('multimode_mode_mode_cnot_v3_offset_experiment',
                {'mode': mode,'mode2':mode2, 'offset_exp': 0,'number':1, 'load_photon':True,'include_cz_correction':False,'update_config': True,'data_file':data_file})

    seq_exp.run('multimode_mode_mode_cnot_v3_offset_experiment',
                {'mode': mode,'mode2':mode2, 'offset_exp': 1, 'load_photon':True,'update_config': True,'data_file':data_file })


def multimode_mode_mode_cnot_test(seq_exp, mode,mode2, data_file=None,data_file_2=None):
    # frequency_stabilization(seq_exp)

    seq_exp.run('multimode_mode_mode_cnot_v2_offset_experiment',
                {'mode': mode,'mode2':mode2, 'offset_exp': 6, 'load_photon':False,'number':1,'update_config': True})
    for number in arange(0,6):
        seq_exp.run('multimode_mode_mode_cnot_v2_offset_experiment',
                    {'mode': mode,'mode2':mode2, 'offset_exp': 6, 'load_photon':True,'number':number,'update_config': True })

def multimode_mode_mode_cnot_test_v3(seq_exp, mode, mode2, offset_exp, load_photon,test_one,number, data_file=None,data_file_2=None):
    frequency_stabilization(seq_exp)
    ef_frequency_calibration(seq_exp)

    if test_one:
        seq_exp.run('multimode_mode_mode_cnot_v3_offset_experiment',
                    {'mode': mode,'mode2':mode2, 'offset_exp': 2, 'load_photon':False,'number':1,'update_config': True,'data_file':data_file})
    for num in arange(number):
        seq_exp.run('multimode_mode_mode_cnot_v3_offset_experiment',
                    {'mode': mode,'mode2':mode2, 'offset_exp': 2, 'load_photon':True,'number':num,'update_config': True,'data_file':data_file})

def multimode_mode_mode_cz_calibration_v3(seq_exp, mode,mode2, data_file=None,data_file_2=None):
    # pulse_calibration(seq_exp,phase_exp=False)
    # ef_frequency_calibration(seq_exp)

    seq_exp.run('multimode_mode_mode_cz_v3_offset_experiment',
                {'mode': mode,'mode2':mode2, 'offset_exp': 0, 'number':1, 'load_photon':False,'update_config': True,'data_file':data_file})

    seq_exp.run('multimode_mode_mode_cz_v3_offset_experiment',
                {'mode': mode,'mode2':mode2, 'offset_exp': 0,'number':1, 'load_photon':True,'update_config': True,'data_file':data_file})

    seq_exp.run('multimode_mode_mode_cz_v3_offset_experiment',
                {'mode': mode,'mode2':mode2, 'offset_exp': 0,'number':1, 'load_photon':True,'include_cz_correction':False,'update_config': True,'data_file':data_file})

    seq_exp.run('multimode_mode_mode_cz_v3_offset_experiment',
                {'mode': mode,'mode2':mode2, 'offset_exp': 1, 'load_photon':True,'update_config': True,'data_file':data_file })

def multimode_mode_mode_cz_calibration_v3_debug(seq_exp, mode,mode2, data_file=None,data_file_2=None):

    seq_exp.run('multimode_mode_mode_cz_v3_offset_experiment',
                {'mode': mode,'mode2':mode2, 'offset_exp': 0,'number':1, 'load_photon':True,'update_config': False})


def multimode_mode_mode_cz_test_v3(seq_exp, mode, mode2, offset_exp, load_photon,test_one,number, data_file=None,data_file_2=None):
    # frequency_stabilization(seq_exp)
    # ef_frequency_calibration(seq_exp)

    if test_one:
        seq_exp.run('multimode_mode_mode_cz_v3_offset_experiment',
                    {'mode': mode,'mode2':mode2, 'offset_exp': 2, 'load_photon':False,'number':1,'update_config': True})

        seq_exp.run('multimode_mode_mode_cz_v3_offset_experiment',
                    {'mode': mode,'mode2':mode2, 'offset_exp': 2, 'load_photon':True,'number':1,'update_config': True})

    for num in arange(number):
        seq_exp.run('multimode_mode_mode_cz_v3_offset_experiment',
                    {'mode': mode,'mode2':mode2, 'offset_exp': offset_exp, 'load_photon':load_photon,'number':num,'update_config': True,'data_file':data_file})





def multimode_cz_2modes_calibration(seq_exp, mode, mode2, data_file, data_file_2=None):


    frequency_stabilization(seq_exp)
    #seq_exp.run('EF_Ramsey',{'update_config':True,'data_file':data_file_2})
    seq_exp.run('multimode_mode_mode_cz_v2_offset_experiment',
                {'mode': mode, 'mode2': mode2, 'offset_exp': 0, 'update_config': True, "data_file": data_file})

    frequency_stabilization(seq_exp)
    seq_exp.run('EF_Ramsey',{'update_config':True,'data_file':data_file_2})
    seq_exp.run('multimode_mode_mode_cz_v2_offset_experiment',
                {'mode': mode, 'mode2': mode2, 'offset_exp': 1, 'update_config': True, "data_file": data_file})

def multimode_stark_shift_calibration(seq_exp, mode, mode2, data_file):


    frequency_stabilization(seq_exp)

    seq_exp.run('multimode_ac_stark_shift_experiment',
                {'mode': mode, 'mode2': mode2, 'offset_exp': 0, 'update_config': True, "data_file": data_file})

    frequency_stabilization(seq_exp)

    seq_exp.run('multimode_ac_stark_shift_experiment',
                {'mode': mode, 'mode2': mode2, 'offset_exp': 1, 'update_config': True, "data_file": data_file})


def multimode_cz_test(seq_exp, mode ,data_file, data_file_2 ):
    multimode_cz_calibration(seq_exp, mode, data_file_2)
    seq_exp.run('multimode_qubit_mode_cz_v2_offset_experiment', {'mode': mode
        , 'offset_exp': 3, 'load_photon': False, "data_file": data_file})
    seq_exp.run('multimode_qubit_mode_cz_v2_offset_experiment', {'mode': mode
        , 'offset_exp': 3, 'load_photon': True, "data_file": data_file})


def multimode_cz_2modes_test(seq_exp, mode, mode2 ,data_file,data_file_2 ):
    multimode_cz_2modes_calibration(seq_exp, mode, mode2,data_file_2)

    seq_exp.run('multimode_mode_mode_cz_v2_offset_experiment',
                {'mode': mode, 'mode2': mode2, 'offset_exp': 3, 'load_photon': False, 'update_config': True, "data_file": data_file})
    seq_exp.run('multimode_mode_mode_cz_v2_offset_experiment',
                {'mode': mode, 'mode2': mode2, 'offset_exp': 3, 'load_photon': True, 'update_config': True, "data_file": data_file})


def run_multimode_seq_experiment(expt_name,lp_enable=True,**kwargs):
    seq_exp = SequentialExperiment(lp_enable)
    prefix = expt_name.lower()
    data_file = get_data_filename(prefix)

    if expt_name.lower() == 'multimode_pulse_calibration':
        multimode_pulse_calibration(seq_exp,kwargs['mode'])

    if expt_name.lower() == 'multimode_ef_pulse_calibration':
        multimode_ef_pulse_calibration(seq_exp,kwargs['mode'])

    if expt_name.lower() == 'multimode_ef_dc_offset_recalibration':
        multimode_ef_dc_offset_recalibration(seq_exp,kwargs['mode'])

    if expt_name.lower() == 'multimode_dc_offset_recalibration':
        multimode_dc_offset_recalibration(seq_exp,kwargs['mode'])

    if expt_name.lower() == 'multimode_dc_offset_calibration':
        multimode_dc_offset_calibration(seq_exp,kwargs['mode'])

    if expt_name.lower() == 'multimode_ef_dc_offset_calibration':
        multimode_ef_dc_offset_calibration(seq_exp,kwargs['mode'])

    if expt_name.lower() == 'multimode_pi_pi_phase_calibration':
        multimode_pi_pi_phase_calibration(seq_exp,kwargs['mode'])

    if expt_name.lower() == 'multimode_cphase_calibration':

        pulse_calibration(seq_exp)
        multimode_pulse_calibration(seq_exp,mode=kwargs['mode'])
        multimode_dc_offset_recalibration(seq_exp,mode=kwargs['mode'])
        multimode_ef_pulse_calibration(seq_exp,mode=kwargs['mode'])
        multimode_ef_dc_offset_recalibration(seq_exp,mode=kwargs['mode'])
        multimode_pulse_calibration(seq_exp,mode=kwargs['mode2'])
        multimode_pi_pi_phase_calibration(seq_exp,mode=kwargs['mode2'])

        for i in arange(4):
            frequency_stabilization(seq_exp)
            seq_exp.run('multimode_qubit_mode_cz_offset_experiment',{'mode':kwargs['mode'],'mode2':kwargs['mode2'],'offset_exp':0,'load_photon':0,'update_config':True})
            seq_exp.run('multimode_qubit_mode_cz_offset_experiment',{'mode':kwargs['mode'],'mode2':kwargs['mode2'],'offset_exp':1,'load_photon':0,'update_config':True})
            for offset_exp in arange(2,4):
                for load_photon in arange(2):
                    frequency_stabilization(seq_exp)
                    seq_exp.run('multimode_qubit_mode_cz_offset_experiment',{'mode':kwargs['mode'],'mode2':kwargs['mode2'],'offset_exp':offset_exp,'load_photon':load_photon,'update_config':True})


    if expt_name.lower() == 'sequential_state_dep_shift_calibration':
        modelist = array([1,6,9])
        frequency_stabilization(seq_exp)
        for mode in modelist:
            seq_exp.run('multimode_state_dep_shift',{'mode':mode,'exp':0,'qubit_shift_ge':1,'qubit_shift_ef':0,'update_config':True,'data_file':data_file})

    if expt_name.lower() == 'sequential_state_dep_qubit_pulse_calibration':
        modelist = array([0,1,3,4,5,6,7,9,10])
        # pulse_calibration(seq_exp,phase_exp=False)
        j = 0
        for mode in modelist:
            if j%3 == 0:
                pulse_calibration(seq_exp,phase_exp=False)
            seq_exp.run('multimode_state_dep_shift',{'mode':mode,'exp':7,'qubit_shift_ge':1,'qubit_shift_ef':0,'load_photon':True,'shift_freq':False,'update_config':False,'data_file':data_file})
            j+=1

    if expt_name.lower() == 'qubit_pulse_with_photon_vs_freq_calibration':
        mode = kwargs['mode']
        pulse_calibration(seq_exp,phase_exp=False)
        freqlist = linspace(-5e6,5e6,21)
        for freq in freqlist:
            seq_exp.run('multimode_state_dep_shift',{'mode':mode,'exp':8,'qubit_shift_ge':1,'qubit_shift_ef':0,'load_photon':True,'shift_freq':False,'add_freq':freq,'update_config':False,'data_file':data_file})

    if expt_name.lower() == 'readout_multimode_cross_kerr_experiment':
        modelist = array([0,1,3,4,5,6,7,9,10])
        for mode in modelist:
            seq_exp.run('multimode_state_dep_shift',{'mode':mode,'mode2':11,'exp':9,'qubit_shift_ge':1,'qubit_shift_ef':0,'load_photon':True,'update_config':False,'data_file':data_file})

    if expt_name.lower() == 'sequential_state_dep_pulse_phase_calibration':
        modelist = array([0,1,3,4,5,6,7,9,10])
        pulse_calibration(seq_exp,phase_exp=True)
        for mode in modelist:
            seq_exp.run('multimode_state_dep_shift',{'mode':mode,'exp':3,'qubit_shift_ge':1,'qubit_shift_ef':0,'load_photon':False,'update_config':False,'data_file':data_file})




    if expt_name.lower() == 'sequential_state_dep_sideband_pulse_calibration':
        mode1 = kwargs['mode']
        modelist = array([0,1,3,4,5,6,7,9,10])
        mode2list=[]
        for mode in modelist:
            if mode == mode1:
                pass
            else:
                mode2list.append(mode)

        pulse_calibration(seq_exp,phase_exp=True)
        for mode2 in mode2list:
            seq_exp.run('multimode_state_dep_shift',{'mode':mode1,'mode2':mode2,'exp':6,'add_freq':0,'shift_freq':kwargs['shift_freq'],'shift_freq_q':kwargs['shift_freq_q'],'load_photon':True,'qubit_shift_ge':1,'qubit_shift_ef':0,'update_config':False,'data_file':data_file})



    if expt_name.lower() == 'testing_echo_pi_sb':
        mode1 = kwargs['mode']
        for i in arange(5):
            seq_exp.run('multimode_echo_sideband_experiment',{'mode':mode1,'exp':2,'load_photon':True,'update_config':False, 'data_file':data_file})

    if expt_name.lower() == 'testing_echo_pi_sb_without_photon_load':
        modelist = array([0,3,4,7,9,10])
        seq_exp.run('multimode_echo_sideband_experiment',{'exp':3,'modelist':modelist,'update_config':False})
        seq_exp.run('multimode_echo_sideband_experiment',{'exp':4,'modelist':modelist,'update_config':False})

    if expt_name.lower() == 'calibrating_echo_pi_sb':
        mode1 = kwargs['mode']
        seq_exp.run('multimode_echo_sideband_experiment',{'mode':mode1,'exp':0,'load_photon':False,'update_config':True})

    if expt_name.lower() == 'calibrating_echo_pi_sb2':
        mode1 = kwargs['mode']
        seq_exp.run('multimode_echo_sideband_experiment',{'mode':mode1,'exp':1,'load_photon':False,'update_config':True})

    if expt_name.lower() == 'testing_echo_pi_q':
        seq_exp.run('multimode_state_dep_shift',{'exp':13,'add_freq':0,'load_photon':False,'qubit_shift_ge':1,'qubit_shift_ef':0,'update_config':False})

    if expt_name.lower() == 'calibrating_echo_pi_q':
        seq_exp.run('multimode_state_dep_shift',{'exp':12,'add_freq':0,'load_photon':False,'qubit_shift_ge':1,'qubit_shift_ef':0,'update_config':False})

    if expt_name.lower() == 'testing_pi_half_pi_shift':
        pulse_calibration(seq_exp,phase_exp=False)
        for i in arange(10):
            seq_exp.run('multimode_state_dep_shift',{'exp':14,'add_freq':0,'load_photon':kwargs['load_photon'],'qubit_shift_ge':1,'qubit_shift_ef':0,'update_config':False, 'data_file': data_file })

    if expt_name.lower() == 'qubit_mode_cross_kerr':
        modelist = array([0,1])
        mode = 0
        for mode2 in modelist:
        # pulse_calibration(seq_exp,phase_exp=False)
            for exp in arange(2):
                seq_exp.run('multimode_qubit_mode_cross_kerr',{'mode':mode,'mode2':mode2,'exp':exp,'update_config':True})


    if expt_name.lower() == 'cphase_segment_tests':
        modelist = array([4])
        pulse_calibration(seq_exp,phase_exp=True)
        multimode_pi_pi_phase_calibration(seq_exp,mode=6)
        multimode_ef_pulse_calibration(seq_exp,mode=4)
        for j in arange(3):
            for mode in modelist:
                for subexp in arange(5):
                    seq_exp.run('multimode_state_dep_shift',{'mode':mode,'mode2':6,'exp':6,'qubit_shift_ge':0,'qubit_shift_ef':1,'subexp':subexp,'update_config':True})
                    frequency_stabilization(seq_exp)

    if expt_name.lower() == 'multimode_cz_calibration':
        multimode_cz_calibration(seq_exp, kwargs['mode'])

    if expt_name.lower() == 'multimode_cz_2modes_calibration':
        multimode_cz_2modes_calibration(seq_exp, kwargs['mode'],kwargs['mode2'])


    if expt_name.lower() == 'multimode_cz_test':
        multimode_cz_test(seq_exp, kwargs['mode'], data_file)

    if expt_name.lower() == 'multimode_cz_2modes_test':
        multimode_cz_2modes_test(seq_exp, kwargs['mode'],kwargs['mode2'], data_file)

    if expt_name.lower() == 'sequential_multimode_cz_calibration':
        #data_file_2 = get_data_filename("sequential_ef_ramsey")
        while True:
            modelist = array([1])
            for mode in modelist:
                multimode_cz_calibration(seq_exp, mode, data_file)


    if expt_name.lower() == 'sequential_multimode_cz_2modes_calibration':
        #data_file_2 = get_data_filename("sequential_ef_ramsey")
        while True:
            mode = 1
            mode2 = 6
            multimode_cz_2modes_calibration(seq_exp, mode, mode2,data_file)


    if expt_name.lower() == 'sequential_multimode_stark_shift':
        while True:
            mode = 5
            mode2 = 6
            multimode_stark_shift_calibration(seq_exp, mode, mode2,data_file)

    if expt_name.lower() == 'sequential_multimode_cz_test':
        data_file_2 = get_data_filename("sequential_multimode_cz_calibration")
        while True:
            modelist = array([1])
            for mode in modelist:
                multimode_cz_test(seq_exp, mode, data_file, data_file_2)


    if expt_name.lower() == 'sequential_multimode_cz_2modes_test':

        data_file_2 = get_data_filename("sequential_multimode_cz_2modes_calibration")

        modelist = array([1,5,6,9])
        for mode in modelist:
            for mode2 in modelist:
                if not mode == mode2:
                    multimode_cz_2modes_test(seq_exp, mode, mode2, data_file, data_file_2)



    if expt_name.lower() == 'multimode_ge_calibration_all':
        multimode_ge_calibration_all(seq_exp,kwargs)

    if expt_name.lower() == 'multimode_ef_calibration_all':
        multimode_ef_pulse_calibration(seq_exp,kwargs['mode'])
        multimode_ef_dc_offset_recalibration(seq_exp,kwargs['mode'])


    if expt_name.lower() == 'sequential_multimode_calibration':

        # modelist = array([0,1,2,3,4,5,6,7,8,9,10])
        modelist = array([1,6,9])

        pulse_calibration(seq_exp)
        for mode in modelist:
            seq_exp.run('multimode_calibrate_offset',{'exp':'multimode_rabi','dc_offset_guess':0,'mode':mode,'update_config':True,'data_file':data_file})
            # multimode_dc_offset_recalibration(seq_exp,mode)
            # # multimode_ef_pulse_calibration(seq_exp,mode)
            # # multimode_ef_dc_offset_recalibration(seq_exp,mode)
            # # multimode_pi_pi_phase_calibration(seq_exp,mode)


    if expt_name.lower() == 'sequential_ef_pulse_calibration':
        update_config = True
        print "Update config = " +str(update_config)
        modelist = array([1,6,9])
        # modelist = array([1,6])
        # pulse_calibration(seq_exp)
        # ef_pulse_calibration(seq_exp)
        for mode in modelist:
           seq_exp.run('multimode_calibrate_ef_sideband',{'exp':'multimode_ef_rabi','dc_offset_guess_ef':0,'mode':mode,'sb_cool':False,'update_config':update_config,'data_file':data_file})


    if expt_name.lower() == 'sequential_ef_dc_offset_calibration':
        update_config = True
        print "Update config = " +str(update_config)
        modelist = array([1,6,9])

        # pulse_calibration(seq_exp)
        # ef_pulse_calibration(seq_exp)
        for mode in modelist:
            seq_exp.run('multimode_calibrate_ef_sideband',{'exp':'short_multimode_ef_ramsey','dc_offset_guess_ef':0,'mode':mode,'update_config':update_config,'data_file':data_file})

    if expt_name.lower() == 'sequential_ef_dc_offset_recalibration':
        update_config = True
        print "Update config = " +str(update_config)
        modelist = array([1,6,9])

        for mode in modelist:
             seq_exp.run('multimode_calibrate_ef_sideband',{'exp':'long_multimode_ef_ramsey','dc_offset_guess_ef':cfg['multimodes'][mode]['dc_offset_freq_ef'],'mode':mode,'update_config':update_config,'data_file':data_file})

    if expt_name.lower() == 'sequential_dc_offset_recalibration':
        update_config = True
        print "Update config = " +str(update_config)
        # modelist = array([0,1,2,3,4,5,6,7,8,9,10])
        modelist = array([1,6,9])
        # pulse_calibration(seq_exp)
        for mode in modelist:
            print "RE calibrating DC offset for mode: " +str(mode)
            seq_exp.run('multimode_calibrate_offset',{'exp':'long_multimode_ramsey','dc_offset_guess':cfg['multimodes'][mode]['dc_offset_freq'],'mode':mode,'update_config':update_config,'data_file':data_file})

    if expt_name.lower() == 'sequential_multimode_t1':
        modelist = arange([1,6,9])

        for mode in modelist:
            seq_exp.run('multimode_t1',{'mode':mode,'update_config':True,'data_file':data_file})


    if expt_name.lower() == 'sequential_dc_offset_calibration':
        modelist = array([1,6,9])


        pulse_calibration(seq_exp)
        for mode in modelist:
            print "Calibrating DC offset for mode: " +str(mode)
            seq_exp.run('multimode_calibrate_offset',{'exp':'short_multimode_ramsey','dc_offset_guess':0,'mode':mode,'update_config':True,'data_file':data_file})

    if expt_name.lower() == 'sequential_pi_pi_phase_calibration':
        # modelist = array([0,1,3,4,5,6,7,9,10])
        modelist = array([1,6,9])
        # pulse_calibration(seq_exp,phase_exp=True)
        for mode in modelist:
            seq_exp.run('multimode_pi_pi_experiment',{'mode':mode,'update_config':True,'data_file':data_file})


    if expt_name.lower() == 'multimode_rabi_scan':



        # freqspan = linspace(-10,20,31)
        # freqlist = array([1.7745, 2.295, 2.48076])*1e9
        # amplist = array([0.4,0.2,0.375])
        # modelist = array([1,6,9])

        freqspan = linspace(-10,20,31)
        freqlist = array([1.664, 1.828, 1.864, 1.944, 2.12, 2.360, 2.434, 2.582])*1e9
        modelist = array([0, 2, 3, 4, 5, 7, 8, 10])
        amplist = 0.4*ones(len(modelist))
        # freqspan = linspace(0,1000,1001)
        # freqlist = array([1.64])*1e9
        # amplist = array([0.4])
        # modelist = array([-1])


        for i in arange(len(modelist)):
            print "running Rabi sweep around mode %s"%(modelist[i])
            for freq in freqspan:
                flux_freq = freqlist[i] + freq*1e6
                seq_exp.run('multimode_rabi_sweep',{'flux_freq':flux_freq,'amp':amplist[i],"data_file":data_file})



    if expt_name.lower() == 'multimode_ef_rabi_scan':

        # freqspan = linspace(-1,25,26)
        # freqlist = array([2.5104, 2.6862])*1e9
        # amplist = array([0.4,0.4])
        # modelist = array([6,9])

        # freqspan = linspace(-12,12,25)
        # freqlist = array([1.995, 2.51876, 2.69456])*1e9 -5e6
        # amplist = array([0.15,0.2,0.2])
        # modelist = array([1,6,9])
        # freqspan = linspace(-12,12,25)
        # freqlist = array([1.995])*1e9 -5e6
        # amplist = array([0.1])
        # modelist = array([1])

        # freqspan = linspace(-15,15,31)
        # freqlist = array([1.664, 1.828, 1.864, 1.944, 2.12, 2.360, 2.434, 2.582])*1e9 + 202.23e6
        # modelist = array([0, 2, 3, 4, 5, 7, 8, 10])
        # amplist = 0.2*ones(len(modelist))
        #
        freqspan = linspace(-15,15,31)
        freqlist = array([1.864, 1.944, 2.12, 2.360, 2.434, 2.582])*1e9 + 202.23e6
        modelist = array([ 3, 4, 5, 7, 8, 10])
        amplist = 0.2*ones(len(modelist))

        for i in arange(len(modelist)):

            if modelist[i] in modelist:
                print "running ef Rabi sweep around mode %s"%(modelist[i])
                for freq in freqspan:
                    flux_freq = freqlist[i] + freq*1e6
                    seq_exp.run('multimode_ef_rabi_sweep',{'flux_freq':flux_freq,'amp':amplist[i],"data_file":data_file})
            else:
                pass


    if expt_name.lower() == 'sequential_single_mode_rb':
        for i in arange(kwargs['number']):
            if i%8 == 0:
                pulse_calibration(seq_exp,phase_exp=True)
                multimode_pulse_calibration(seq_exp,kwargs['mode'])
                multimode_pi_pi_phase_calibration(seq_exp,kwargs['mode'])
            seq_exp.run('single_mode_rb',{'mode':kwargs['mode'],"data_file":data_file})



    if expt_name.lower() == 'sequential_cphase_amplification':
        mode1 = 6
        mode2 = 1
        multimode_pulse_calibration(seq_exp,mode1)
        multimode_ef_pulse_calibration(seq_exp,mode2)
        multimode_pi_pi_phase_calibration(seq_exp,mode1)
        multimode_ef_dc_offset_recalibration(seq_exp,mode2)
        for i in arange(0,15):
            frequency_stabilization(seq_exp)
        seq_exp.run('multimode_ef_pi_pi_experiment',{'mode_1':mode1,'mode_2':mode2,'update_config':True})
        seq_exp.run('multimode_cphase_amplification',{'mode_1':mode1,'mode_2':mode2,'number':i,"data_file":data_file})

    if expt_name.lower() == 'sequential_cnot_amplification':
        mode1 = kwargs['control_mode']
        mode2 = kwargs['target_mode']
        # multimode_pulse_calibration(seq_exp,mode1)
        # multimode_ef_pulse_calibration(seq_exp,mode2)
        # multimode_pi_pi_phase_calibration(seq_exp,mode1)
        # multimode_ef_dc_offset_recalibration(seq_exp,mode2)
        for i in arange(0,12,2):
            # frequency_stabilization(seq_exp)
            seq_exp.run('multimode_cnot_amplification',{'mode_1':mode1,'mode_2':mode2,'number':i,"data_file":data_file})


    if expt_name.lower() == 'cphase_amplification':
        mode1 = kwargs['control_mode']
        mode2 = kwargs['target_mode']
        seq_exp.run('multimode_ef_pi_pi_experiment',{'mode_1':mode1,'mode_2':mode2,'update_config':True})
        seq_exp.run('multimode_cphase_amplification',{'mode_1':mode1,'mode_2':mode2,'number':15})

    if expt_name.lower() == 'sequential_cnot_calibration':
        multimode_mode_mode_cnot_calibration_v3(seq_exp,mode=kwargs['mode'],mode2=kwargs['mode2'],data_file = data_file)

    if expt_name.lower() == 'sequential_cnot_testing':
        multimode_mode_mode_cnot_test_v3(seq_exp,mode=kwargs['mode'],mode2=kwargs['mode2'],offset_exp=kwargs['offset_exp'],load_photon=kwargs['load_photon'],number=kwargs['number'],test_one=kwargs['test_one'],data_file=data_file)

    if expt_name.lower() == 'sequential_cz_calibration':
        multimode_mode_mode_cz_calibration_v3(seq_exp,mode=kwargs['mode'],mode2=kwargs['mode2'],data_file=data_file)
    if expt_name.lower() == 'sequential_cz_testing':
        multimode_mode_mode_cz_test_v3(seq_exp,mode=kwargs['mode'],mode2=kwargs['mode2'],offset_exp=kwargs['offset_exp'],load_photon=kwargs['load_photon'],number=kwargs['number'],test_one=kwargs['test_one'],data_file=data_file)

    if expt_name.lower() == 'multimode_vacuum_rabi':
        seq_exp.run('multimode_vacuum_rabi',{'mode':kwargs['mode'],'update_config':False})


    if expt_name.lower() == 'sequential_direct_spam_phase_calibration':
        mode=kwargs['mode']
        mode2list = [0,1,3,4,5,6,7,9]
        mode2list = [0,1,5,6,9,10]
        frequency_stabilization(seq_exp)
        for mode2 in mode2list:
            if mode2 == mode:
                pass
            else:
                seq_exp.run('multimode_mode_mode_cz_v3_offset_experiment',
                            {'mode': mode,'mode2':mode2, 'offset_exp': 5, 'load_photon':True,'update_config': True,'data_file':data_file})

    if expt_name.lower() == 'sequential_multimode_entanglement':
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        idmlist=[id1,id2]
        for idm in idmlist:
            seq_exp.run('multimode_entanglement',{'id1':kwargs['id1'],'id2':kwargs['id2'],'idm':idm,'2_mode':True,'GHZ':kwargs['GHZ'],'data_file':data_file})


    if expt_name.lower() == 'sequential_two_mode_tomography_phase_sweep':
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        # pulse_calibration(seq_exp,phase_exp=True)
        for tom_num in arange(15):
            seq_exp.run('multimode_two_resonator_tomography_phase_sweep',{'id1':kwargs['id1'],'id2':kwargs['id2'],'tomography_num':tom_num,'state_num':kwargs['state_num'],'data_file':data_file})

    if expt_name.lower() == 'sequential_process_tomography':
        update_config = True
        print "Update config = " +str(update_config)
        # Changes input state; measures a given correlator
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        idlist = array([id1,id2])

        frequency_stabilization(seq_exp)
        pulse_calibration(seq_exp,phase_exp=True)
        ef_frequency_calibration(seq_exp)
        ef_pulse_calibration(seq_exp)

        for state_num in arange(16):
            seq_exp.run('multimode_process_tomography_phase_sweep',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':state_num,'gate_num':kwargs['gate_num'],'tomography_num':kwargs['tom_num'],\
                                                                    'sb_cool':kwargs['sb_cool'],'phase_correct_cz':kwargs['phase_correct_cz'],'phase_correct_cnot':kwargs['phase_correct_cnot'],\
                                                                    'sweep_final_sb':kwargs['sweep_final_sb'],'sweep_cnot':kwargs['sweep_cnot'],'update_config': kwargs['update_config'],'data_file':data_file})

    if expt_name.lower() == 'multimode_process_tomography_correlations':
        # Correlator for a given input
        update_config = False
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        state_num = kwargs['state_num']
        gate_num = kwargs['gate_num']
        tom_num = kwargs['tom_num']
        # pulse_calibration(seq_exp,phase_exp=True)
        seq_exp.run('multimode_process_tomography_phase_sweep',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':state_num,'gate_num':kwargs['gate_num'],'tomography_num':kwargs['tom_num'],'sb_cool':kwargs['sb_cool'],\
                                                                    'phase_correct_cz':kwargs['phase_correct_cz'],'phase_correct_cnot':kwargs['phase_correct_cnot'],'sweep_final_sb':kwargs['sweep_final_sb'],'sweep_cnot':kwargs['sweep_cnot'],'update_config': kwargs['update_config']})

    if expt_name.lower() == 'sequential_process_tomography_correlations':
        # Correlator for a given input
        id1 = kwargs['id1']
        modelist = kwargs['modelist']
        state_num = kwargs['state_num']
        gate_num = kwargs['gate_num']
        tom_num = kwargs['tom_num']

        id2list=[]
        for i in arange(len(modelist)):
            if modelist[i] == id1:
                pass
            else:
                id2list.append(modelist[i])
        frequency_stabilization(seq_exp)
        ef_frequency_calibration(seq_exp)
        for id2 in id2list:
            seq_exp.run('multimode_process_tomography_phase_sweep',{'id1':kwargs['id1'],'id2':id2,'state_num':state_num,'gate_num':gate_num,'tomography_num':tom_num})




    if expt_name.lower() == 'tomography_pulse_length_sweep':
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        pulse_calibration(seq_exp,phase_exp=True)
        tom_num = kwargs['tom_num']
        seq_exp.run('multimode_two_resonator_tomography_phase_sweep',{'id1':kwargs['id1'],'id2':kwargs['id2'],'tomography_num':tom_num,'state_num':kwargs['state_num'],'data_file':data_file})


    if expt_name.lower() == 'multimode_dc_offset_scan_ge':

        freqspan = linspace(-10,20,31)
        freqlist = array([1.7745, 2.295, 2.48076])*1e9
        amplist = array([0.8,0.4,0.75])
        modelist = array([1,6,9])

        for i in arange(len(modelist)):
            print "running DC offset scan around mode %s"%(modelist[i])
            for freq in freqspan:
                flux_freq = freqlist[i] + freq*1e6
                seq_exp.run('multimode_dc_offset_experiment',{'freq':flux_freq,'amp':amplist[i],'sideband':"ge","data_file":data_file})

    if expt_name.lower() == 'multimode_dc_offset_scan_ef':


        # freqspan = linspace(-10,10,21)
        # freqlist = array([1.9854, 2.5104, 2.6862])*1e9
        # amplist = array([0.15,0.2,0.2])
        # modelist = array([1,6,9])

        freqspan = linspace(0,1000,1001)
        freqlist = array([1.64])*1e9 - 27.5e6
        amplist = array([0.15])
        modelist = array([-1])

        # freqspan = linspace(-1,25,26)
        # freqlist = array([1.9854, 2.5104, 2.6862])*1e9
        # amplist = array([0.3,0.4,0.4])
        # modelist = array([1,6,9])

        for i in arange(len(modelist)):
            print "running DC offset scan around mode %s"%(modelist[i])
            for freq in freqspan:
                flux_freq = freqlist[i] + freq*1e6
                print "Sweep frequency: " + str(flux_freq/1e9) + " GHz"
                seq_exp.run('multimode_dc_offset_experiment',{'freq':flux_freq,'amp':amplist[i],'sideband':"ef","data_file":data_file})

    if expt_name.lower() == 'multimode_ef_rabi_scan_corrected':

        freqspan = linspace(-1,25,26)
        freqlist = array([1.9854, 2.5104, 2.6862])*1e9
        amplist = np.load(r'S:\_Data\160912 - 2D Multimode Qubit (Chip MM3, 11 modes)\iPython Notebooks\ef_amplitudes_corr.npy')
        print np.shape(amplist)
        modelist = array([1,6,9])


        for i in arange(len(modelist)):
            print "running corrected ef sideband Rabi scan around mode %s"%(modelist[i])
            for ii,freq in enumerate(freqspan):
                flux_freq = freqlist[i] + freq*1e6
                print "Sweep frequency: " + str(flux_freq/1e9) + " GHz"
                seq_exp.run('multimode_ef_rabi_sweep',{'flux_freq':flux_freq,'amp':amplist[i][ii],"data_file":data_file})
            else:
                pass



    if expt_name.lower() == 'multimode_dc_offset_scan_ef_corrected':

        freqspan = linspace(-1,25,26)
        freqlist = array([1.9854, 2.5104, 2.6862])*1e9
        amplist = np.load(r'S:\_Data\160912 - 2D Multimode Qubit (Chip MM3, 11 modes)\iPython Notebooks\ef_amplitudes_corr.npy')
        print np.shape(amplist)
        modelist = array([1,6,9])


        for i in arange(len(modelist)):
            print "running corrected ef DC offset scan around mode %s"%(modelist[i])
            for ii,freq in enumerate(freqspan):
                flux_freq = freqlist[i] + freq*1e6
                print "Sweep frequency: " + str(flux_freq/1e9) + " GHz"
                seq_exp.run('multimode_dc_offset_experiment',{'freq':flux_freq,'amp':amplist[i][ii],'sideband':"ef","data_file":data_file})



    if expt_name.lower() == 'sequential_chirp_calibration':
        frequency_stabilization(seq_exp)
        modelist = array([0,1,5,6,9,10])

        for mode in modelist:
            seq_exp.run('multimode_dc_offset_experiment',{'mode_calibration':True,'mode':mode,'sideband':"ef",'update_config':True,"data_file":data_file})

    if expt_name.lower() == 'pi_pi_phase_test':

        for time in arange(0,300,10):
            print "mode = " + str(kwargs['mode'])
            print "sweep time = " + str(time) + " ns"
            seq_exp.run('multimode_pi_pi_experiment',{'mode':kwargs['mode'],'sweep_time':True,'time':time,'update_config':False,"data_file":data_file})


    if expt_name.lower() == 'find_chirp_freq_experiment':
        frequency_stabilization(seq_exp)
        for addfreq in linspace(-3e6,6e6,19):
            seq_exp.run('multimode_calibrate_ef_sideband',{'exp':'multimode_ef_rabi','dc_offset_guess_ef':0,'add_freq':addfreq,'mode':kwargs['mode'],'update_config':False,"data_file":data_file})