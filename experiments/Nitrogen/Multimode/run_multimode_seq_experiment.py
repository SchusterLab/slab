__author__ = 'Nelson/Vatsan'

from slab.experiments.Nitrogen.ExpLib.SequentialExperiment import *
from slab.experiments.Nitrogen.General.run_seq_experiment import *
from numpy import delete,linspace
from slab import *
import os
import json
from slab.dsfit import*


datapath = os.getcwd() + '\data'
config_file = os.path.join(datapath, "..\\config" + ".json")
with open(config_file, 'r') as fid:
        cfg_str = fid.read()

cfg = AttrDict(json.loads(cfg_str))

def get_data_filename(prefix):
    return  os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))

def multimode_pulse_calibration(seq_exp, mode,update_config=True):
    # pulse_calibration(seq_exp)
    # kwargs['update_config']
    seq_exp.run('multimode_calibrate_offset',{'exp':'multimode_rabi','dc_offset_guess':0,'mode':mode,'update_config':update_config})

def multimode_dc_offset_calibration(seq_exp, mode):
    # pulse_calibration(seq_exp)
    seq_exp.run('multimode_calibrate_offset',{'exp':'short_multimode_ramsey','dc_offset_guess':0,'mode':mode,'update_config':True})

def multimode_dc_offset_recalibration(seq_exp, mode):
    # pulse_calibration(seq_exp)
    seq_exp.run('multimode_calibrate_offset',{'exp':'long_multimode_ramsey','dc_offset_guess':cfg['multimodes'][mode]['dc_offset_freq'],'mode':mode,'update_config':True})

def multimode_ef_pulse_calibration(seq_exp, mode):
    # pulse_calibration(seq_exp)
    seq_exp.run('multimode_calibrate_ef_sideband',{'exp':'multimode_ef_rabi','dc_offset_guess_ef':0,'mode':mode,'update_config':True})

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
    # frequency_stabilization(seq_exp)
    # ef_frequency_calibration(seq_exp)

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

def process_tomography_find_ef_qubit_phase_ef_offset(seq_exp, mode, mode2):
    contrast_list=[]
    ef_sb_offset_list = linspace(0,360.0,6)
    for ef_sb_offset in ef_sb_offset_list:
        seq_exp.run('multimode_process_tomography_gate_fid_expt',{'id1':mode,'id2':mode2,'state_num':6,\
                                                                                'gate_num':0,'tomography_num':4,'phase_correct_cz':True,'phase_correct_cnot':False,\
                                                                                'sweep_final_sb':False,'sweep_ef_qubit_phase':True,'ef_sb_offset': ef_sb_offset,\
                                                                                'sweep_cnot':False,'truncated_save':False,'update_config': False})
        contrast_list.append(seq_exp.expt.contrast)

    # x_at_extremum = sin_phase(ef_sb_offset_list,contrast_list,180.0,'max')
    x_at_extremum = ef_sb_offset_list[argmax(contrast_list)]
    return x_at_extremum

def run_multimode_seq_experiment(expt_name,lp_enable=True,**kwargs):
    seq_exp = SequentialExperiment(lp_enable)
    prefix = expt_name.lower()
    data_file = get_data_filename(prefix)

    if expt_name.lower() == 'multimode_pulse_calibration':
        update_config = kwargs['update_config']
        print("Update config = " +str(update_config))

        multimode_pulse_calibration(seq_exp,kwargs['mode'],update_config=update_config)

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



################################################ Sequential multimode gate calibration experiments ######################################################


    if expt_name.lower() == 'sequential_multimode_stark_shift':
        while True:
            mode = 5
            mode2 = 6
            multimode_stark_shift_calibration(seq_exp, mode, mode2,data_file)

    if expt_name.lower() == 'sequential_multimode_calibration':
        update_config = kwargs['update_config']
        print("Update config = " +str(update_config))
        modelist = kwargs['modelist']

        pulse_calibration(seq_exp)
        for mode in modelist:
            seq_exp.run('multimode_calibrate_offset',{'exp':'multimode_rabi','dc_offset_guess':0,'mode':mode,'update_config':update_config,'data_file':data_file})


    if expt_name.lower() == 'sequential_ef_pulse_calibration':
        update_config = kwargs['update_config']
        print("Update config = " +str(update_config))
        modelist = kwargs['modelist']

        for mode in modelist:
           seq_exp.run('multimode_calibrate_ef_sideband',{'exp':'multimode_ef_rabi','dc_offset_guess_ef':0,'mode':mode,'sb_cool':False,'update_config':update_config,'data_file':data_file})


    if expt_name.lower() == 'sequential_ef_dc_offset_calibration':
        update_config = kwargs['update_config']
        print("Update config = " +str(update_config))
        modelist = kwargs['modelist']


        # pulse_calibration(seq_exp)
        # ef_pulse_calibration(seq_exp)
        for mode in modelist:
            seq_exp.run('multimode_calibrate_ef_sideband',{'exp':'short_multimode_ef_ramsey','dc_offset_guess_ef':0,'mode':mode,'update_config':update_config,'data_file':data_file})

    if expt_name.lower() == 'sequential_ef_dc_offset_recalibration':
        update_config = kwargs['update_config']
        print("Update config = " +str(update_config))
        modelist = kwargs['modelist']

        for mode in modelist:
             seq_exp.run('multimode_calibrate_ef_sideband',{'exp':'long_multimode_ef_ramsey','dc_offset_guess_ef':cfg['multimodes'][mode]['dc_offset_freq_ef'],'mode':mode,'update_config':update_config,'data_file':data_file})

    if expt_name.lower() == 'sequential_dc_offset_recalibration':
        update_config = kwargs['update_config']
        print("Update config = " +str(update_config))
        modelist = kwargs['modelist']

        for mode in modelist:
            print("RE calibrating DC offset for mode: " +str(mode))
            seq_exp.run('multimode_calibrate_offset',{'exp':'long_multimode_ramsey','dc_offset_guess':cfg['multimodes'][mode]['dc_offset_freq'],'mode':mode,'update_config':update_config,'data_file':data_file})

    if expt_name.lower() == 'sequential_multimode_t1':
        modelist = arange([0,1,3,4,5,6,7,9,10])

        for mode in modelist:
            seq_exp.run('multimode_t1',{'mode':mode,'update_config':True,'data_file':data_file})


    if expt_name.lower() == 'sequential_dc_offset_calibration':
        update_config = kwargs['update_config']
        print("Update config = " +str(update_config))
        modelist = kwargs['modelist']

        # pulse_calibration(seq_exp)
        for mode in modelist:
            print("Calibrating DC offset for mode: " +str(mode))
            seq_exp.run('multimode_calibrate_offset',{'exp':'short_multimode_ramsey','dc_offset_guess':0,'mode':mode,'update_config':True,'data_file':data_file})

    if expt_name.lower() == 'sequential_pi_pi_phase_calibration':
        update_config = kwargs['update_config']
        print("Update config = " +str(update_config))
        modelist = kwargs['modelist']

        for mode in modelist:
            seq_exp.run('multimode_pi_pi_experiment',{'mode':mode,'update_config':update_config,'data_file':data_file})


################################################################### Multimode Rabi scans ##################################################################

    if expt_name.lower() == 'multimode_rabi_scan':

        freqspan = linspace(-10,10,21)
        freqlist = array([ 1.6823167 ,  1.78388067,  1.9039681 ,  1.94607441,  2.13101483,
        2.31223641,  2.41044112,  2.48847049,  2.58910637])*1e9
        modelist = array([0,1,3,4,5,6,7,9,10])
        modeindexlist = kwargs['modeindexlist']
        amplist = array([ 1.89801629,  1.29881571,  2.80828336,  0.88971589,  0.79432847,
        1.19547487,  3.17182375,  1.2783333  ,  1.0])


        freqspan = linspace(-10,10,21)
        freqlist = array([ 1.6823167 ,  1.78388067,  1.9039681 ,  1.94607441,  2.13101483,
        2.31223641,  2.41044112,  2.48847049,  2.58910637])*1e9
        modelist = array([0,1,3,4,5,6,7,9,10])
        modeindexlist = kwargs['modeindexlist']
        amplist = array([ 1.89801629,  1.29881571,  2.80828336,  0.88971589,  0.79432847,
        1.19547487,  3.17182375,  1.2783333  ,  1.0])

        for i in modeindexlist:
            print("running Rabi sweep around mode %s"%(modelist[i]))
            # frequency_stabilization(seq_exp)
            for freq in freqspan:
                flux_freq = freqlist[i] + freq*1e6
                seq_exp.run('multimode_rabi_sweep',{'flux_freq':flux_freq,'amp':amplist[i],"data_file":data_file})




    if expt_name.lower() == 'multimode_rabi_scan_all':

        # freqspan = linspace(-10,10,21)
        # freqlist = array([ 1.6823167 ,  1.78388067,  1.9039681 ,  1.94607441,  2.13101483,
        # 2.31223641,  2.41044112,  2.48847049,  2.58910637])*1e9
        # modelist = array([0,1,3,4,5,6,7,9,10])
        # modeindexlist = kwargs['modeindexlist']
        # amplist = array([ 1.89801629,  1.29881571,  2.80828336,  0.88971589,  0.79432847,
        # 1.19547487,  3.17182375,  1.2783333  ,  1.0])
        #
        #
        freqspan = linspace(0,152,153)
        freqlist = array([ 1.798])*1e9
        modelist = array([-1])
        modeindexlist = kwargs['modeindexlist']
        amplist = array([0.5])

        for i in modeindexlist:
            print("running Rabi sweep around mode %s"%(modelist[i]))
            # frequency_stabilization(seq_exp)
            for freq in freqspan:
                flux_freq = freqlist[i] + freq*1e6
                seq_exp.run('multimode_rabi_sweep',{'flux_freq':flux_freq,'amp':amplist[i],"data_file":data_file})



    if expt_name.lower() == 'multimode_rabi_scan_all2':

        # freqspan = linspace(-10,10,21)
        # freqlist = array([ 1.6823167 ,  1.78388067,  1.9039681 ,  1.94607441,  2.13101483,
        # 2.31223641,  2.41044112,  2.48847049,  2.58910637])*1e9
        # modelist = array([0,1,3,4,5,6,7,9,10])
        # modeindexlist = kwargs['modeindexlist']
        # amplist = array([ 1.89801629,  1.29881571,  2.80828336,  0.88971589,  0.79432847,
        # 1.19547487,  3.17182375,  1.2783333  ,  1.0])
        #
        #
        freqspan = linspace(0,433,434)
        freqlist = array([ 2.443])*1e9
        modelist = array([-1])
        modeindexlist = kwargs['modeindexlist']
        amplist = array([0.5])

        for i in modeindexlist:
            print("running Rabi sweep around mode %s"%(modelist[i]))
            # frequency_stabilization(seq_exp)
            for freq in freqspan:
                flux_freq = freqlist[i] + freq*1e6
                seq_exp.run('multimode_rabi_sweep',{'flux_freq':flux_freq,'amp':amplist[i],"data_file":data_file})

    if expt_name.lower() == 'multimode_ef_rabi_scan':


        freqspan = [linspace(-2.5,2.5,15),linspace(-5.0,5.0,15)]
        freqlist = array([ 5.03583899,  5.69091682])*1e9
        modelist = arange(len(freqlist))
        modeindexlist = arange(len(freqlist))#kwargs['modeindexlist']
        amplist = ones(len(freqlist))

        for i in modeindexlist:
            print("running ef Rabi sweep around mode %s"%(modelist[i]))
            for freq in freqspan[i]:
                flux_freq = freqlist[i] + freq*1e6
                seq_exp.run('multimode_ef_rabi_sweep',{'flux_freq':flux_freq,'amp':amplist[i],"data_file":data_file})




####################################################################### DC offset scans ########################################

    if expt_name.lower() == 'multimode_dc_offset_scan_ge':

        # freqspan = linspace(0,1000,301)
        # freqlist = array([1.54])*1e9
        freqspan = linspace(0,2500,51)
        freqlist = array([0.8])*1e9
        modelist = array([-1])

        amp = kwargs['amp']

        for i in arange(len(freqlist)):
            # print "running DC offset scan around mode %s"%(modelist[i])
            for freq in freqspan:
                flux_freq = freqlist[i] + freq*1e6
                seq_exp.run('multimode_dc_offset_experiment', expt_kwargs = {'freq':flux_freq,'amp':amp,'sideband':"ge","data_file":data_file})

    if expt_name.lower() == 'multimode_dc_offset_vs_amplitude':

        freq = kwargs['freq']
        amplist = kwargs['amplist']
        if 'sideband' in kwargs:
            sideband = kwargs['sideband']
        else:
            sideband = "ge"
        for amp in amplist:
            seq_exp.run('multimode_dc_offset_experiment',{'freq':freq,'timelist':kwargs['timelist'],'ramsey_freq':kwargs['ramsey_freq'],'amp':amp,'sideband':sideband,"data_file":data_file})


    if expt_name.lower() == 'multimode_dc_offset_scan_ef':

        freqspan = linspace(0,1000,501)
        freqlist = array([1.64])*1e9 - 27.5e6
        amplist = array([0.2])
        modelist = array([-1])


        for i in arange(len(modelist)):
            print("running DC offset scan around mode %s"%(modelist[i]))
            for freq in freqspan:
                flux_freq = freqlist[i] + freq*1e6
                print("Sweep frequency: " + str(flux_freq/1e9) + " GHz")
                seq_exp.run('multimode_dc_offset_experiment',{'freq':flux_freq,'amp':amplist[i],'sideband':"ef","data_file":data_file})

################################################################### State dependent shifts, composite pulses, Cross-Kerr tests ###################################

    if expt_name.lower() == 'sequential_state_dep_shift_calibration':
        modelist = kwargs['modelist']
        # frequency_stabilization(seq_exp)
        for mode in modelist:
            frequency_stabilization(seq_exp)
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
    if expt_name.lower() == 'sequential_single_mode_rb':
        for i in arange(kwargs['number']):
            if i%4 == 0:
                #pulse_calibration(seq_exp,phase_exp=True)
                #multimode_pulse_calibration(seq_exp,kwargs['mode'])
                #multimode_pi_pi_phase_calibration(seq_exp,kwargs['mode'])
                frequency_stabilization(seq_exp)
                pulse_calibration(seq_exp,phase_exp=True)
            seq_exp.run('single_mode_rb',{'mode':kwargs['mode'],"data_file":data_file})


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


    if expt_name.lower() == 'sequential_chirp_calibration':
        frequency_stabilization(seq_exp)
        modelist = array([0,1,5,6,9,10])

        for mode in modelist:
            seq_exp.run('multimode_dc_offset_experiment',{'mode_calibration':True,'mode':mode,'sideband':"ef",'update_config':True,"data_file":data_file})

    if expt_name.lower() == 'pi_pi_phase_test':

        for time in arange(0,300,10):
            print("mode = " + str(kwargs['mode']))
            print("sweep time = " + str(time) + " ns")
            seq_exp.run('multimode_pi_pi_experiment',{'mode':kwargs['mode'],'sweep_time':True,'time':time,'update_config':False,"data_file":data_file})


    if expt_name.lower() == 'find_chirp_freq_experiment':
        frequency_stabilization(seq_exp)
        for addfreq in linspace(-3e6,6e6,19):
            seq_exp.run('multimode_calibrate_ef_sideband',{'exp':'multimode_ef_rabi','dc_offset_guess_ef':0,'add_freq':addfreq,'mode':kwargs['mode'],'update_config':False,"data_file":data_file})



################################################################## Gate Calibration / Amplification experiments ######################################

    if expt_name.lower() == 'sequential_cnot_calibration':
        multimode_mode_mode_cnot_calibration_v3(seq_exp,mode=kwargs['mode'],mode2=kwargs['mode2'],data_file = data_file)

    if expt_name.lower() == 'sequential_cnot_testing':
        multimode_mode_mode_cnot_test_v3(seq_exp,mode=kwargs['mode'],mode2=kwargs['mode2'],offset_exp=kwargs['offset_exp'],load_photon=kwargs['load_photon'],number=kwargs['number'],test_one=kwargs['test_one'],data_file=data_file)

    if expt_name.lower() == 'sequential_cz_calibration':
        multimode_mode_mode_cz_calibration_v3(seq_exp,mode=kwargs['mode'],mode2=kwargs['mode2'],data_file=data_file)
    if expt_name.lower() == 'sequential_cz_testing':
        multimode_mode_mode_cz_test_v3(seq_exp,mode=kwargs['mode'],mode2=kwargs['mode2'],offset_exp=kwargs['offset_exp'],load_photon=kwargs['load_photon'],number=kwargs['number'],test_one=kwargs['test_one'],data_file=data_file)



################################################################# Process tomography experiments ###########################################################

    if expt_name.lower() == 'process_tomography_sweeping_cphase_all':
        # Correlator for a given input
        print("Update config = " +str(kwargs['update_config']))
        print("Process tomography protocol = %s" %(kwargs['protocol']))
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        gate_num = kwargs['gate_num']
        phase = kwargs['cnot_ef_qubit_phase']
        osc_mat = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                   [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                   [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                   [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

        for s in arange(16):
            for t in arange(15):
                if osc_mat[s][t] == 1.0:
                    if kwargs['protocol'] == 1:
                        seq_exp.run('multimode_process_tomography_phase_sweep_new',{'id1':kwargs['id1'],'id2':kwargs['id2'],'pair_index':kwargs['pair_index'],'state_num':s,'gate_num':kwargs['gate_num'],'tomography_num':t,\
                                                                        'sb_cool':kwargs['sb_cool'],'phase_correct_cz':kwargs['phase_correct_cz'],'phase_correct_cnot':kwargs['phase_correct_cnot'],\
                                                                        'sweep_final_sb':kwargs['sweep_final_sb'],'cnot_ef_qubit_phase':phase,'sweep_cnot':kwargs['sweep_cnot'],\
                                                                        'update_config': kwargs['update_config'],"data_file":data_file})
                    else:
                        seq_exp.run('multimode_process_tomography_gate_fid_expt',{'id1':kwargs['id1'],'id2':kwargs['id2'],'pair_index':kwargs['pair_index'],'state_num':s,'gate_num':kwargs['gate_num'],'tomography_num':t,\
                                                                        'sb_cool':kwargs['sb_cool'],'phase_correct_cz':kwargs['phase_correct_cz'],'phase_correct_cnot':kwargs['phase_correct_cnot'],\
                                                                        'sweep_final_sb':kwargs['sweep_final_sb'],'cnot_ef_qubit_phase':phase,'sweep_cnot':kwargs['sweep_cnot'],\
                                                                        'update_config': kwargs['update_config'],"data_file":data_file})


    if expt_name.lower() == 'process_tomography_sweeping_cnot_all':
        # Correlator for a given input
        print("Update config = " +str(kwargs['update_config']))
        print("Process tomography protocol = %s" %(kwargs['protocol']))
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        gate_num = kwargs['gate_num']
        phase = kwargs['cnot_ef_qubit_phase']
        slist = array([5,6,9,10])
        tlist=  array([4,5,8,9])

        for t in tlist:
            for s in slist:
                if kwargs['protocol'] == 1:
                    seq_exp.run('multimode_process_tomography_phase_sweep_new',{'id1':kwargs['id1'],'id2':kwargs['id2'],'pair_index':kwargs['pair_index'],'state_num':s,'gate_num':kwargs['gate_num'],'tomography_num':t,\
                                                                        'sb_cool':kwargs['sb_cool'],'phase_correct_cz':True,'phase_correct_cnot':False,\
                                                                        'sweep_final_sb':False,'cnot_ef_qubit_phase':phase,'sweep_cnot':True,\
                                                                        'update_config': kwargs['update_config'],"data_file":data_file})
                else:
                    seq_exp.run('multimode_process_tomography_gate_fid_expt',{'id1':kwargs['id1'],'id2':kwargs['id2'],'pair_index':kwargs['pair_index'],'state_num':s,'gate_num':kwargs['gate_num'],'tomography_num':t,\
                                                                        'sb_cool':kwargs['sb_cool'],'phase_correct_cz':True,'phase_correct_cnot':False,\
                                                                        'sweep_final_sb':False,'cnot_ef_qubit_phase':phase,'sweep_cnot':True,\
                                                                        'update_config': kwargs['update_config'],"data_file":data_file})


    if expt_name.lower() == 'multimode_process_tomography_correlations_vs_ef_qubit_phase':
        # Correlator for a given input
        update_config = False
        print("Process tomography protocol = %s" %(kwargs['protocol']))
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        state_num = kwargs['state_num']
        gate_num = kwargs['gate_num']
        tom_num = kwargs['tom_num']
        sweep_phase_start = kwargs['start_phase']
        sweep_phase_stop = kwargs['stop_phase']
        sweep_phase_step = kwargs['step_phase']


        # pulse_calibration(seq_exp,phase_exp=True)
        for phase in arange(sweep_phase_start,sweep_phase_stop,sweep_phase_step):
            if kwargs['protocol'] == 1:
                seq_exp.run('multimode_process_tomography_phase_sweep_new',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':state_num,'gate_num':kwargs['gate_num'],'pair_index':kwargs['pair_index'],'tomography_num':kwargs['tom_num'],\
                                                                        'sb_cool':kwargs['sb_cool'],'phase_correct_cz':kwargs['phase_correct_cz'],'phase_correct_cnot':kwargs['phase_correct_cnot'],\
                                                                        'sweep_final_sb':kwargs['sweep_final_sb'], 'cnot_ef_qubit_phase':phase,'sweep_cnot':kwargs['sweep_cnot'],\
                                                                        'update_config': kwargs['update_config'],"data_file":data_file})
            else:
                seq_exp.run('multimode_process_tomography_gate_fid_expt',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':state_num,'gate_num':kwargs['gate_num'],'pair_index':kwargs['pair_index'],'tomography_num':kwargs['tom_num'],\
                                                                        'sb_cool':kwargs['sb_cool'],'phase_correct_cz':kwargs['phase_correct_cz'],'phase_correct_cnot':kwargs['phase_correct_cnot'],\
                                                                        'sweep_final_sb':kwargs['sweep_final_sb'], 'cnot_ef_qubit_phase':phase,'sweep_cnot':kwargs['sweep_cnot'],\
                                                                        'update_config': kwargs['update_config'],"data_file":data_file})

    if expt_name.lower() == 'process_tomography_sweeping_final_sb_all':
        # Correlator for a given input
        print("Update config = False")
        print("Process tomography protocol = %s" %(kwargs['protocol']))
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        gate_num = kwargs['gate_num']
        phase = kwargs['cnot_ef_qubit_phase']
        slist = arange(16)
        tlist=  arange(15)

        for t in tlist:
            frequency_stabilization(seq_exp)
            pulse_calibration(seq_exp)
            for s in slist:
                if kwargs['protocol'] == 1:
                    seq_exp.run('multimode_process_tomography_phase_sweep_new',{'id1':kwargs['id1'],'id2':kwargs['id2'],'pair_index':kwargs['pair_index'],'state_num':s,'gate_num':kwargs['gate_num'],'tomography_num':t,\
                                                                        'sb_cool':kwargs['sb_cool'],'phase_correct_cz':True,'phase_correct_cnot':True,\
                                                                        'sweep_final_sb':True,'cnot_ef_qubit_phase':phase,'sweep_cnot':False,\
                                                                        'update_config': False,"data_file":data_file})
                else:
                    seq_exp.run('multimode_process_tomography_gate_fid_expt',{'id1':kwargs['id1'],'id2':kwargs['id2'],'pair_index':kwargs['pair_index'],'state_num':s,'gate_num':kwargs['gate_num'],'tomography_num':t,\
                                                                    'sb_cool':kwargs['sb_cool'],'phase_correct_cz':True,'phase_correct_cnot':True,\
                                                                    'sweep_final_sb':True,'cnot_ef_qubit_phase':phase,'sweep_cnot':False,\
                                                                    'update_config': False,"data_file":data_file})

    if expt_name.lower() == 'process_tomography_sweeping_final_sb_vs_tom_num':
        # Correlator for a given input
        print("Update config = False")
        print("Process tomography protocol = %s" %(kwargs['protocol']))
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        gate_num = kwargs['gate_num']

        slist = arange(16)
        tlist=  arange(15)

        t =  kwargs['tom_num']
        # frequency_stabilization(seq_exp)
        # pulse_calibration(seq_exp,phase_exp=False)
        for s in slist:
            if kwargs['protocol'] == 1:
                phase = kwargs['cnot_ef_qubit_phase']
                seq_exp.run('multimode_process_tomography_phase_sweep_new',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':s,'pair_index':kwargs['pair_index'],'gate_num':kwargs['gate_num'],'tomography_num':t,\
                                                                    'sb_cool':kwargs['sb_cool'],'phase_correct_cz':True,'phase_correct_cnot':True,\
                                                                    'sweep_final_sb':True,'cnot_ef_qubit_phase':phase,'sweep_cnot':False,\
                                                                    'update_config': False,"data_file":data_file})
            else:
                seq_exp.run('multimode_process_tomography_gate_fid_expt',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':s,'gate_num':kwargs['gate_num'],'tomography_num':t,\
                                                                    'sb_cool':kwargs['sb_cool'],'phase_correct_cz':True,'phase_correct_cnot':True,\
                                                                    'sweep_final_sb':True, 'use_saved_cnot_ef_qubit_phase':True,'sweep_cnot':False,\
                                                                    'update_config': False,"data_file":data_file})

    if expt_name.lower() == 'process_tomography_sweeping_cphase_trunctated':
        # Correlator for a given input
        print("Update config = " +str(kwargs['update_config']))
        print("Process tomography protocol = %s" %(kwargs['protocol']))
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        gate_num = 0
        lookup = array([5, 53, 55, 101, 103, 165, 167, 197])
        Lslist =[[5,6,9,10,13,14],[5,6,9,10],[7,11],[5,6,9,10],[7,11],[5,6,9,10],[7,11],[5,6,9,10,13,14],[7,11]]
        Ltlist =[[0,1],[3,7],[3,7],[6],[6],[10],[10],[12,13],[12,13]]
        LslistCNOT =[5,6,9,10]
        LtlistCNOT=[4,5,8,9]
        Ltlist =[[0,1],[3,7],[3,7],[6],[6],[10],[10],[12,13],[12,13]]

        for expt_num in lookup:
            t = expt_num/16
            s = expt_num%16
            if kwargs['protocol'] == 1:
                seq_exp.run('multimode_process_tomography_phase_sweep_new',{'id1':kwargs['id1'],'id2':kwargs['id2'],'pair_index':kwargs['pair_index'],'state_num':s,\
                                                                                'gate_num':gate_num,'tomography_num':t,'phase_correct_cz':False,'phase_correct_cnot':False,\
                                                                                'sweep_final_sb':False,'sweep_cnot':False,'truncated_save':True,\
                                                                                'update_config': kwargs['update_config'],"data_file":data_file})
            else:
                seq_exp.run('multimode_process_tomography_gate_fid_expt',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':s,\
                                                                                'gate_num':gate_num,'tomography_num':t,'phase_correct_cz':False,'phase_correct_cnot':False,\
                                                                                'sweep_final_sb':False,'sweep_cnot':False,'truncated_save':True,\
                                                                                'update_config': kwargs['update_config'],"data_file":data_file})



    if expt_name.lower() == 'process_tomography_sweeping_cnot_trunctated':
        # Correlator for a given input
        print("Update config = " +str(kwargs['update_config']))
        print("Process tomography protocol = %s" %(kwargs['protocol']))
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        gate_num = 0
        lookup = array([69, 86, 137, 154])

        LslistCNOT =[5,6,9,10]
        LtlistCNOT=[4,5,8,9]

        # frequency_stabilization(seq_exp)
        # pulse_calibration(seq_exp,phase_exp=False)

        for expt_num in lookup:
            t = expt_num/16
            s = expt_num%16
            if kwargs['protocol'] == 1:
                seq_exp.run('multimode_process_tomography_phase_sweep_new',{'id1':kwargs['id1'],'id2':kwargs['id2'],'pair_index':kwargs['pair_index'],'state_num':s,\
                                                                                'gate_num':gate_num,'tomography_num':t,'phase_correct_cz':True,'phase_correct_cnot':False,\
                                                                                'sweep_final_sb':False,'sweep_cnot':True,'truncated_save':True,\
                                                                                'cnot_ef_qubit_phase':kwargs['cnot_ef_qubit_phase'],\
                                                                                'update_config': kwargs['update_config'],"data_file":data_file})
            else:
                seq_exp.run('multimode_process_tomography_gate_fid_expt',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':s,\
                                                                                'gate_num':gate_num,'tomography_num':t,'phase_correct_cz':True,'phase_correct_cnot':False,\
                                                                                'sweep_final_sb':False,'sweep_cnot':True,'truncated_save':True,\
                                                                                'use_saved_cnot_ef_qubit_phase':True,\
                                                                                'update_config': kwargs['update_config'],"data_file":data_file})



    if expt_name.lower() == 'process_tomography_sweeping_ef_qubit_phase':
        # Correlator for a given input
        print("Update config = " +str(False))
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        gate_num = 0


        t = 4
        s = 6
        if kwargs['protocol'] == 1:
            seq_exp.run('multimode_process_tomography_phase_sweep_new',{'id1':kwargs['id1'],'id2':kwargs['id2'],'pair_index':kwargs['pair_index'],'state_num':s,\
                                                                                'gate_num':gate_num,'tomography_num':t,'phase_correct_cz':True,'phase_correct_cnot':False,\
                                                                                'sweep_final_sb':False,'sweep_ef_qubit_phase':True,'ef_sb_offset': kwargs['ef_sb_offset'],\
                                                                                'sweep_cnot':False,'truncated_save':False,'update_config': False,"data_file":data_file})
        else:
            seq_exp.run('multimode_process_tomography_gate_fid_expt',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':s,\
                                                                                'gate_num':gate_num,'tomography_num':t,'phase_correct_cz':True,'phase_correct_cnot':False,\
                                                                                'sweep_final_sb':False,'sweep_ef_qubit_phase':True,'ef_sb_offset': kwargs['ef_sb_offset'],\
                                                                                'sweep_cnot':False,'truncated_save':False,'update_config': False,"data_file":data_file})




    if expt_name.lower() == 'process_tomography_calibrations_truncated_all':


        # Correlator for a given input

        print("Update config = " +str(kwargs['update_config']))

        print("Calibrating total phase error in state preparation and measurement")


        # print "Process tomography protocol = %s" %(kwargs['protocol'])
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        gate_num = 0
        lookup = array([5, 53, 55, 101, 103, 197])
        Lslist =[[5,6,9,10,13,14],[5,6,9,10],[7,11],[5,6,9,10],[7,11],[5,6,9,10,13,14]]
        Ltlist =[[0,1],[3,7],[3,7],[6,10],[6,10],[12,13]]
        LslistCNOT =[5,6,9,10]
        LtlistCNOT=[4,5,8,9]
        Ltlist =[[0,1],[3,7],[3,7],[6],[6],[10],[10],[12,13],[12,13]]
        lookup2 = array([69, 86, 137, 154])

        ### CPhase calibrations: Total error from state preparation and measurement segments of process tomography



        for expt_num in lookup:
            t = expt_num/16
            s = expt_num%16

            seq_exp.run('multimode_process_tomography_gate_fid_expt',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':s,\
                                                                                'gate_num':gate_num,'tomography_num':t,'phase_correct_cz':False,'phase_correct_cnot':False,\
                                                                                'sweep_final_sb':False,'sweep_cnot':False,'truncated_save':True,\
                                                                                'update_config': kwargs['update_config'],"data_file":data_file})


        ### Finding optimal ef sb offset for fiding optimal ef qubit phase
        print("Finding ef sb offset phase to best find optimal ef qubit phase")

        ef_sb_offset = process_tomography_find_ef_qubit_phase_ef_offset(seq_exp, kwargs['id1'], kwargs['id2'])


        # seq_exp.run('multimode_process_tomography_gate_fid_expt',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':6,\
        #                                                                     'gate_num':gate_num,'tomography_num':4,'phase_correct_cz':True,'phase_correct_cnot':False,\
        #                                                                     'sweep_final_sb':False,'sweep_cnot':False,'sweep_ef_sb_offset_phase':True,'truncated_save':True,\
        #                                                                     'update_config': kwargs['update_config']})

        # ef_sb_offset = seq_exp.expt.optimal_ef_sb_offset


        print("Optimal ef sb offset phase for finding optimal ef qubit phase: " + str(ef_sb_offset))

        ### Finding optimal ef qubit phase

        print("Calibrating ef qubit phase through using XX correlator")
        seq_exp.run('multimode_process_tomography_gate_fid_expt',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':6,\
                                                                                'gate_num':gate_num,'tomography_num':4,'phase_correct_cz':True,'phase_correct_cnot':False,\
                                                                                'sweep_final_sb':False,'sweep_ef_qubit_phase':True,'ef_sb_offset': ef_sb_offset,\
                                                                                'sweep_cnot':False,'truncated_save':False,'update_config': False,"data_file":data_file})
        print("Seperating state preparation and measurement error ")

        #### Calibrating phase to be added & subtracted from CZ & CNOT gates, to isolate preparation and measurement errors

        for expt_num in lookup2:
            t = expt_num/16
            s = expt_num%16

            seq_exp.run('multimode_process_tomography_gate_fid_expt',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':s,\
                                                                                'gate_num':gate_num,'tomography_num':t,'phase_correct_cz':True,'phase_correct_cnot':False,\
                                                                                'sweep_final_sb':False,'sweep_cnot':True,'truncated_save':True,\
                                                                                'use_saved_cnot_ef_qubit_phase':True,\
                                                                                'update_config': kwargs['update_config'],"data_file":data_file})


    if expt_name.lower() == 'process_tomography_sweeping_final_sb':
        # Correlator for a given input
        print("Update config = False")
        # print "Process tomography protocol = %s" %(kwargs['protocol'])
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        gate_num = kwargs['gate_num']
        slist = arange(16)
        slist = arange(16)
        tlist=  kwargs['tom_list']

        for t in tlist:
            for s in slist:

                seq_exp.run('multimode_process_tomography_gate_fid_expt',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':s,'gate_num':kwargs['gate_num'],'tomography_num':t,\
                                                                    'sb_cool':kwargs['sb_cool'],'phase_correct_cz':True,'phase_correct_cnot':True,\
                                                                    'sweep_final_sb':True, 'use_saved_cnot_ef_qubit_phase':True,'sweep_cnot':False,\
                                                                    'update_config': False,"data_file":data_file})


    if expt_name.lower() == 'process_tomography_without_sweep':
        # Correlator for a given input
        print("Update config = False")
        # print "Process tomography protocol = %s" %(kwargs['protocol'])
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        gate_num = kwargs['gate_num']
        slist = arange(16)


        seq_exp.run('multimode_process_tomography_2',{'id1':kwargs['id1'],'id2':kwargs['id2'],'gate_num':kwargs['gate_num'],'proc_tom_set':0,'update_config': False,"data_file":data_file})
        seq_exp.run('multimode_process_tomography_2',{'id1':kwargs['id1'],'id2':kwargs['id2'],'gate_num':kwargs['gate_num'],'proc_tom_set':1,'update_config': False,"data_file":data_file})


    if expt_name.lower() == 'process_tomography_nosweep_id_cz_3':
        # Correlator for a given input
        print("Update config = False")
        # print "Process tomography protocol = %s" %(kwargs['protocol'])
        id1 = kwargs['id1']
        id2 = kwargs['id2']


        seq_exp.run('multimode_process_tomography_2',{'id1':kwargs['id1'],'id2':kwargs['id2'],'gate_num':0,'proc_tom_set':0,'update_config': False,"data_file":data_file})
        seq_exp.run('multimode_process_tomography_2',{'id1':kwargs['id1'],'id2':kwargs['id2'],'gate_num':0,'proc_tom_set':1,'update_config': False,"data_file":data_file})
        seq_exp.run('multimode_process_tomography_2',{'id1':kwargs['id1'],'id2':kwargs['id2'],'gate_num':1,'proc_tom_set':0,'update_config': False,"data_file":data_file})
        seq_exp.run('multimode_process_tomography_2',{'id1':kwargs['id1'],'id2':kwargs['id2'],'gate_num':1,'proc_tom_set':1,'update_config': False,"data_file":data_file})

    if expt_name.lower() == 'process_tomography_nosweep_cz_3_idphase_flip':
        # Correlator for a given input
        print("Update config = False")
        # print "Process tomography protocol = %s" %(kwargs['protocol'])
        id1 = kwargs['id1']
        id2 = kwargs['id2']

        print("Process tomography for CZ(" + str(id2) +", "+str(id1)+"), extra pi cphase added to Identity")


        seq_exp.run('multimode_process_tomography_2',{'id1':kwargs['id1'],'id2':kwargs['id2'],'gate_num':-1,'proc_tom_set':0,'update_config': False,"data_file":data_file})
        seq_exp.run('multimode_process_tomography_2',{'id1':kwargs['id1'],'id2':kwargs['id2'],'gate_num':-1,'proc_tom_set':1,'update_config': False,"data_file":data_file})


    if expt_name.lower() == 'final_sb_sweep_only_cnot_id_cz_3':
        LslistCNOT =[5,6,9,10]
        LtlistCNOT= [4,5,8,9]

        for s in LslistCNOT:
            for t in LtlistCNOT:
                seq_exp.run('multimode_process_tomography_gate_fid_expt',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':s,'gate_num':0,'tomography_num':t,\
                                                                    'sb_cool':False,'phase_correct_cz':True,'phase_correct_cnot':True,\
                                                                    'sweep_final_sb':True, 'use_saved_cnot_ef_qubit_phase':True,'sweep_cnot':False,\
                                                                    'update_config': False,"data_file":data_file})
        for s in LslistCNOT:
            for t in LtlistCNOT:
                seq_exp.run('multimode_process_tomography_gate_fid_expt',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':s,'gate_num':1,'tomography_num':t,\
                                                                    'sb_cool':False,'phase_correct_cz':True,'phase_correct_cnot':True,\
                                                                    'sweep_final_sb':True, 'use_saved_cnot_ef_qubit_phase':True,'sweep_cnot':False,\
                                                                    'update_config': False,"data_file":data_file})


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
        # frequency_stabilization(seq_exp)
        # ef_frequency_calibration(seq_exp)
        for id2 in id2list:
            seq_exp.run('multimode_process_tomography_gate_fid_expt',{'id1':kwargs['id1'],'id2':id2,'state_num':state_num,'gate_num':gate_num,'tomography_num':tom_num,\
                                                                      'phase_correct_cz':True,'phase_correct_cnot':True,\
                                                                    'sweep_final_sb':True, 'use_saved_cnot_ef_qubit_phase':True,'sweep_cnot':False,\
                                                                    'update_config': False,"data_file":data_file})




######################### Process tomography test experiments ########################################################

### Testing CZ-->ZC and prep swap for correlators XZ,YZ

    if expt_name.lower() == 'process_tomography_xz_yz_test':

        # Correlator for a given input

        print("Update config = " +str(kwargs['update_config']))
        # print "Process tomography protocol = %s" %(kwargs['protocol'])
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        gate_num = 0
        ### Only XZ,YZ correlators
        lookup = array([101, 103, 165, 167])
        Lslist =[[5,6,9,10,13,14],[5,6,9,10],[7,11],[5,6,9,10],[7,11],[5,6,9,10],[7,11],[5,6,9,10,13,14],[7,11]]
        Ltlist =[[0,1],[3,7],[3,7],[6],[6],[10],[10],[12,13],[12,13]]
        LslistCNOT =[5,6,9,10]
        LtlistCNOT=[4,5,8,9]
        lookup2 = array([69, 86, 137, 154])
        print("Goes here")

        ### CPhase calibrations: Total error from state preparation and measurement segments of process tomography

        for expt_num in lookup:
            t = expt_num/16
            s = expt_num%16

            seq_exp.run('multimode_process_tomography_phase_sweep_test',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':s,\
                                                                                'gate_num':gate_num,'tomography_num':t,'phase_correct_cz':False,'phase_correct_cnot':False,\
                                                                                'sweep_final_sb':False,'sweep_cnot':False,'truncated_save':True,\
                                                                                'update_config': kwargs['update_config'],"data_file":data_file})


    if expt_name.lower() == 'process_tomography_xz_yz_test_final_sb':
        # Correlator for a given input
        print("Update config = False")
        # print "Process tomography protocol = %s" %(kwargs['protocol'])
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        gate_num = kwargs['gate_num']
        slist = arange(16)
        tlist=  array(6,10)

        for t in tlist:
            for s in slist:

                seq_exp.run('multimode_process_tomography_phase_sweep_test',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':s,'gate_num':kwargs['gate_num'],'tomography_num':t,\
                                                                    'sb_cool':kwargs['sb_cool'],'phase_correct_cz':True,'phase_correct_cnot':True,\
                                                                    'sweep_final_sb':True, 'use_saved_cnot_ef_qubit_phase':True,'sweep_cnot':False,\
                                                                    'update_config': False,"data_file":data_file})



########################################################################## Multimode Entanglement Experiments #######################################################

    if expt_name.lower() == 'sequential_multimode_entanglement':
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        idmlist=[id1,id2]
        for idm in idmlist:
            seq_exp.run('multimode_entanglement',{'id1':kwargs['id1'],'id2':kwargs['id2'],'idm':idm,'2_mode':True,'GHZ':kwargs['GHZ'],'data_file':data_file})



    if expt_name.lower() == 'entanglement_polytope_measurement':
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        id3 = kwargs['id3']

        idmlist=[id1,id2,id3]
        tom_pulse_list = arange(3)
        for idm in idmlist:
            for tom_pulse in tom_pulse_list:
                seq_exp.run('multimode_general_entanglement',{'id1':kwargs['id1'],'id2':kwargs['id2'],'id3':kwargs['id3'],'idm':idm,'tomography':True,'GHZ':kwargs['GHZ'],'tom_pulse':tom_pulse,'number':kwargs['number'],'data_file':data_file})

    if expt_name.lower() == 'entanglement_polytope_measurement_all':
        idlist = kwargs['idlist']
        number = kwargs['number']

        if len(idlist) < 9:
            idlist = append(idlist,-ones(9-len(idlist)))

        idlist= [int(id) for id in idlist]

        idmlist=idlist[:number]
        tom_pulse_list = arange(3)
        for idm in idmlist:
            for tom_pulse in tom_pulse_list:
                seq_exp.run('multimode_general_entanglement',{'id1':idlist[0],'id2':idlist[1],'id3':idlist[2],'id4':idlist[3],\
                                                              'id5':idlist[4],'id6':idlist[5],'id7':idlist[6],'id8':idlist[7],'id9':idlist[8],\
                                                              'idm':idm,'tomography':True,'GHZ':kwargs['GHZ'],'tom_pulse':tom_pulse,'number':number,'data_file':data_file})


    if expt_name.lower() == 'sequential_two_mode_tomography_phase_sweep':
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        # pulse_calibration(seq_exp,phase_exp=True)
        for tom_num in arange(15):
            seq_exp.run('multimode_two_resonator_tomography_phase_sweep',{'id1':kwargs['id1'],'id2':kwargs['id2'],'tomography_num':tom_num,'state_num':kwargs['state_num'],'data_file':data_file})

    if expt_name.lower() == 'sequential_ghz_entanglement_witness':
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        id3 = kwargs['id3']
        final_sb_offset = kwargs['final_sb_offset']
        pi_q_ef_offset =  kwargs['pi_q_ef_offset']
        for tom_num in arange(4):
            seq_exp.run('multimode_ghz_entanglement_witness',{'id1':kwargs['id1'],'id2':kwargs['id2'],'id3':kwargs['id3'],'sweep_ef_qubit_phase':False,\
                                                              'final_sb_offset':final_sb_offset,'pi_q_ef_offset':pi_q_ef_offset,\
                                                              'tomography_num':tom_num,'state_num':0,'data_file':data_file})

    if expt_name.lower() == 'sequential_ghz_entanglement_witness_ef_sweep':
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        id3 = kwargs['id3']
        final_sb_offset = kwargs['final_sb_offset']
        pi_q_ef_offset = 0
        for tom_num in arange(4):
            seq_exp.run('multimode_ghz_entanglement_witness',{'id1':kwargs['id1'],'id2':kwargs['id2'],'id3':kwargs['id3'],'sweep_ef_qubit_phase':True,\
                                                              'final_sb_offset':final_sb_offset,'pi_q_ef_offset':pi_q_ef_offset,\
                                                              'tomography_num':tom_num,'state_num':0,'data_file':data_file})


    if expt_name.lower() == 'sequential_ghz_entanglement_witness_echo_pi':
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        id3 = kwargs['id3']
        final_sb_offset = kwargs['final_sb_offset']
        ramsey_freq = kwargs['ramsey_freq']
        number_pi_pulses = kwargs['number_pi_pulses']
        startstopstep = kwargs['startstopstep']
        pi_q_ef_offset = 0
        for tom_num in arange(4):
            seq_exp.run('multimode_ghz_entanglement_witness',{'id1':kwargs['id1'],'id2':kwargs['id2'],'id3':kwargs['id3'],'sweep_ef_qubit_phase':False,\
                                                              'echo_pi_time_sweep':True,'final_sb_offset':final_sb_offset,'pi_q_ef_offset':pi_q_ef_offset,\
                                                              'ramsey_freq':ramsey_freq,'number_pi_pulses':number_pi_pulses,'startstopstep':startstopstep,'tomography_num':tom_num,'state_num':0,'data_file':data_file})




######################################################################## Obsolete Experimental Sequences ###############################################################################

# Deleted all expts with old wrong process tomography protocol

    if expt_name.lower() == 'multimode_ef_rabi_scan_corrected':

        freqspan = linspace(-1,25,26)
        freqlist = array([1.9854, 2.5104, 2.6862])*1e9
        amplist = np.load(r'S:\_Data\160912 - 2D Multimode Qubit (Chip MM3, 11 modes)\iPython Notebooks\ef_amplitudes_corr.npy')
        print(np.shape(amplist))
        modelist = array([1,6,9])


        for i in arange(len(modelist)):
            print("running corrected ef sideband Rabi scan around mode %s"%(modelist[i]))
            for ii,freq in enumerate(freqspan):
                flux_freq = freqlist[i] + freq*1e6
                print("Sweep frequency: " + str(flux_freq/1e9) + " GHz")
                seq_exp.run('multimode_ef_rabi_sweep',{'flux_freq':flux_freq,'amp':amplist[i][ii],"data_file":data_file})
            else:
                pass



    if expt_name.lower() == 'multimode_dc_offset_scan_ef_corrected':

        freqspan = linspace(-1,25,26)
        freqlist = array([1.9854, 2.5104, 2.6862])*1e9
        amplist = np.load(r'S:\_Data\160912 - 2D Multimode Qubit (Chip MM3, 11 modes)\iPython Notebooks\ef_amplitudes_corr.npy')
        print(np.shape(amplist))
        modelist = array([1,6,9])


        for i in arange(len(modelist)):
            print("running corrected ef DC offset scan around mode %s"%(modelist[i]))
            for ii,freq in enumerate(freqspan):
                flux_freq = freqlist[i] + freq*1e6
                print("Sweep frequency: " + str(flux_freq/1e9) + " GHz")
                seq_exp.run('multimode_dc_offset_experiment',{'freq':flux_freq,'amp':amplist[i][ii],'sideband':"ef","data_file":data_file})




    if expt_name.lower() == 'tomography_pulse_length_sweep':
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        pulse_calibration(seq_exp,phase_exp=True)
        tom_num = kwargs['tom_num']
        seq_exp.run('multimode_two_resonator_tomography_phase_sweep',{'id1':kwargs['id1'],'id2':kwargs['id2'],'tomography_num':tom_num,'state_num':kwargs['state_num'],'data_file':data_file})


    if expt_name.lower() == 'multimode_process_tomography_correlations_vs_time':
        # Correlator for a given input
        update_config = False
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        state_num = kwargs['state_num']
        gate_num = kwargs['gate_num']
        tom_num = kwargs['tom_num']
        sweep_time_start = 0
        sweep_time_stop =100
        sweep_time_step = 5


        # pulse_calibration(seq_exp,phase_exp=True)
        for sweep_time in arange(sweep_time_start,sweep_time_stop,sweep_time_step):
            seq_exp.run('multimode_process_tomography_phase_sweep_new',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':state_num,'gate_num':kwargs['gate_num'],'tomography_num':kwargs['tom_num'],'sb_cool':kwargs['sb_cool'],\
                                                                        'phase_correct_cz':kwargs['phase_correct_cz'],'phase_correct_cnot':kwargs['phase_correct_cnot'],'sweep_final_sb':kwargs['sweep_final_sb'],\
                                                                        'sweep_time':sweep_time,'sweep_cnot':kwargs['sweep_cnot'],'update_config': kwargs['update_config'],"data_file":data_file})

    if expt_name.lower() == 'process_tomography_debug':
        # Correlator for a given input
        print("Update config = " +str(kwargs['update_config']))
        id1 = kwargs['id1']
        id2 = kwargs['id2']
        state_num_start = kwargs['state_num_start']
        state_num_stop = kwargs['state_num_stop']
        gate_num = kwargs['gate_num']
        tom_num = kwargs['tom_num']
        phase = kwargs['cnot_ef_qubit_phase']

        for state_num in arange(state_num_start,state_num_stop):
            seq_exp.run('multimode_process_tomography_phase_sweep_new',{'id1':kwargs['id1'],'id2':kwargs['id2'],'state_num':state_num,'gate_num':kwargs['gate_num'],'tomography_num':kwargs['tom_num'],\
                                                                        'sb_cool':kwargs['sb_cool'],'phase_correct_cz':kwargs['phase_correct_cz'],'phase_correct_cnot':kwargs['phase_correct_cnot'],\
                                                                        'sweep_final_sb':kwargs['sweep_final_sb'],'cnot_ef_qubit_phase':phase,'sweep_cnot':kwargs['sweep_cnot'],\
                                                                        'update_config': kwargs['update_config'],"data_file":data_file})


### Obsolete
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
        seq_exp.run('multimode_ef_pi_pi_experiment',expt_kwargs={'mode_1':mode1,'mode_2':mode2,'update_config':True})
        seq_exp.run('multimode_cphase_amplification',expt_kwargs={'mode_1':mode1,'mode_2':mode2,'number':15})



#############################  Blue sideband experiments YAO ###########################################


    if expt_name.lower() == 'multimode_bluesideband_sweep':

        # offset_list=[-81656602.748319939, -80685740.24701868, -79599004.754083157, -78597136.31114462, -7.77e+07, -76807382.828515679, -76009899.855674267, -75209535.153403714, -74459064.946205765, -73689192.183382601, -72983759.987144202, -72272657.000603467, -71674416.255530506, -71201969.173228472, -70642005.063694507, -70177149.894169152, -69898846.796902165, -69611643.262061238, -69518137.464103252, -69601738.595220476, -69552498.380955413, -69494148.22051169, -69394164.918533742, -69319497.915831298, -69006328.29023242, -68794345.798679218, -6.865e+07, -68505009.638900757, -68590321.673740223, -68604714.195392057]
        freqspan = linspace(0,20,21) #MHz
        freqcenter = 3.21e9#2.745e9

        # freqspan = linspace(-1,1,11)        # freqcenter = -array([66.6])*1e6

        # phasespan = linspace(-30,30,201)
        # phasecenter = array([0])

        # delay = linspace(93,200,1)

        # predrive_time = linspace(0,100,101)

        with SlabFile(data_file) as f:
            f.append_line('freq', freqspan  * 1e6 + freqcenter)
            f.close()


        for ii,freq in enumerate(freqspan):

            sweep_freq = freqcenter + freq*1e6
            # sweep_freq = 4.83*1e9
            #add_freq = offset_list[ii]
            print("running BlueSideband sweep at %s"%(sweep_freq))
            #print "qubit DC offset at %s"%(add_freq)
            seq_exp.run('multimode_bluesideband_sweep',expt_kwargs={'flux_freq':sweep_freq,"data_file":data_file})

        # for freq in freqspan:
        #     sweep_freq = freqcenter + freq*1e6
        #     print "running qubit drive sweep at %s"%(sweep_freq)
        #     seq_exp.run('multimode_bluesideband_sweep',{'add_freq':sweep_freq,"data_file":data_file})

        # for phase in phasespan:
        #     sweep_phase = phasecenter + phase
        #     sweep_phase = 90
        #     print "setting pi/2 pulse phase to %s"%(sweep_phase)
        #     seq_exp.run('multimode_bluesideband_sweep',{'pi_pulse_phase':sweep_phase,"data_file":data_file})

        # for qubit_delay in delay:
        #     print "qubit delay set to %s"%(qubit_delay)
        #     seq_exp.run('multimode_bluesideband_sweep',{'qubit_delay':qubit_delay,"data_file":data_file})

        # for pi_pulse_delay in delay:
        #     print "pi pulse delay set to %s"%(pi_pulse_delay)
        #     seq_exp.run('multimode_bluesideband_sweep',{'pi_pulse_delay':pi_pulse_delay,"data_file":data_file})

        # for predrive in predrive_time:
        #     print "predrive time set to %s"%(predrive)
        #     seq_exp.run('multimode_bluesideband_sweep',{'predrive':predrive,"data_file":data_file})


##### 2017/11/06 Charge sideband experiments: Quantum flute


    if expt_name.lower() == 'multimode_rabi_line_cut_scan':


        freqspan = arange(-100,150,50.0)
        freqlist = array([ 5.668])*1e9
        modelist = array([1])
        modeindexlist = [0]#kwargs['modeindexlist']
        amplist = array([1.0])

        for i in modeindexlist:
            print("running charge Rabi sweep around mode %s"%(modelist[i]))
            for freq in freqspan:
                flux_freq = freqlist[i] + freq*1e6
                seq_exp.run('multimode_rabi_line_cut_sweep',expt_kwargs={'flux_freq':flux_freq,'amp':amplist[i],"data_file":data_file})



    if expt_name.lower() == 'multimode_rabi_line_cut_all':

        freqlist = array([0.77531355 ,0.92060819,1.16491355,  1.35761355,  1.55951355,1.75491355,  1.95901355])*1e9
        modelist = array([1])
        amplist = 1*ones(len(freqlist))

        for i,flux_freq in enumerate(freqlist):
            print("running ef Rabi sweep around mode %s"%(i))
            seq_exp.run('multimode_rabi_line_cut_sweep',expt_kwargs={'flux_freq':flux_freq,'amp':amplist[i],"data_file":data_file})


    if expt_name.lower() == 'multimode_charge_sideband_rabi':

        freqlist = array([0.802427224096])*1e9
        modelist = array([0])
        amplist = 1*ones(len(freqlist))

        for i,flux_freq in enumerate(freqlist):
            print("running charge sideband Rabi sweep at nu = %s GHz"%(flux_freq/1e9))
            seq_exp.run('multimode_charge_sideband_rabi_sweep',expt_kwargs={'flux_freq':flux_freq,'amp':amplist[i],"data_file":data_file})


    if expt_name.lower() == 'multimode_pulse_probe_iq_amp_sweep':


        amplist = arange(0,0.7,0.1)

        for i,amp in enumerate(amplist):
            seq_exp.run('multimode_pulse_probe_iq',expt_kwargs={'amp':amp,"data_file":data_file})