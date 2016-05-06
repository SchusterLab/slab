__author__ = 'Nelson'

from slab.instruments.Alazar import Alazar
from numpy import arange
import json
from slab import *
import gc
from run_sequential_experiment_file import frequency_stabilization,pulse_calibration

def prepare_alazar(cfg, expt_name, expt = None):
    if expt == None:
        cfg['alazar']['samplesPerRecord'] = 2 ** (cfg['readout']['width'] - 1).bit_length()
        sequence_length = int((cfg[expt_name.lower()]['stop']-cfg[expt_name.lower()]['start'])/cfg[expt_name.lower()]['step'])
        if (cfg[expt_name.lower()]['use_pi_calibration']):
            sequence_length+=2

        cfg['alazar']['recordsPerBuffer'] = sequence_length
        cfg['alazar']['recordsPerAcquisition'] = int(
            sequence_length * min(cfg[expt_name.lower()]['averages'], 100))
        print "Prep Card"
        adc = Alazar(cfg['alazar'])
    else:
        cfg['alazar']['samplesPerRecord'] = 2 ** (cfg['readout']['width'] - 1).bit_length()
        sequence_length = int((cfg[expt_name.lower()][expt]['stop']-cfg[expt_name.lower()][expt]['start'])/cfg[expt_name.lower()][expt]['step'])
        if (cfg[expt_name.lower()]['use_pi_calibration']):
            sequence_length+=2

        cfg['alazar']['recordsPerBuffer'] = sequence_length
        cfg['alazar']['recordsPerAcquisition'] = int(
            sequence_length * min(cfg[expt_name.lower()][expt]['averages'], 100))
        print "Prep Card"
        adc = Alazar(cfg['alazar'])
    return adc

def multimode_pulse_calibration():
    datapath = os.getcwd() + '\data'
    config_file = os.path.join(datapath, "..\\config" + ".json")
    with open(config_file, 'r') as fid:
        cfg_str = fid.read()

    cfg = AttrDict(json.loads(cfg_str))
    experiment_started = True
    from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeCalibrateOffsetExperiment

    save_to_file = True
    # mode_pts = array([0, 1, 3, 4, 5, 6, 9])
    mode_num=1


    adc = prepare_alazar(cfg, 'multimode_calibrate_offset_experiment', 'multimode_rabi')
    prefix = 'multimode_rabi_mode_' + str(mode_num) + '_experiment'
    data_file = os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))
    expt = MultimodeCalibrateOffsetExperiment(path=datapath, data_file=data_file, adc=adc,
                                              exp='multimode_rabi', dc_offset_guess=0, mode=mode_num,
                                              data_prefix = prefix,
                                              liveplot_enabled=True)
    expt.go()
    if save_to_file:
        expt.save_config()
        print "Saved Multimode Rabi pi and 2pi lengths to the config file"
    else:
        pass

    adc.close()
    expt = None

    del expt
    gc.collect()


def run_multimode_sequential_experiment(expt_name):
    import os
    import difflib

    expt_list = ['Multimode_Rabi']
    datapath = os.getcwd() + '\data'

    config_file = os.path.join(datapath, "..\\config" + ".json")
    with open(config_file, 'r') as fid:
        cfg_str = fid.read()

    cfg = AttrDict(json.loads(cfg_str))

    def prepare_alazar(cfg, expt_name, expt=None):
        if expt == None:
            cfg['alazar']['samplesPerRecord'] = 2 ** (cfg['readout']['width'] - 1).bit_length()
            sequence_length = int(
                (cfg[expt_name.lower()]['stop'] - cfg[expt_name.lower()]['start']) / cfg[expt_name.lower()]['step'])
            if (cfg[expt_name.lower()]['use_pi_calibration']):
                sequence_length += 2

            cfg['alazar']['recordsPerBuffer'] = sequence_length
            cfg['alazar']['recordsPerAcquisition'] = int(
                sequence_length * min(cfg[expt_name.lower()]['averages'], 100))
            print "Prep Card"
            adc = Alazar(cfg['alazar'])
        else:
            cfg['alazar']['samplesPerRecord'] = 2 ** (cfg['readout']['width'] - 1).bit_length()
            sequence_length = int((cfg[expt_name.lower()][expt]['stop'] - cfg[expt_name.lower()][expt]['start']) /
                                  cfg[expt_name.lower()][expt]['step'])
            if (cfg[expt_name.lower()]['use_pi_calibration']):
                sequence_length += 2

            cfg['alazar']['recordsPerBuffer'] = sequence_length
            cfg['alazar']['recordsPerAcquisition'] = int(
                sequence_length * min(cfg[expt_name.lower()][expt]['averages'], 100))
            print "Prep Card"
            adc = Alazar(cfg['alazar'])
        return adc


    expt = None

    experiment_started = False

    if  expt_name.lower() == 'calibrate_multimode':
        experiment_started = True
        multimode_pulse_calibration()

    if expt_name.lower() == 'multimode_rabi_sweep':
        experiment_started = True
        cfg['alazar']['samplesPerRecord'] = 2 ** (cfg['readout']['width'] - 1).bit_length()
        sequence_length = (cfg[expt_name.lower()]['stop'] - cfg[expt_name.lower()]['start']) / cfg[expt_name.lower()][
            'step']
        if (cfg[expt_name.lower()]['use_pi_calibration']):
            sequence_length += 2

        cfg['alazar']['recordsPerBuffer'] = sequence_length
        cfg['alazar']['recordsPerAcquisition'] = int(
            sequence_length * min(cfg[expt_name.lower()]['averages'], 100))
        print "Prep Card"
        adc = Alazar(cfg['alazar'])
        flux_freq_pts = arange(cfg[expt_name.lower()]['freq_start'], cfg[expt_name.lower()]['freq_stop'],
                               cfg[expt_name.lower()]['freq_step'])
        prefix = 'Multimode_Rabi_Sweep'
        data_file = os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeRabiSweepExperiment
        # Do Multimode Rabi
        for ii, flux_freq in enumerate(flux_freq_pts):
            print "Running Multimode Rabi Sweep with flux frequency: " + str(flux_freq)
            expt = MultimodeRabiSweepExperiment(path=datapath, data_file=data_file, adc=adc, flux_freq=flux_freq,
                                                liveplot_enabled=False)
            expt.go()

            expt = None
            del expt

            gc.collect()
        pass

    if expt_name.lower() == 'multimode_ef_rabi_sweep':
        experiment_started = True
        cfg['alazar']['samplesPerRecord'] = 2 ** (cfg['readout']['width'] - 1).bit_length()
        sequence_length = (cfg[expt_name.lower()]['stop'] - cfg[expt_name.lower()]['start']) / cfg[expt_name.lower()][
            'step']
        if (cfg[expt_name.lower()]['use_pi_calibration']):
            sequence_length += 2

        cfg['alazar']['recordsPerBuffer'] = sequence_length
        cfg['alazar']['recordsPerAcquisition'] = int(
            sequence_length * min(cfg[expt_name.lower()]['averages'], 100))
        print "Prep Card"
        adc = Alazar(cfg['alazar'])
        flux_freq_pts = arange(cfg[expt_name.lower()]['freq_start'], cfg[expt_name.lower()]['freq_stop'],
                               cfg[expt_name.lower()]['freq_step'])
        prefix = 'Multimode_EF_Rabi_Sweep'
        data_file = os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeEFRabiSweepExperiment
        # Do Multimode Rabi
        for ii, flux_freq in enumerate(flux_freq_pts):
            print "Running Multimode EF Rabi Sweep with flux frequency: " + str(flux_freq)
            expt = MultimodeEFRabiSweepExperiment(path=datapath, data_file=data_file, adc=adc, flux_freq=flux_freq,
                                                  liveplot_enabled=False)
            expt.go()

            expt = None
            del expt

            gc.collect()
        pass

    if expt_name.lower() == 'multimode_cphase_optimization_sweep':

        experiment_started = True
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import CPhaseOptimizationSweepExperiment

        adc = prepare_alazar(cfg, expt_name)

        idle_time_pts = arange(cfg[expt_name.lower()]['length_start'], cfg[expt_name.lower()]['length_stop'],
                               cfg[expt_name.lower()]['length_step'])

        prefix = 'multimode_cphase_optimization_sweep'
        data_file = os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))

        for ii, idle_time in enumerate(idle_time_pts):
            print "Idle_time: " + str(idle_time)
            expt = CPhaseOptimizationSweepExperiment(path=datapath, data_file=data_file, adc=adc, idle_time=idle_time,
                                                     liveplot_enabled=True)
            expt.go()

            expt = None
            del expt
            gc.collect()

    if expt_name.lower() == 'multimode_general_entanglement':

        experiment_started = True
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeGeneralEntanglementExperiment

        adc = prepare_alazar(cfg, expt_name)

        list_num = array([7])
        for list in list_num:

            if(list == 2):
                idm_list = array([9, 1, 6, 4, 5, 3])
                numberlist =arange(3,1,-1)
            elif(list==3):
                idm_list = array([9, 6, 1, 4, 5, 3])
                numberlist =arange(6,1,-1)
            elif(list==4):
                idm_list = array([6, 1, 4, 5, 9, 3])
                numberlist =arange(6,1,-1)

            elif(list==5):
                idm_list = array([1, 4, 6, 5, 9, 3])
                numberlist =array([3])
            elif(list==6):
                idm_list = array([1,6,9,4,5,3])
                # numberlist =arange(6,1,-1)
                numberlist =array([3])
            elif(list==7):
                idm_list = array([6,1,9,4,5,3])
                # numberlist =arange(6,1,-1)
                numberlist =array([3])

            for number in numberlist:
                # frequency_stabilization()
                for ii, idm in enumerate(idm_list):
                    prefix = 'multimode_general_entanglement_' + str(number) + '_l_' + str(list)+'_experiment'
                    data_file = os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))
                    print "Measured mode " + str(idm)
                    expt = MultimodeGeneralEntanglementExperiment(path=datapath, data_file=data_file, adc=adc, idm=idm, id1=idm_list[0],
                                                                  id2=idm_list[1], id3=idm_list[2], id4=idm_list[3], id5=idm_list[4], id6=idm_list[5], number=number,
                                                                  liveplot_enabled=True)
                    expt.go()
                    # expt.save_config()
                    expt = None
                    del expt
                    gc.collect()

    if expt_name.lower() == 'multimode_two_resonator_tomography_phase_sweep':

        experiment_started = True
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import \
            MultimodeTwoResonatorTomographyPhaseSweepExperiment
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import RamseyExperiment

        adc = prepare_alazar(cfg, expt_name)

        tom_pts = arange(0, 15)
        state_pts = arange(3, 7)

        prefix = 'multimode_two_resonator_tomography_phase_sweep'
        data_file = os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))

        for state_num in state_pts:
            print "State_number: " + str(state_num)
            # frequency_stabilization()

            for ii, tomography_num in enumerate(tom_pts):
                prefix = 'multimode_two_resonator_tomography_phase_sweep_' + str(state_num) + '_' + str(
                    tomography_num) + '_experiment'
                data_file = os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))
                print "Tomography_pulse_number: " + str(tomography_num)
                expt = MultimodeTwoResonatorTomographyPhaseSweepExperiment(path=datapath, data_file=data_file, adc=adc,
                                                                           tomography_num=tomography_num,
                                                                           state_num=state_num, liveplot_enabled=True)
                expt.go()

                expt = None
                del expt
                gc.collect()

    if expt_name.lower() == 'multimode_three_mode_correlation_experiment':

        experiment_started = True
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import \
            MultimodeThreeModeCorrelationExperiment
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import RamseyExperiment

        adc = prepare_alazar(cfg, expt_name)

        # tom_pts = arange(0, 9)
        state_pts = array([0])
        tom_pts =array([7])
        prefix = 'multimode_two_resonator_tomography_phase_sweep'
        data_file = os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))

        for state_num in state_pts:
            print "State_number: " + str(state_num)
            # frequency_stabilization()

            for ii, tomography_num in enumerate(tom_pts):
                prefix = 'multimode_three_mode_tomography_4_' + str(state_num) + '_' + str(
                    tomography_num) + '_experiment'
                data_file = os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))
                print "Tomography_pulse_number: " + str(tomography_num)
                expt = MultimodeThreeModeCorrelationExperiment(path=datapath, data_file=data_file, adc=adc,
                                                                           tomography_num=tomography_num,
                                                                           state_num=state_num, liveplot_enabled=True)
                expt.go()

                expt = None
                del expt
                gc.collect()



    if expt_name.lower() == 'multimode_calibrate_offset_experiment':

        experiment_started = True
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeCalibrateOffsetExperiment

        calibrate_sideband = False
        save_to_file = True
        # mode_pts = array([0, 1, 3, 4, 5, 6, 9])
        mode_pts = array([6])



        for ii, mode_num in enumerate(mode_pts):

            if calibrate_sideband:
                adc = prepare_alazar(cfg, expt_name, 'multimode_rabi')
                prefix = 'multimode_rabi_mode_' + str(mode_num) + '_experiment'
                data_file = os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))
                expt = MultimodeCalibrateOffsetExperiment(path=datapath, data_file=data_file, adc=adc,
                                                          exp='multimode_rabi', dc_offset_guess=0, mode=mode_num,
                                                          data_prefix = prefix,
                                                          liveplot_enabled=True)
                expt.go()
                if save_to_file:
                    expt.save_config()
                    print "Saved Multimode Rabi pi and 2pi lengths to the config file"
                else:
                    pass

                adc.close()
                expt = None

                del expt
                gc.collect()

            else:

                adc = prepare_alazar(cfg, expt_name, 'short_multimode_ramsey')
                prefix = 'short_multimode_ramsey_mode_' + str(mode_num) + '_experiment'
                data_file = os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))
                expt = MultimodeCalibrateOffsetExperiment(path=datapath, data_file=data_file, adc=adc,
                                                          exp='short_multimode_ramsey', dc_offset_guess=0,
                                                          mode=mode_num, liveplot_enabled=True)
                expt.go()
                print expt.suggested_dc_offset_freq
                dc_offset_guess = expt.suggested_dc_offset_freq
                adc.close()
                expt = None

                del expt
                gc.collect()

                adc = prepare_alazar(cfg, expt_name, 'long_multimode_ramsey')
                prefix = 'long_multimode_ramsey_mode_' + str(mode_num) + '_experiment'
                data_file = os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))
                expt = MultimodeCalibrateOffsetExperiment(path=datapath, data_file=data_file, adc=adc,
                                                          exp='long_multimode_ramsey', mode=mode_num,
                                                          dc_offset_guess=dc_offset_guess, liveplot_enabled=True)
                expt.go()
                print expt.suggested_dc_offset_freq
                expt.save_config()
                expt = None
                adc.close()

                gc.collect()


    if expt_name.lower() == 'multimode_calibrate_ef_sideband_experiment':

        experiment_started = True
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeCalibrateEFSidebandExperiment

        mode_pts = array([1])

        calibrate_sideband = True
        save_to_file = True
        for ii, mode_num in enumerate(mode_pts):

            if calibrate_sideband:

                adc = prepare_alazar(cfg, expt_name, 'multimode_ef_rabi')
                prefix = 'multimode_ef_rabi_mode_' + str(mode_num) + '_experiment'
                data_file = os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))
                expt = MultimodeCalibrateEFSidebandExperiment(path=datapath, data_file=data_file, adc=adc,
                                                              exp='multimode_ef_rabi', dc_offset_guess_ef=0,
                                                              mode=mode_num, liveplot_enabled=True)
                expt.go()
                expt.save_config()
                adc.close()
                expt = None
                print
                del expt
                gc.collect()

                print "Calibrated ge sideband for mode %s" %(mode_num)
            else:

                adc = prepare_alazar(cfg, expt_name, 'short_multimode_ef_ramsey')
                prefix = 'multimode_ef_ramsey_mode_' + str(mode_num) + '_experiment'
                data_file = os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))
                expt = MultimodeCalibrateEFSidebandExperiment(path=datapath, data_file=data_file, adc=adc,
                                                              exp='short_multimode_ef_ramsey', dc_offset_guess_ef=0,
                                                              mode=mode_num, liveplot_enabled=True)
                expt.go()
                print expt.suggested_dc_offset_freq_ef
                dc_offset_guess_ef = expt.suggested_dc_offset_freq_ef
                adc.close()
                expt = None
                print
                del expt
                gc.collect()

                adc = prepare_alazar(cfg, expt_name, 'long_multimode_ef_ramsey')
                prefix = 'long_multimode_ef_ramsey_mode_' + str(mode_num) + '_experiment'
                data_file = os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))
                expt = MultimodeCalibrateEFSidebandExperiment(path=datapath, data_file=data_file, adc=adc,
                                                              exp='long_multimode_ef_ramsey', mode=mode_num,
                                                              dc_offset_guess_ef=dc_offset_guess_ef,
                                                              liveplot_enabled=True)
                expt.go()
                print expt.suggested_dc_offset_freq_ef
                expt.save_config()
                expt = None
                adc.close()
                gc.collect()



    if expt_name.lower() == 'multimode_dc_offset_experiment':

        experiment_started = True
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeDCOffsetExperiment

        # freq_pts = [2.14e9,2.45e9,2.63e9,2.94e9]
        freq_pts = linspace(1328859060.4,2e9,100)
        amp_pts = array([0.25])
        i=7
        j=0
        frequency_stabilization()
        adc = prepare_alazar(cfg, expt_name)
        for freq in freq_pts:
            # i+=1

            for ii, amp in enumerate(amp_pts):
                if j%10 == 0:
                    frequency_stabilization()
                j+=1

                prefix1 = 'Multimode_DC_Offset_'
                prefix2 = str(i)
                prefix3 = '_Experiment'
                prefix = prefix1 + prefix2 + prefix3
                data_file = os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))
                expt = MultimodeDCOffsetExperiment(path=datapath, data_file=data_file, adc=adc,amp = amp, freq= freq, data_prefix = prefix, liveplot_enabled=True)
                expt.go()
                expt = None
                print
                del expt
                gc.collect()

        adc.close()



    if expt_name.lower() == 'sequential_single_mode_randomized_benchmarking':
        experiment_started = True
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeSingleResonatorRandomizedBenchmarkingExperiment

        for i in arange(32):
            pulse_calibration(phase_exp=True)
            multimode_pulse_calibration()
            expt = MultimodeSingleResonatorRandomizedBenchmarkingExperiment(liveplot_enabled=False, path=datapath, trigger_period=0.001)
            expt.go()
            del expt
            gc.collect()

    if not experiment_started:
        close_match = difflib.get_close_matches(expt_name, expt_list)
        print "No experiment found for: " + expt_name
        print "Do you mean: " + close_match[0] + "?"



