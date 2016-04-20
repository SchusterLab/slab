__author__ = 'Nelson'

from slab.instruments.Alazar import Alazar
from slab import *
import json
from numpy import*
import gc

liveplot_enabled = True


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

def frequency_stabilization():
    datapath = os.getcwd() + '\data'
    config_file = os.path.join(datapath, "..\\config" + ".json")
    with open(config_file, 'r') as fid:
        cfg_str = fid.read()

    cfg = AttrDict(json.loads(cfg_str))
    experiment_started = True
    from slab.experiments.General.SingleQubitPulseSequenceExperiment import RamseyExperiment
    expt = RamseyExperiment(path=datapath)
    expt.go()
    if (abs(expt.offset_freq) < 50e3):
        pass
    else:
        print expt.flux
        flux_offset = -expt.offset_freq/(expt.freq_flux_slope)
        print flux_offset
        if (abs(flux_offset) < 0.000002):
            flux2 = expt.flux + flux_offset
            print flux2
            expt = RamseyExperiment(path=datapath, flux = flux2)
            expt.go()
            offset_freq2 = expt.offset_freq
            flux_offset2 = -expt.offset_freq/(expt.freq_flux_slope)
            flux3 = flux2 + flux_offset2
            if (abs(offset_freq2) < 50e3):
                print "Great success! Frequency calibrated"
                expt.save_config()
            else:
                if (abs(flux_offset2) < 0.000002):
                    expt = RamseyExperiment(path=datapath, flux = flux3)
                    expt.go()
                    if (abs(expt.offset_freq) < 100e3):
                        print "Frequency calibrated"
                        expt.save_config()
                    else:
                        print "Try again: not converged after 2 tries"
                else:
                    print "Large change in flux is required; please do so manually"
                    pass
        else:
            print "Large change in flux is required; please do so manually"
            pass

def pulse_calibration(phase_exp=True):
    datapath = os.getcwd() + '\data'
    config_file = os.path.join(datapath, "..\\config" + ".json")
    with open(config_file, 'r') as fid:
        cfg_str = fid.read()

    lp_enable=False
    cfg = AttrDict(json.loads(cfg_str))
    experiment_started = True
    from slab.experiments.General.SingleQubitPulseSequenceExperiment import RamseyExperiment
    from slab.experiments.General.SingleQubitPulseSequenceExperiment import RabiExperiment
    from slab.experiments.General.SingleQubitPulseSequenceExperiment import HalfPiYPhaseOptimizationExperiment

    expt = RamseyExperiment(path=datapath,trigger_period = 0.0002, liveplot_enabled=lp_enable)
    expt.go()
    if (abs(expt.offset_freq) < 50e3):
        pass
    else:
        print expt.flux
        flux_offset = -expt.offset_freq/(expt.freq_flux_slope)
        print flux_offset
        if (abs(flux_offset) < 0.000002):
            flux2 = expt.flux + flux_offset
            print flux2
            expt = RamseyExperiment(path=datapath, flux = flux2,liveplot_enabled=lp_enable)
            expt.go()
            offset_freq2 = expt.offset_freq
            flux_offset2 = -expt.offset_freq/(expt.freq_flux_slope)
            flux3 = flux2 + flux_offset2
            if (abs(offset_freq2) < 50e3):
                print "Great success! Frequency calibrated"
                expt.save_config()
            else:
                if (abs(flux_offset2) < 0.000002):
                    expt = RamseyExperiment(path=datapath, flux = flux3,liveplot_enabled=lp_enable)
                    expt.go()
                    if (abs(expt.offset_freq) < 100e3):
                        print "Frequency calibrated"
                        expt.save_config()
                    else:
                        print "Try again: not converged after 2 tries"
                else:
                    print "Large change in flux is required; please do so manually"
                    pass
        else:
            print "Large change in flux is required; please do so manually"
            pass


    expt = RabiExperiment(path=datapath,liveplot_enabled=lp_enable)
    expt.go()
    print "ge pi and pi/2 pulses recalibrated"
    expt.save_config()
    if phase_exp:
        expt = HalfPiYPhaseOptimizationExperiment(path=datapath,liveplot_enabled=lp_enable)
        expt.go()
        print "Offset phase recalibrated"
        expt.save_config()
        del expt
        gc.collect()
    else:
        pass


def run_sequential_experiment(expt_name):
    import os
    import difflib

    expt_list = ['Frequency_Calibration','Rabi_Sweep','HalfPiXOptimization_sweep', 'tune_up_experiment']

    datapath = os.getcwd() + '\data'
    config_file = os.path.join(datapath, "..\\config" + ".json")
    with open(config_file, 'r') as fid:
        cfg_str = fid.read()

    cfg = AttrDict(json.loads(cfg_str))

    expt = None

    experiment_started = False

    if expt_name.lower() == 'calibration_experiment':
        experiment_started = True
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import RamseyExperiment
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import RabiExperiment
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import EFRabiExperiment
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import EFRamseyExperiment


        # Do Frequency Calibration

        expt = RamseyExperiment(path=datapath)
        expt.go()
        print expt.offset_freq

        expt = RabiExperiment(path=datapath)
        expt.go()

        expt = EFRabiExperiment(path=datapath)
        expt.go()

        expt = EFRamseyExperiment(path=datapath)
        expt.go()
        del expt
        gc.collect()

    if expt_name.lower() == 'monitor_frequency_experiment':
        experiment_started = True
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import RamseyExperiment
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import EFRamseyExperiment


        # Do Frequency Calibration

        for i in arange(20):
            expt = RamseyExperiment(path=datapath)
            expt.go()
            print expt.offset_freq


            expt = EFRamseyExperiment(path=datapath)
            expt.go()
            del expt
            gc.collect()

    if expt_name.lower() == 'sequential_error_amplification':
        experiment_started = True
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import SingleQubitErrorAmplificationExperiment
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import SingleQubitErrorAmplificationPhaseOffsetExperiment

        for i in arange(5):
            pulse_calibration(phase_exp=True)
            expt = SingleQubitErrorAmplificationPhaseOffsetExperiment(path=datapath, trigger_period=0.001, c0 = "half_pi",ci= "half_pi" )
            expt.go()
            expt.save_config()
            del expt
            gc.collect()
        for i in arange(5):
            pulse_calibration(phase_exp=True)
            expt = SingleQubitErrorAmplificationPhaseOffsetExperiment(path=datapath, trigger_period=0.001, c0 = "pi",ci= "half_pi" )
            expt.go()
            expt.save_config()
            del expt
            gc.collect()
        for i in arange(5):
            pulse_calibration(phase_exp=True)
            expt = SingleQubitErrorAmplificationPhaseOffsetExperiment(path=datapath, trigger_period=0.001, c0 = "pi",ci= "pi" )
            expt.go()
            expt.save_config()
            del expt
            gc.collect()
        for i in arange(5):
            pulse_calibration(phase_exp=True)
            expt = SingleQubitErrorAmplificationPhaseOffsetExperiment(path=datapath, trigger_period=0.001, c0 = "half_pi",ci= "pi" )
            expt.go()
            expt.save_config()
            del expt
            gc.collect()



    if expt_name.lower() == 'sequential_randomized_benchmarking':
        experiment_started = True
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import SingleQubitRandomizedBenchmarkingExperiment

        for i in arange(32):
            pulse_calibration(phase_exp=False)
            expt = SingleQubitRandomizedBenchmarkingExperiment(path=datapath, trigger_period=0.001)
            expt.go()
            del expt
            gc.collect()

    if expt_name.lower() == 'sequential_randomized_benchmarking_phase_offset':
        experiment_started = True
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import SingleQubitRandomizedBenchmarkingPhaseOffsetExperiment

        for i in arange(32):
            pulse_calibration(phase_exp=True)
            expt = SingleQubitRandomizedBenchmarkingPhaseOffsetExperiment(path=datapath, trigger_period=0.001)
            expt.go()
            del expt
            gc.collect()

    if expt_name.lower() == 'tomography_tune_up_experiment':
        experiment_started = True
        print expt_name + " is running!"
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import RamseyExperiment
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import RabiExperiment
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import EFRabiExperiment
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import EFRamseyExperiment
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import HalfPiYPhaseOptimizationExperiment


        # Do Frequency Calibration

        expt = RamseyExperiment(path=datapath)
        expt.go()
        if (abs(expt.offset_freq) < 50e3):
            pass
        else:
            print expt.flux
            flux_offset = -expt.offset_freq/(expt.freq_flux_slope)
            print flux_offset
            if (abs(flux_offset) < 0.000002):
                flux2 = expt.flux + flux_offset
                print flux2
                expt = RamseyExperiment(path=datapath, flux = flux2)
                expt.go()
                offset_freq2 = expt.offset_freq
                flux_offset2 = -expt.offset_freq/(expt.freq_flux_slope)
                flux3 = flux2 + flux_offset2
                if (abs(offset_freq2) < 50e3):
                    print "Great success! Frequency calibrated"
                    expt.save_config()
                else:
                    if (abs(flux_offset2) < 0.000002):
                        expt = RamseyExperiment(path=datapath, flux = flux3)
                        expt.go()
                        if (abs(expt.offset_freq) < 100e3):
                            print "Frequency calibrated"
                            expt.save_config()
                        else:
                            print "Try again: not converged after 2 tries"
                    else:
                        print "Large change in flux is required; please do so manually"
                        pass
            else:
                print "Large change in flux is required; please do so manually"
                pass


        expt = RabiExperiment(path=datapath)
        expt.go()
        print "ge pi and pi/2 pulses recalibrated"
        expt.save_config()

        expt = EFRabiExperiment(path=datapath)
        expt.go()
        print "ef pi and pi/2 pulses recalibrated"
        expt.save_config()


        expt = EFRamseyExperiment(path=datapath)
        expt.go()
        expt.save_config()
        expt.go()

        # #

        # expt = HalfPiYPhaseOptimizationExperiment(path=datapath)
        # expt.go()
        # #
        # print flux2
        del expt
        gc.collect()

    if expt_name.lower() == 'offset_phase_calibration_experiment':
        qubit_dc_offset_list_pi = array([0,0,0,1.7e6, 1.7e6,1.7e6,1.8e6,1.8e6,1.8e6,1.9e6,1.9e6,1.9e6,2e6,2e6,2e6])

        experiment_started = True
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import HalfPiYPhaseOptimizationExperiment
        # Do Frequency Calibration
        # frequency_stabilization()
        j=0
        for qubit_dc_offset in qubit_dc_offset_list_pi:
            if j > 0 and j%3 == 0:
                frequency_stabilization()

            expt = HalfPiYPhaseOptimizationExperiment(path=datapath, qubit_dc_offset = qubit_dc_offset)
            expt.go()
            expt.save_config()
            del expt
            gc.collect()
            j+=1

    if expt_name.lower() == 'frequency_calibration':
        frequency_stabilization()

    if expt_name.lower() == 'pulse_calibration':
        pulse_calibration(phase_exp=True)

    if expt_name.lower() == 'rabi_sweep':
        experiment_started = True
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import RabiSweepExperiment

        cfg['alazar']['samplesPerRecord'] = 2 ** (cfg['readout']['width'] - 1).bit_length()
        sequence_length = (cfg[expt_name.lower()]['stop']-cfg[expt_name.lower()]['start'])/cfg[expt_name.lower()]['step']
        if (cfg[expt_name.lower()]['use_pi_calibration']):
            sequence_length+=2

        cfg['alazar']['recordsPerBuffer'] = sequence_length
        cfg['alazar']['recordsPerAcquisition'] = int(
            sequence_length * min(cfg[expt_name.lower()]['averages'], 100))
        print "Prep Card"
        adc = Alazar(cfg['alazar'])

        drive_freq_pts = arange(cfg[expt_name.lower()]['freq_start'],cfg[expt_name.lower()]['freq_stop'],cfg[expt_name.lower()]['freq_step'])
        prefix = 'Rabi_Sweep'
        data_file =  os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))

        for ii, drive_freq in enumerate(drive_freq_pts):
            print drive_freq
            expt = RabiSweepExperiment(path=datapath,data_file=data_file, adc=adc, drive_freq=drive_freq, liveplot_enabled=False)
            expt.go()

            expt = None
            del expt

            gc.collect()
    if expt_name.lower() == 'rabi_ramsey_t1_flux_sweep':
        experiment_started = True
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import RabiRamseyT1FluxSweepExperiment



        prefix = 'rabi_ramsey_t1_flux_sweep'
        data_file =  os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))

        flux_pts = np.load(r'S:\_Data\160104 - 2D Multimode Qubit (Chip MM3, 11 modes)\iPython Notebooks\flux_sm.npy')
        drive_pts = np.load(r'S:\_Data\160104 - 2D Multimode Qubit (Chip MM3, 11 modes)\iPython Notebooks\qubit_frequencies_sm.npy')
        readout_pts = np.load(r'S:\_Data\160104 - 2D Multimode Qubit (Chip MM3, 11 modes)\iPython Notebooks\readout_frequencies_sm.npy')


        # flux_pts=[-0.012]
        # drive_pts=[4.844e9]
        # readout_pts=[5257100000.0]

        qubit_alpha = cfg['qubit']['alpha']

        for ii, flux in enumerate(flux_pts):
            drive_freq = drive_pts[ii]
            readout_freq = readout_pts[ii]
            print "Flux: " + str(flux)
            print "Drive frequency: " + str(drive_freq)
            print "Readout frequency: " + str(readout_freq)


            ### 1st Rabi
            print "### Running 1st Rabi."
            adc = prepare_alazar(cfg, 'rabi_ramsey_t1_flux_sweep','rabi')
            expt = RabiRamseyT1FluxSweepExperiment(path=datapath,exp = 'rabi', flux = flux,data_file=data_file, adc=adc,drive_freq=drive_freq, readout_freq = readout_freq,liveplot_enabled=liveplot_enabled)
            expt.go()
            pi_length = expt.pi_length
            half_pi_length = expt.half_pi_length

            adc.close()
            expt = None
            del expt
            gc.collect()

            ### 1st Ramsey
            print "### Running 1st Ramsey."
            adc = prepare_alazar(cfg, 'rabi_ramsey_t1_flux_sweep','ramsey')
            expt = RabiRamseyT1FluxSweepExperiment(path=datapath,exp = 'ramsey', flux = flux,half_pi_length=half_pi_length,data_file=data_file, adc=adc, drive_freq=drive_freq, readout_freq = readout_freq, liveplot_enabled=liveplot_enabled)
            expt.go()
            suggested_qubit_freq = expt.suggested_qubit_freq
            drive_freq = suggested_qubit_freq

            adc.close()
            expt = None
            del expt
            gc.collect()


            ### 2nd Rabi
            print "### Running 2nd Rabi."
            adc = prepare_alazar(cfg, 'rabi_ramsey_t1_flux_sweep','rabi')
            expt = RabiRamseyT1FluxSweepExperiment(path=datapath,exp = 'rabi', flux = flux,data_file=data_file, adc=adc, drive_freq=drive_freq, readout_freq = readout_freq, liveplot_enabled=liveplot_enabled)
            expt.go()
            pi_length = expt.pi_length
            half_pi_length = expt.half_pi_length

            adc.close()
            expt = None
            del expt
            gc.collect()


            ### Ramsey long
            print "### Running Ramsey Long."
            adc = prepare_alazar(cfg, 'rabi_ramsey_t1_flux_sweep','ramsey_long')
            expt = RabiRamseyT1FluxSweepExperiment(path=datapath,exp = 'ramsey_long', flux = flux,half_pi_length=half_pi_length,data_file=data_file, adc=adc, drive_freq=drive_freq, readout_freq = readout_freq, liveplot_enabled=liveplot_enabled)
            expt.go()

            adc.close()
            expt = None
            del expt
            gc.collect()

            ### Rabi Long
            print "### Running Rabi Long."
            adc = prepare_alazar(cfg, 'rabi_ramsey_t1_flux_sweep','rabi_long')
            expt = RabiRamseyT1FluxSweepExperiment(path=datapath,exp = 'rabi_long', flux = flux,data_file=data_file, adc=adc, drive_freq=drive_freq, readout_freq = readout_freq, liveplot_enabled=liveplot_enabled)
            expt.go()
            # pi_length = expt.pi_length
            # half_pi_length = expt.half_pi_length

            adc.close()
            expt = None
            del expt
            gc.collect()

            ### remove adc
            adc = None
            del adc


            ### t1
            print "### Running T1."
            adc = prepare_alazar(cfg, 'rabi_ramsey_t1_flux_sweep', 't1')
            expt = RabiRamseyT1FluxSweepExperiment(path=datapath,exp = 't1', flux = flux,pi_length=pi_length,data_file=data_file, adc=adc, drive_freq=drive_freq, readout_freq = readout_freq, liveplot_enabled=liveplot_enabled)
            expt.go()

            adc.close()
            expt = None
            del expt
            gc.collect()


            ### Pi/2 phase sweep
            print "### Running pi/2 phase sweep experiment"
            adc = prepare_alazar(cfg, 'rabi_ramsey_t1_flux_sweep','half_pi_phase_sweep')
            expt = RabiRamseyT1FluxSweepExperiment(path=datapath,exp = 'half_pi_phase_sweep', flux = flux,half_pi_length=half_pi_length,data_file=data_file, adc=adc,drive_freq=drive_freq, readout_freq = readout_freq,liveplot_enabled=liveplot_enabled)
            expt.go()

            adc.close()
            expt = None
            del expt
            gc.collect()


            ### EF Rabi
            print "### Running EF Rabi."
            adc = prepare_alazar(cfg, 'rabi_ramsey_t1_flux_sweep', 'ef_rabi')
            expt = RabiRamseyT1FluxSweepExperiment(path=datapath,exp = 'ef_rabi', flux = flux,pi_length=pi_length,data_file=data_file, adc=adc, drive_freq=drive_freq, readout_freq = readout_freq, liveplot_enabled=liveplot_enabled, alpha=qubit_alpha)
            expt.go()
            ef_pi_length = expt.pi_length
            ef_half_pi_length = expt.half_pi_length

            adc.close()
            expt = None
            del expt
            gc.collect()

            ### EF Ramsey
            print "### Running EF Ramsey."
            adc = prepare_alazar(cfg, 'rabi_ramsey_t1_flux_sweep', 'ef_ramsey')
            expt = RabiRamseyT1FluxSweepExperiment(path=datapath,exp = 'ef_ramsey', flux = flux,pi_length=pi_length,ef_half_pi_length=ef_half_pi_length,data_file=data_file, adc=adc, drive_freq=drive_freq, readout_freq = readout_freq, liveplot_enabled=liveplot_enabled, alpha=qubit_alpha)
            expt.go()
            suggested_alpha_freq = expt.suggested_qubit_alpha
            print "Suggested alpha frequency: " + str(suggested_alpha_freq)
            qubit_alpha = suggested_alpha_freq

            adc.close()
            expt = None
            del expt
            gc.collect()

            ### EF Rabi
            print "### Running EF Rabi."
            adc = prepare_alazar(cfg, 'rabi_ramsey_t1_flux_sweep', 'ef_rabi')
            expt = RabiRamseyT1FluxSweepExperiment(path=datapath,exp = 'ef_rabi', flux = flux,pi_length=pi_length,data_file=data_file, adc=adc, drive_freq=drive_freq, readout_freq = readout_freq, liveplot_enabled=liveplot_enabled, alpha=qubit_alpha)
            expt.go()
            ef_pi_length = expt.pi_length
            ef_half_pi_length = expt.half_pi_length

            adc.close()
            expt = None
            del expt
            gc.collect()


            ### EF T1
            print "### Running EF T1."
            adc = prepare_alazar(cfg, 'rabi_ramsey_t1_flux_sweep', 'ef_t1')
            expt = RabiRamseyT1FluxSweepExperiment(path=datapath,exp = 'ef_t1', flux = flux,pi_length=pi_length,ef_pi_length=ef_pi_length,data_file=data_file, adc=adc, drive_freq=drive_freq, readout_freq = readout_freq, liveplot_enabled=liveplot_enabled, alpha=qubit_alpha)
            expt.go()

            adc.close()
            expt = None
            del expt
            gc.collect()


            ## EF Ramsey Long
            print "### Running EF Ramsey."
            adc = prepare_alazar(cfg, 'rabi_ramsey_t1_flux_sweep', 'ef_ramsey_long')
            expt = RabiRamseyT1FluxSweepExperiment(path=datapath,exp = 'ef_ramsey_long', flux = flux,pi_length=pi_length,ef_half_pi_length=ef_half_pi_length,data_file=data_file, adc=adc, drive_freq=drive_freq, readout_freq = readout_freq, liveplot_enabled=liveplot_enabled, alpha=qubit_alpha)
            expt.go()
            suggested_alpha_freq = expt.suggested_qubit_alpha
            print "Suggested alpha frequency: " + str(suggested_alpha_freq)
            qubit_alpha = suggested_alpha_freq

            adc.close()
            expt = None
            del expt
            gc.collect()

            ### remove adc
            adc = None
            del adc

            gc.collect()

    if expt_name.lower() == 'halfpixoptimization_sweep':

        experiment_started = True
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import HalfPiXOptimizationSweepExperiment

        adc = prepare_alazar(cfg, expt_name)

        if cfg[expt_name.lower()]['sweep_param'] == "length":
            pulse_sweep_pts = arange(cfg[expt_name.lower()]['length_start'],cfg[expt_name.lower()]['length_stop'],cfg[expt_name.lower()]['length_step'])
            pulse_amp = cfg[expt_name.lower()]['amp']
        elif cfg[expt_name.lower()]['sweep_param'] == "amp":
            pulse_sweep_pts = arange(cfg[expt_name.lower()]['amp_start'],cfg[expt_name.lower()]['amp_stop'],cfg[expt_name.lower()]['amp_step'])
            pulse_length = cfg[expt_name.lower()]['length']

        prefix = 'HalfPiXOptimization_Sweep'
        data_file =  os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))

        if cfg[expt_name.lower()]['sweep_param'] == "length":
            for ii, pulse_sweep in enumerate(pulse_sweep_pts):
                print "Pulse Sweep: " + str(pulse_sweep)
                expt = HalfPiXOptimizationSweepExperiment(path=datapath,data_file=data_file, adc=adc, pulse_length=pulse_sweep, pulse_amp = pulse_amp, liveplot_enabled = False)
                expt.go()

                expt = None
                del expt

                gc.collect()
        elif cfg[expt_name.lower()]['sweep_param'] == "amp":
            for ii, pulse_sweep in enumerate(pulse_sweep_pts):
                print "Pulse Sweep: " + str(pulse_sweep)
                expt = HalfPiXOptimizationSweepExperiment(path=datapath,data_file=data_file, adc=adc, pulse_length=pulse_length, pulse_amp = pulse_sweep, liveplot_enabled = False)
                expt.go()

                expt = None
                del expt

                gc.collect()


    if expt_name.lower() == 'pixoptimization_sweep':
        experiment_started = True
        from slab.experiments.General.SingleQubitPulseSequenceExperiment import PiXOptimizationSweepExperiment

        # cfg['alazar']['samplesPerRecord'] = 2 ** (cfg['readout']['width'] - 1).bit_length()
        # sequence_length = (cfg[expt_name.lower()]['stop']-cfg[expt_name.lower()]['start'])/cfg[expt_name.lower()]['step']
        # if (cfg[expt_name.lower()]['use_pi_calibration']):
        #     sequence_length+=2
        #
        # cfg['alazar']['recordsPerBuffer'] = sequence_length
        # cfg['alazar']['recordsPerAcquisition'] = int(
        #     sequence_length * min(cfg[expt_name.lower()]['averages'], 100))
        # print "Prep Card"
        # adc = Alazar(cfg['alazar'])

        adc = prepare_alazar(cfg, expt_name)

        if cfg[expt_name.lower()]['sweep_param'] == "length":
            pulse_sweep_pts = arange(cfg[expt_name.lower()]['length_start'],cfg[expt_name.lower()]['length_stop'],cfg[expt_name.lower()]['length_step'])
            pulse_amp = cfg[expt_name.lower()]['amp']
        elif cfg[expt_name.lower()]['sweep_param'] == "amp":
            pulse_sweep_pts = arange(cfg[expt_name.lower()]['amp_start'],cfg[expt_name.lower()]['amp_stop'],cfg[expt_name.lower()]['amp_step'])
            pulse_length = cfg[expt_name.lower()]['length']

        prefix = 'PiXOptimization_Sweep'
        data_file =  os.path.join(datapath, get_next_filename(datapath, prefix, suffix='.h5'))

        # for ii, pulse_sweep in enumerate(pulse_sweep_pts):
        #     print "Pulse Sweep: " + str(pulse_sweep)
        #     expt = PiXOptimizationSweepExperiment(path=datapath,data_file=data_file, adc=adc, pulse_length=pulse_sweep, liveplot_enabled = False)
        #     expt.go()
        #
        #     expt = None
        #     del expt
        #
        #     gc.collect()

        if cfg[expt_name.lower()]['sweep_param'] == "length":
            for ii, pulse_sweep in enumerate(pulse_sweep_pts):
                print "Pulse Sweep: " + str(pulse_sweep)
                expt = PiXOptimizationSweepExperiment(path=datapath,data_file=data_file, adc=adc, pulse_length=pulse_sweep, pulse_amp = pulse_amp, liveplot_enabled = False)
                expt.go()

                expt = None
                del expt

                gc.collect()
        elif cfg[expt_name.lower()]['sweep_param'] == "amp":
            for ii, pulse_sweep in enumerate(pulse_sweep_pts):
                print "Pulse Sweep: " + str(pulse_sweep)
                expt = PiXOptimizationSweepExperiment(path=datapath,data_file=data_file, adc=adc, pulse_length=pulse_length, pulse_amp = pulse_sweep, liveplot_enabled = False)
                expt.go()

                expt = None
                del expt

                gc.collect()


    if not experiment_started:
        close_match = difflib.get_close_matches(expt_name, expt_list)
        print "No experiment found for: " + expt_name
        print "Do you mean: " + close_match[0] + "?"

