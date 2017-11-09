__author__ = 'Nelson'




import numpy as np
def run_experiment(expt_name,lp_enable = False, **kwargs):
    import os
    import difflib

    expt_list = ['Vacuum_Rabi', 'CW_Drive', 'Pulse_Probe', 'Rabi', 'Histogram', 'T1', 'Ramsey', 'Spin_Echo', 'EF_Rabi', 'EF_Ramsey','EF_T1','HalfPiXOptimization','PiXOptimization','HalfPiYPhaseOptimization']
    datapath = os.getcwd() + '\data'
    expt = None

    if expt_name.lower() == 'vacuum_rabi':
        # from slab.experiments.Alex.General.VacuumRabiExperiment import VacuumRabiExperiment
        from slab.experiments.Alex.General.SingleQubitPulseSequenceExperiment import VacuumRabiExperiment
        # Do Vacuum Rabi
        expt = VacuumRabiExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    # if expt_name.lower() == 'cw_drive':
    #     from slab.experiments.Alex.General.CWDriveExperiment import CWDriveExperiment
    #     # Do CW drive experiment
    #     expt = CWDriveExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'pulse_probe':
        # from slab.experiments.Alex.General.PulseProbeExperiment import PulseProbeExperiment
        from slab.experiments.Alex.General.SingleQubitPulseSequenceExperiment import PulseProbeExperiment
        # Do pulse_probe experiment
        expt = PulseProbeExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'rabi':
        from slab.experiments.Alex.General.SingleQubitPulseSequenceExperiment import RabiExperiment
        # Do Rabi Experiment
        expt = RabiExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'rabi_thermalizer':
        from slab.experiments.Alex.General.SingleQubitPulseSequenceExperiment import RabiThermalizerExperiment
        # Do Rabi Experiment
        expt = RabiThermalizerExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    # if expt_name.lower() == 'histogram':
    #     from slab.experiments.Alex.General.HistogramExperiment import HistogramExperiment
    #     # Do Histogram Experiment
    #     expt = HistogramExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'histogram_rabi_thermalizer':
        from slab.experiments.Alex.General.HistogramHeteroExperiment import HistogramHeteroExperiment
        # Do Histogram Experiment
        expt = HistogramHeteroExperiment(path=datapath, prefix="HistogramRabiThermalizer",liveplot_enabled=lp_enable, **kwargs)

    if expt_name.lower() == 'histogram_hetero':
        from slab.experiments.Alex.General.HistogramHeteroExperiment import HistogramHeteroExperiment
        # Do Histogram Experiment
        expt = HistogramHeteroExperiment(path=datapath, liveplot_enabled=lp_enable, **kwargs)

    if expt_name.lower() == 't1':
        from slab.experiments.Alex.General.SingleQubitPulseSequenceExperiment import T1Experiment
        # Do T1 Experiment
        expt = T1Experiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'ramsey':
        from slab.experiments.Alex.General.SingleQubitPulseSequenceExperiment import RamseyExperiment
        # Do Ramsey Experiment
        expt = RamseyExperiment(path=datapath, liveplot_enabled = lp_enable, trigger_period=0.0002, **kwargs)

    if expt_name.lower() == 'spin_echo':
        from slab.experiments.Alex.General.SingleQubitPulseSequenceExperiment import SpinEchoExperiment
        # Do Ramsey Experiment
        expt = SpinEchoExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'ef_rabi':
        from slab.experiments.Alex.General.SingleQubitPulseSequenceExperiment import EFRabiExperiment
        # Do EF Rabi Experiment
        expt = EFRabiExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'ef_ramsey':
        from slab.experiments.Alex.General.SingleQubitPulseSequenceExperiment import EFRamseyExperiment
        # Do EF Ramsey Experiment
        expt = EFRamseyExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'ef_t1':
        from slab.experiments.Alex.General.SingleQubitPulseSequenceExperiment import EFT1Experiment
        # Do EF T1 Experiment
        expt = EFT1Experiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'halfpixoptimization':
        from slab.experiments.Alex.General.SingleQubitPulseSequenceExperiment import HalfPiXOptimizationExperiment
        # Do EF T1 Experiment
        expt = HalfPiXOptimizationExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'pixoptimization':
        from slab.experiments.Alex.General.SingleQubitPulseSequenceExperiment import PiXOptimizationExperiment
        # Do EF T1 Experiment
        expt = PiXOptimizationExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'halfpiyphaseoptimization':
        from slab.experiments.Alex.General.SingleQubitPulseSequenceExperiment import HalfPiYPhaseOptimizationExperiment
        # Do EF T1 Experiment
        expt = HalfPiYPhaseOptimizationExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'tomography':
        from slab.experiments.Alex.General.SingleQubitPulseSequenceExperiment import TomographyExperiment
        # Do EF T1 Experiment
        expt = TomographyExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)


    if expt_name.lower() == 'randomized_benchmarking':
        from slab.experiments.Alex.General.SingleQubitPulseSequenceExperiment import SingleQubitRandomizedBenchmarkingExperiment
        expt = SingleQubitRandomizedBenchmarkingExperiment(path=datapath, liveplot_enabled = lp_enable,trigger_period = 0.001, **kwargs)

    if expt_name.lower() == 'randomized_benchmarking_phase_offset':

        from slab.experiments.Alex.General.SingleQubitPulseSequenceExperiment import SingleQubitRandomizedBenchmarkingPhaseOffsetExperiment
        expt = SingleQubitRandomizedBenchmarkingPhaseOffsetExperiment(path=datapath, liveplot_enabled = lp_enable,trigger_period = 0.001, **kwargs)

    if expt_name.lower() == 'error_amplification':
        from slab.experiments.Alex.General.SingleQubitPulseSequenceExperiment import SingleQubitErrorAmplificationExperiment
        expt = SingleQubitErrorAmplificationExperiment(path=datapath, liveplot_enabled = lp_enable,trigger_period = 0.001, **kwargs)

    if expt_name.lower() == 'error_amplification_phase_offset':
        from slab.experiments.Alex.General.SingleQubitPulseSequenceExperiment import SingleQubitErrorAmplificationPhaseOffsetExperiment
        expt = SingleQubitErrorAmplificationPhaseOffsetExperiment(path=datapath, liveplot_enabled = lp_enable,trigger_period = 0.001, **kwargs)

    if expt_name.lower() == 'rabi_ramsey_t1_flux_sweep':

        from slab.experiments.Alex.General.SingleQubitPulseSequenceExperiment import RabiRamseyT1FluxSweepExperiment
        expt = RabiRamseyT1FluxSweepExperiment(path=datapath, liveplot_enabled=lp_enable,trigger_period=0.001, **kwargs)

    if expt != None:
        expt.go()
    else:
        print 'Aborted: No experiment found with name:', expt_name.lower()

    return expt