__author__ = 'Nelson'




import numpy as np
def run_experiment(expt_name,lp_enable = True, **kwargs):
    import os
    import difflib

    expt_list = ['Vacuum_Rabi', 'CW_Drive', 'Pulse_Probe', 'Rabi', 'Histogram', 'T1', 'Ramsey', 'Spin_Echo', 'EF_Rabi', 'EF_Ramsey','EF_T1','HalfPiXOptimization','PiXOptimization','HalfPiYPhaseOptimization']
    datapath = os.getcwd() + '\data'
    expt = None

    if expt_name.lower() == 'vacuum_rabi':
        from slab.experiments.Nitrogen.General.VacuumRabiExperiment import VacuumRabiExperiment
        # Do Vacuum Rabi
        expt = VacuumRabiExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'cw_drive':
        from slab.experiments.Nitrogen.General.CWDriveExperiment import CWDriveExperiment
        # Do CW drive experiment
        expt = CWDriveExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'pulse_probe':
        from slab.experiments.Nitrogen.General.PulseProbeExperiment import PulseProbeExperiment
        # Do CW drive experiment
        expt = PulseProbeExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'pulse_probe_iq':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import PulseProbeIQExperiment
        # Do CW drive experiment
        expt = PulseProbeIQExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'rabi':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import RabiExperiment
        # Do Rabi Experiment
        expt = RabiExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'rabi_sweep':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import RabiSweepExperiment
        # Do Rabi Experiment
        expt = RabiSweepExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'ef_rabi_sweep':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import EFRabiSweepExperiment
        # Do Rabi Experiment
        expt = EFRabiSweepExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'histogram':
        from slab.experiments.Nitrogen.General.HistogramExperiment import HistogramExperiment
        # Do Histogram Experiment
        expt = HistogramExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 't1':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import T1Experiment
        # Do T1 Experiment
        expt = T1Experiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 't1rho':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import T1rhoExperiment
        # Do T1 Experiment
        expt = T1rhoExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'ramsey':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import RamseyExperiment
        # Do Ramsey Experiment
        expt = RamseyExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'spin_echo':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import SpinEchoExperiment
        # Do Ramsey Experiment
        expt = SpinEchoExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'ef_rabi':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import EFRabiExperiment
        # Do EF Rabi Experiment
        expt = EFRabiExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'ef_ramsey':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import EFRamseyExperiment
        # Do EF Ramsey Experiment
        expt = EFRamseyExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'ef_t1':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import EFT1Experiment
        # Do EF T1 Experiment
        expt = EFT1Experiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'halfpixoptimization':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import HalfPiXOptimizationExperiment
        # Do EF T1 Experiment
        expt = HalfPiXOptimizationExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'pixoptimization':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import PiXOptimizationExperiment
        # Do EF T1 Experiment
        expt = PiXOptimizationExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'halfpiyphaseoptimization':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import HalfPiYPhaseOptimizationExperiment
        # Do EF T1 Experiment
        expt = HalfPiYPhaseOptimizationExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'efpulsephaseoptimization':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import EFPulsePhaseOptimizationExperiment
        # Do EF T1 Experiment
        expt = EFPulsePhaseOptimizationExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'tomography':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import TomographyExperiment
        # Do EF T1 Experiment
        expt = TomographyExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'rabi_ramsey_t1_flux_sweep':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import RabiRamseyT1FluxSweepExperiment
        # Do EF T1 Experiment
        expt = RabiRamseyT1FluxSweepExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'randomized_benchmarking':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import SingleQubitRandomizedBenchmarkingExperiment
        expt = SingleQubitRandomizedBenchmarkingExperiment(path=datapath, liveplot_enabled = lp_enable,trigger_period = 0.001, **kwargs)

    if expt_name.lower() == 'randomized_benchmarking_phase_offset':

        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import SingleQubitRandomizedBenchmarkingPhaseOffsetExperiment
        expt = SingleQubitRandomizedBenchmarkingPhaseOffsetExperiment(path=datapath, liveplot_enabled = lp_enable,trigger_period = 0.001, **kwargs)

    if expt_name.lower() == 'error_amplification':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import SingleQubitErrorAmplificationExperiment
        expt = SingleQubitErrorAmplificationExperiment(path=datapath, liveplot_enabled = lp_enable,trigger_period = 0.001, **kwargs)

    if expt_name.lower() == 'error_amplification_phase_offset':
        from slab.experiments.Nitrogen.General.SingleQubitPulseSequenceExperiment import SingleQubitErrorAmplificationPhaseOffsetExperiment
        expt = SingleQubitErrorAmplificationPhaseOffsetExperiment(path=datapath, liveplot_enabled = lp_enable,trigger_period = 0.001, **kwargs)

    if expt != None:
        expt.go()

    return expt