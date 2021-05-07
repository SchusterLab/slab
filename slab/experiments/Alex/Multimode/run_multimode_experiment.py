__author__ = 'Nelson'


def run_multimode_experiment(expt_name, lp_enable = True, **kwargs):
    import os
    import difflib

    expt_list=['Multimode_Rabi','Multimode_T1','Multimode_EF_Rabi','Multimode_Ramsey','Multimode_EF_Rabi', \
               'Multimode_EF_Ramsey','Multimode_Entanglement','Multimode_CPhase_Experiment','multimode_Pi_Pi_experiment']
    datapath=os.getcwd()+'\data'
    expt = None

    if expt_name.lower()=='multimode_rabi':
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeRabiExperiment
        #Do Multimode Rabi
        expt=MultimodeRabiExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_ef_rabi':
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeEFRabiExperiment
        #Do Multimode Rabi
        expt=MultimodeEFRabiExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_ramsey':
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeRamseyExperiment
        #Do Multimode Rabi
        expt=MultimodeRamseyExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_ef_ramsey':
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeEFRamseyExperiment
        #Do Multimode Rabi
        expt=MultimodeEFRamseyExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)


    if expt_name.lower()=='multimode_t1':
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeT1Experiment
        expt=MultimodeT1Experiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_entanglement':
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeEntanglementExperiment
        expt=MultimodeEntanglementExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_cphase_tests_experiment':
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeCPhaseTestsExperiment
        expt=MultimodeCPhaseTestsExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_cphase_experiment':
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeCPhaseExperiment
        expt=MultimodeCPhaseExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_cnot_experiment':
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeCNOTExperiment
        expt=MultimodeCNOTExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_pi_pi_experiment':
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodePi_PiExperiment
        expt=MultimodePi_PiExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_single_resonator_tomography':
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeSingleResonatorTomography
        expt=MultimodeSingleResonatorTomography(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_two_resonator_tomography':
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeTwoResonatorTomography
        expt=MultimodeTwoResonatorTomography(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_three_mode_correlation_experiment':
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeThreeModeCorrelationExperiment
        expt=MultimodeThreeModeCorrelationExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='single_mode_rb':
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeSingleResonatorRandomizedBenchmarkingExperiment
        expt=MultimodeSingleResonatorRandomizedBenchmarkingExperiment(path=datapath,liveplot_enabled = lp_enable,trigger_period = 0.001, **kwargs)

    if expt_name.lower()=='multimode_cphase_amplification':
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeCPhaseAmplificationExperiment
        expt=MultimodeCPhaseAmplificationExperiment(path=datapath,liveplot_enabled = lp_enable,trigger_period = 0.0002, **kwargs)

    if expt_name.lower()=='multimode_calibrate_ef_sideband':
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeCalibrateEFSidebandExperiment
        expt=MultimodeCalibrateEFSidebandExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)


    if expt_name.lower()=='multimode_calibrate_offset':
        from slab.experiments.Multimode.MultimodePulseSequenceExperiment import MultimodeCalibrateOffsetExperiment
        expt=MultimodeCalibrateOffsetExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt !=None:
        expt.go()

    return expt
