__author__ = 'Nelson'


def run_multimode_experiment(expt_name, lp_enable = True, **kwargs):
    import os
    import difflib

    expt_list=['Multimode_Rabi','Multimode_T1','Multimode_EF_Rabi','Multimode_Ramsey','Multimode_EF_Rabi', \
               'Multimode_EF_Ramsey','Multimode_Entanglement','Multimode_CPhase_Experiment','multimode_Pi_Pi_experiment','multimode_ef_rabi_sweep']
    datapath=os.getcwd()+'\data'
    expt = None
    from slab.experiments.Multimode.MultimodePulseSequenceExperiment import *
    if expt_name.lower()=='multimode_rabi':

        #Do Multimode Rabi
        expt=MultimodeRabiExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower() == 'multimode_vacuum_rabi':
        from slab.experiments.General.VacuumRabiExperimentPSB import MultimodeVacuumRabiExperiment
        # Do Vacuum Rabi
        expt = MultimodeVacuumRabiExperiment(path=datapath, liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_rabi_sweep':
        #Multimode ef Rabi sweep experiment
        expt=MultimodeRabiSweepExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)


    if expt_name.lower()=='multimode_rabi_line_cut_sweep':
        #Multimode ef Rabi sweep experiment
        expt=MultimodeRabiLineCutSweepExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)



    if expt_name.lower()=='multimode_bluesideband_sweep':
        #Multimode Blue sideband sweep experiment
        expt=MultimodeBlueSidebandSweepExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_echo_sideband_experiment':
        #Multimode ef Rabi sweep experiment
        expt=MultimodeEchoSidebandExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_ef_rabi':
        #Do Multimode Rabi
        expt=MultimodeEFRabiExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_ef_rabi_sweep':
        #Multimode ef Rabi sweep experiment
        expt=MultimodeEFRabiSweepExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_charge_sideband_rabi_sweep':
        #Multimode ef Rabi sweep experiment
        expt=MultimodeChargeSidebandRabiSweepExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_ramsey':
        #Do Multimode Rabi
        expt=MultimodeRamseyExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_ef_ramsey':
        #Do Multimode Rabi
        expt=MultimodeEFRamseyExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_t1':
        expt=MultimodeT1Experiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_entanglement':
        expt=MultimodeEntanglementExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_general_entanglement':
        expt=MultimodeGeneralEntanglementExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_cphase_tests_experiment':
        expt=MultimodeCPhaseTestsExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_cphase_experiment':
        expt=MultimodeCPhaseExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_cnot_experiment':
        expt=MultimodeCNOTExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_pi_pi_experiment':
        expt=MultimodePi_PiExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_ef_pi_pi_experiment':
        expt=Multimode_ef_Pi_PiExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_qubit_mode_cz_offset_experiment':
        expt=Multimode_Qubit_Mode_CZ_Offset_Experiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_qubit_mode_cz_v2_offset_experiment':
        expt=Multimode_Qubit_Mode_CZ_V2_Offset_Experiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_mode_mode_cz_v2_offset_experiment':
        expt=Multimode_Mode_Mode_CZ_V2_Offset_Experiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_mode_mode_cz_v3_offset_experiment':
        expt=Multimode_Mode_Mode_CZ_V3_Offset_Experiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_mode_mode_cnot_v2_offset_experiment':
        expt=Multimode_Mode_Mode_CNOT_V2_Offset_Experiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_mode_mode_cnot_v3_offset_experiment':
        expt=Multimode_Mode_Mode_CNOT_V3_Offset_Experiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_ac_stark_shift_experiment':
        expt=Multimode_AC_Stark_Shift_Offset_Experiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_dc_offset_experiment':
        expt=MultimodeDCOffsetExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_state_dep_shift':
        expt=Multimode_State_Dep_Shift_Experiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_ef_2pi_experiment':
        expt=Multimode_ef_2pi_Experiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_single_resonator_tomography':
        expt=MultimodeSingleResonatorTomography(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_two_resonator_tomography':
        expt=MultimodeTwoResonatorTomography(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_two_resonator_tomography_phase_sweep':
        expt=MultimodeTwoResonatorTomographyPhaseSweepExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_ghz_entanglement_witness':
        expt=MultimodeGHZEntanglementWitnessExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_process_tomography_phase_sweep_new':
        expt=MultimodeProcessTomographyPhaseSweepExperiment_1(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_process_tomography_gate_fid_expt':
        expt=MultimodeProcessTomographyPhaseSweepExperiment_2(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_process_tomography_2':
        expt=MultimodeProcessTomographyExperiment_2(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_process_tomography_phase_sweep_test':
        expt=MultimodeProcessTomographyPhaseSweepExperiment_test(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_three_mode_correlation_experiment':
        expt=MultimodeThreeModeCorrelationExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='single_mode_rb':
        expt=MultimodeSingleResonatorRandomizedBenchmarkingExperiment(path=datapath,liveplot_enabled = lp_enable,trigger_period = 0.001, **kwargs)

    if expt_name.lower()=='qubit_mode_rb':
        expt=MultimodeQubitModeRandomizedBenchmarkingExperiment(path=datapath,liveplot_enabled = lp_enable,trigger_period = 0.001, **kwargs)

    if expt_name.lower()=='multimode_cphase_amplification':
        expt=MultimodeCPhaseAmplificationExperiment(path=datapath,liveplot_enabled = lp_enable,trigger_period = 0.0002, **kwargs)

    if expt_name.lower()=='multimode_cnot_amplification':
        expt=MultimodeCNOTAmplificationExperiment(path=datapath,liveplot_enabled = lp_enable,trigger_period = 0.0002, **kwargs)

    if expt_name.lower()=='multimode_calibrate_ef_sideband':
        expt=MultimodeCalibrateEFSidebandExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_calibrate_offset':
        expt=MultimodeCalibrateOffsetExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_qubit_mode_cross_kerr':
        expt=MultimodeQubitModeCrossKerrExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)

    if expt_name.lower()=='multimode_pulse_probe_iq':
        #Multimode ef Rabi sweep experiment
        expt=MultimodePulseProbeIQExperiment(path=datapath,liveplot_enabled = lp_enable, **kwargs)


    if expt !=None:
        expt.go()

    return expt
