__author__ = 'Nelson'

from slab import *
from slab.instruments.Alazar import Alazar
from slab.experiments.ExpLib.QubitPulseSequenceExperiment import *
from numpy import mean, arange


class MultimodeRabiExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_Rabi', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeRabiSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        pass

class MultimodeEFRabiExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_EF_Rabi', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeEFRabiSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        pass

class MultimodeRamseyExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_Ramsey', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeRamseySequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        print "Analyzing Ramsey Data"
        fitdata = fitdecaysin(expt_pts, expt_avg_data)

        self.offset_freq =self.cfg['multimode_ramsey']['ramsey_freq'] - fitdata[1] * 1e9

        suggested_offset_freq = self.cfg['multimodes'][int(self.cfg['multimode_ramsey']['id'])]['dc_offset_freq'] - (fitdata[1] * 1e9 - self.cfg['multimode_ramsey']['ramsey_freq'])
        print "Suggested offset frequency: " + str(suggested_offset_freq)
        print "Oscillation frequency: " + str(fitdata[1] * 1e3) + " MHz"
        print "T2*: " + str(fitdata[3]) + " ns"


class MultimodeDCOffsetExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_DC_Offset_Experiment', config_file='..\\config.json', **kwargs):
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        self.amp = self.extra_args['amp']
        self.freq = self.extra_args['freq']
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeDCOffsetSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        print "Analyzing Ramsey Data"
        fitdata = fitdecaysin(expt_pts, expt_avg_data)

        self.offset_freq =self.cfg['multimode_dc_offset_experiment']['ramsey_freq'] - fitdata[1] * 1e9

        print "Flux drive amplitude: %s" %(self.amp)
        print "Offset frequency: " + str(self.offset_freq)
        print "T2*: " + str(fitdata[3]) + " ns"



class MultimodeCalibrateOffsetExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_Calibrate_Offset_Experiment', config_file='..\\config.json', **kwargs):
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        self.exp = self.extra_args['exp']
        self.mode = self.extra_args['mode']
        self.dc_offset_guess =  self.extra_args['dc_offset_guess']

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeCalibrateOffsetSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):

        if self.cfg['multimode_calibrate_offset_experiment']['calibrate_sidebands']:
            print "Analyzing Multimode Rabi Data"
            fitdata = fitdecaysin(expt_pts[5:], expt_avg_data[5:])
            fitdata = fitdecaysin(expt_pts[5:], expt_avg_data[5:])

            if (-fitdata[2]%180 - 90)/(360*fitdata[1]) < 0:
                self.flux_pi_length = (-fitdata[2]%180 + 90)/(360*fitdata[1])
                self.flux_2pi_length = (-fitdata[2]%180 + 270)/(360*fitdata[1])
                print "Flux pi length ge =" + str(self.flux_pi_length)
                print "Flux 2pi length ge =" + str(self.flux_2pi_length)
                if self.cfg['multimode_calibrate_offset_experiment'][self.exp]['save_to_file']:
                    print "writing into config file"
                    self.cfg['multimodes'][self.mode]['flux_pi_length'] =   self.flux_pi_length
                    self.cfg['multimodes'][self.mode]['flux_2pi_length'] =  self.flux_2pi_length
            else:
                self.flux_pi_length = (-fitdata[2]%180 - 90)/(360*fitdata[1])
                self.flux_2pi_length = (-fitdata[2]%180 + 90)/(360*fitdata[1])

                print "Flux pi length ge =" + str(self.flux_pi_length)
                print "Flux 2pi length ge =" + str(self.flux_2pi_length)
                if self.cfg['multimode_calibrate_offset_experiment'][self.exp]['save_to_file']:
                    print "writing into config file"
                    self.cfg['multimodes'][self.mode]['flux_pi_length'] =   self.flux_pi_length
                    self.cfg['multimodes'][self.mode]['flux_2pi_length'] =  self.flux_2pi_length

        elif self.cfg['multimode_calibrate_ef_sideband_experiment']['calibrate_offsets']:


            print "Analyzing Ramsey Data"
            fitdata = fitdecaysin(expt_pts, expt_avg_data)

            self.offset_freq =self.cfg['multimode_calibrate_offset_experiment'][self.exp]['ramsey_freq'] - fitdata[1] * 1e9
            self.suggested_dc_offset_freq = self.dc_offset_guess + self.offset_freq
            # self.suggested_dc_offset_freq = self.offset - (fitdata[1] * 1e9 - self.cfg['multimode_ramsey']['ramsey_freq'])
            print "Suggested offset frequency: " + str(self.suggested_dc_offset_freq)
            print "Oscillation frequency: " + str(fitdata[1] * 1e3) + " MHz"
            print "T2*: " + str(fitdata[3]) + " ns"
            print self.cfg['multimode_calibrate_offset_experiment'][self.exp]
            if self.cfg['multimode_calibrate_offset_experiment'][self.exp]['save_to_file']:
                print "Saving DC offset to config for mode " + str(self.mode)
                print self.cfg['multimodes'][self.mode]['dc_offset_freq']
                self.cfg['multimodes'][self.mode]['dc_offset_freq'] = self.suggested_dc_offset_freq
                print self.cfg['multimodes'][self.mode]['dc_offset_freq']


class MultimodeCalibrateEFSidebandExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_Calibrate_EF_Sideband_experiment', config_file='..\\config.json', **kwargs):
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        self.exp = self.extra_args['exp']
        self.mode = self.extra_args['mode']
        self.dc_offset_guess_ef =  self.extra_args['dc_offset_guess_ef']

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeCalibrateEFSidebandSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):

        if self.cfg['multimode_calibrate_ef_sideband_experiment']['calibrate_sidebands']:
            print "Analyzing EF Rabi Data"
            fitdata = fitdecaysin(expt_pts[5:], expt_avg_data[5:])

            if (-fitdata[2]%180 - 90)/(360*fitdata[1]) < 0:
                self.flux_pi_length_ef = (-fitdata[2]%180 + 90)/(360*fitdata[1])
                self.flux_2pi_length_ef = (-fitdata[2]%180 + 270)/(360*fitdata[1])
                print "Flux pi length EF =" + str(self.flux_pi_length_ef)
                print "Flux 2pi length EF =" + str(self.flux_2pi_length_ef)
                if self.cfg['multimode_calibrate_ef_sideband_experiment'][self.exp]['save_to_file']:
                    self.cfg['multimodes'][self.mode]['flux_pi_length_ef'] =   self.flux_pi_length_ef
                    self.cfg['multimodes'][self.mode]['flux_2pi_length_ef'] =  self.flux_2pi_length_ef
            else:
                self.flux_pi_length_ef = (-fitdata[2]%180 - 90)/(360*fitdata[1])
                self.flux_2pi_length_ef = (-fitdata[2]%180 + 90)/(360*fitdata[1])

                print "Flux pi length EF =" + str(self.flux_pi_length_ef)
                print "Flux 2pi length EF =" + str(self.flux_2pi_length_ef)
                if self.cfg['multimode_calibrate_ef_sideband_experiment'][self.exp]['save_to_file']:
                    self.cfg['multimodes'][self.mode]['flux_pi_length_ef'] =   self.flux_pi_length_ef
                    self.cfg['multimodes'][self.mode]['flux_2pi_length_ef'] =  self.flux_2pi_length_ef

        elif self.cfg['multimode_calibrate_ef_sideband_experiment']['calibrate_offsets']:

            print "Analyzing Ramsey Data"
            fitdata = fitdecaysin(expt_pts, expt_avg_data)
            self.offset_freq =self.cfg['multimode_calibrate_ef_sideband_experiment'][self.exp]['ramsey_freq'] - fitdata[1] * 1e9
            self.suggested_dc_offset_freq_ef = self.dc_offset_guess_ef + self.offset_freq
            # self.suggested_dc_offset_freq = self.offset - (fitdata[1] * 1e9 - self.cfg['multimode_ramsey']['ramsey_freq'])
            print "Suggested offset frequency: " + str(self.suggested_dc_offset_freq_ef)
            print "Oscillation frequency: " + str(fitdata[1] * 1e3) + " MHz"
            print "T2: " + str(fitdata[3]) + " ns"
            print self.cfg['multimode_calibrate_ef_sideband_experiment'][self.exp]
            if self.cfg['multimode_calibrate_ef_sideband_experiment'][self.exp]['save_to_file']:
                print "Saving ef DC offset to config for mode " + str(self.mode)
                print self.cfg['multimodes'][self.mode]['dc_offset_freq_ef']
                self.cfg['multimodes'][self.mode]['dc_offset_freq_ef'] = self.suggested_dc_offset_freq_ef
                print self.cfg['multimodes'][self.mode]['dc_offset_freq_ef']


class MultimodeEFRamseyExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_EF_Ramsey', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeEFRamseySequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):

        print "Analyzing ef Ramsey Data"
        fitdata = fitdecaysin(expt_pts, expt_avg_data)

        self.offset_freq =self.cfg['multimode_ef_ramsey']['ramsey_freq'] - fitdata[1] * 1e9

        suggested_offset_freq = self.cfg['multimodes'][int(self.cfg['multimode_ef_ramsey']['id'])]['dc_offset_freq_ef'] - (fitdata[1] * 1e9 - self.cfg['multimode_ef_ramsey']['ramsey_freq'])
        print "Suggested offset frequency: " + str(suggested_offset_freq)
        print "Oscillation frequency: " + str(fitdata[1] * 1e3) + " MHz"
        print "T2*: " + str(fitdata[3]) + " ns"

class MultimodeRabiSweepExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_Rabi_Sweep', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeRabiSweepSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)



    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        #print self.data_file
        slab_file = SlabFile(self.data_file)
        with slab_file as f:
            f.append_pt('flux_freq', self.flux_freq)
            f.append_line('sweep_expt_avg_data', expt_avg_data)
            f.append_line('sweep_expt_pts', expt_pts)

            f.close()


class MultimodeEFRabiSweepExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_EF_Rabi_Sweep', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeEFRabiSweepSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)



    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        #print self.data_file
        slab_file = SlabFile(self.data_file)
        with slab_file as f:
            f.append_pt('flux_freq', self.flux_freq)
            f.append_line('sweep_expt_avg_data', expt_avg_data)
            f.append_line('sweep_expt_pts', expt_pts)

            f.close()


class MultimodeT1Experiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_T1', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeT1Sequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        pass


class MultimodeEntanglementExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_Entanglement', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeEntanglementSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        pass

class MultimodeGeneralEntanglementExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_General_Entanglement', config_file='..\\config.json', **kwargs):
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        self.id1 = self.extra_args['id1']
        self.id2 = self.extra_args['id2']
        self.id3 = self.extra_args['id3']
        self.id4 = self.extra_args['id4']
        self.id5 = self.extra_args['id5']
        self.id6 = self.extra_args['id6']
        self.idm = self.extra_args['idm']
        self.number = self.extra_args['number']
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeGeneralEntanglementSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        self.cfg['multimode_general_entanglement']['id1'] = self.id1
        self.cfg['multimode_general_entanglement']['id2'] = self.id2
        self.cfg['multimode_general_entanglement']['id3'] = self.id3
        self.cfg['multimode_general_entanglement']['id4'] = self.id4
        self.cfg['multimode_general_entanglement']['id5'] = self.id5
        self.cfg['multimode_general_entanglement']['id6'] = self.id6
        self.cfg['multimode_general_entanglement']['number'] = self.number


class MultimodeEntanglementScalingExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_General_Entanglement', config_file='..\\config.json', **kwargs):
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        self.id1 = self.extra_args['id1']
        self.id2 = self.extra_args['id2']
        self.id3 = self.extra_args['id3']
        self.id4 = self.extra_args['id4']
        self.id5 = self.extra_args['id5']
        self.id6 = self.extra_args['id6']
        self.idm = self.extra_args['idm']
        self.number = self.extra_args['number']
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeGeneralEntanglementSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        pass
        # self.cfg['multimode_general_entanglement']['id1'] = self.id1
        # self.cfg['multimode_general_entanglement']['id2'] = self.id2
        # self.cfg['multimode_general_entanglement']['id3'] = self.id3
        # self.cfg['multimode_general_entanglement']['id4'] = self.id4
        # self.cfg['multimode_general_entanglement']['id5'] = self.id5
        # self.cfg['multimode_general_entanglement']['id6'] = self.id6
        # self.cfg['multimode_general_entanglement']['number'] = self.number



class MultimodeCPhaseTestsExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_CPhase_Tests_Experiment', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeCPhaseTestsSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        pass

class MultimodeCPhaseExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_CPhase_Experiment', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeCPhaseSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        pass


class MultimodeCNOTExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_CNOT_Experiment', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeCNOTSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        pass


class MultimodePi_PiExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_Pi_Pi_Experiment', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodePi_PiSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        pass

class CPhaseOptimizationSweepExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='multimode_cphase_optimization_sweep', config_file='..\\config.json', **kwargs):
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
        self.idle_time = self.extra_args['idle_time']
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=CPhaseOptimizationSweepSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        slab_file = SlabFile(self.data_file)
        with slab_file as f:
            f.append_pt('idle_time', self.idle_time)
            f.append_line('sweep_expt_avg_data', expt_avg_data)
            f.append_line('sweep_expt_pts', expt_pts)

            f.close()

class MultimodeSingleResonatorTomography(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='multimode_Single_Resonator_Tomography', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeSingleResonatorTomographySequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        pass


class MultimodeTwoResonatorTomography(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='multimode_two_Resonator_Tomography', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeTwoResonatorTomographySequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        pass

class MultimodeTwoResonatorTomographyPhaseSweepExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='multimode_two_resonator_tomography_phase_sweep', config_file='..\\config.json', **kwargs):
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
        self.tomography_num = self.extra_args['tomography_num']

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeTwoResonatorTomographyPhaseSweepSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        pass


class MultimodeThreeModeCorrelationExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='multimode_three_mode_correlation_experiment', config_file='..\\config.json', **kwargs):
        # self.extra_args={}
        # for key, value in kwargs.iteritems():
        #     self.extra_args[key] = value
        # self.tomography_num = self.extra_args['tomography_num']

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeThreeModeCorrelationSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        pass