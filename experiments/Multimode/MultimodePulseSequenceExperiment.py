__author__ = 'Nelson'

from slab import *
from slab.instruments.Alazar import Alazar
from slab.experiments.ExpLib.QubitPulseSequenceExperiment import *
from numpy import mean, arange, abs
from slab.dsfit import *


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

        suggested_offset_freq = self.cfg['multimodes'][int(self.cfg['multimode_ramsey']['id'])]['dc_offset_freq'] + self.offset_freq
        print "Suggested offset frequency: " + str(suggested_offset_freq)
        print "Oscillation frequency: " + str(fitdata[1] * 1e3) + " MHz"
        print "T2*: " + str(fitdata[3]) + " ns"


class MultimodeDCOffsetExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_DC_Offset_Experiment', config_file='..\\config.json', **kwargs):
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        if 'mode_calibration' in self.extra_args:
            self.mode_calibration = self.extra_args['mode_calibration']
            self.mode = self.extra_args['mode']
            self.sideband = self.extra_args['sideband']

        else:
            self.amp = self.extra_args['amp']
            self.freq = self.extra_args['freq']
            self.mode_calibration=False


        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeDCOffsetSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        print "Analyzing Ramsey Data"
        fitdata = fitdecaysin(expt_pts, expt_avg_data)

        self.offset_freq =self.cfg['multimode_dc_offset_experiment']['ramsey_freq'] - fitdata[1] * 1e9

        if self.mode_calibration:

            if self.sideband == "ge":
                self.cfg['multimodes'][self.mode]['dc_offset_chirp'] = self.offset_freq
                self.amp = self.cfg['multimodes'][self.mode]['a']
            elif self.sideband == "ef":
                self.cfg['multimodes'][self.mode]['dc_offset_chirp_ef'] = self.offset_freq
                self.amp = self.cfg['multimodes'][self.mode]['a_ef']

            print "Calibrating DC offset from driving sideband " + str(self.sideband) + " for mode " + str(self.mode) + " at a = " + str(self.amp)
            print "Sqaure pulse with ramp = " + str(self.cfg['flux_pulse_info']['sqaure'][0]['ramp_sigma']) + " ns"

        else:
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

        if self.exp=="multimode_rabi":
            print "Analyzing Multimode Rabi Data"
            fitdata = fitdecaysin(expt_pts[:], expt_avg_data[:])

            if (-fitdata[2]%180 - 90)/(360*fitdata[1]) < 0:
                self.flux_pi_length = (-fitdata[2]%180 + 90)/(360*fitdata[1])
                self.flux_2pi_length = (-fitdata[2]%180 + 270)/(360*fitdata[1])
                print "Flux pi length ge =" + str(self.flux_pi_length)
                print "Flux 2pi length ge =" + str(self.flux_2pi_length)

            else:
                # if self.cfg['flux_pulse_info']['ramp_sigma'] > 0:
                self.flux_pi_length = (-fitdata[2]%180 - 90)/(360*fitdata[1])
                self.flux_2pi_length = (-fitdata[2]%180 + 90)/(360*fitdata[1])
                print "Flux pi length ge =" + str(self.flux_pi_length)
                print "Flux 2pi length ge =" + str(self.flux_2pi_length)

            if self.cfg['multimode_calibrate_offset_experiment'][self.exp]['save_to_file']:
                print "writing the flux pulse lengths of mode" +str(self.mode) + " to config file"
                self.cfg['multimodes'][self.mode]['flux_pi_length'] =   self.flux_pi_length
                self.cfg['multimodes'][self.mode]['flux_2pi_length'] =  self.flux_2pi_length

            print "Multimode Rabi contrast for mode " + str(self.mode) + "= " + str(2*fitdata[0])

        else:


            print "Analyzing Ramsey Data"
            fitdata = fitdecaysin(expt_pts, expt_avg_data)
            self.ramsey_freq = self.cfg['multimode_calibrate_offset_experiment'][self.exp]['ramsey_freq']
            print "Ramsey Frequency = " + str(self.ramsey_freq) + " Hz"
            self.offset_freq =self.ramsey_freq - fitdata[1] * 1e9
            print "Oscillation frequency: " + str(fitdata[1] * 1e3) + " MHz"
            print "Deviation in oscillation frequency: " + str(self.offset_freq)+ " Hz"
            print "Dc offset Guess: " + str(self.dc_offset_guess)
            if self.exp == 'long_multimode_ramsey':
                self.suggested_dc_offset_freq = self.dc_offset_guess + self.offset_freq
            else:
                self.suggested_dc_offset_freq = self.dc_offset_guess + self.offset_freq
            # self.suggested_dc_offset_freq = self.offset - (fitdata[1] * 1e9 - self.cfg['multimode_ramsey']['ramsey_freq'])
            print "Suggested DC offset frequency: " + str(self.suggested_dc_offset_freq)

            print "T2*: " + str(fitdata[3]) + " ns"
            print self.cfg['multimode_calibrate_offset_experiment'][self.exp]

            print "Saving DC offset to config for mode " + str(self.mode)
            print "Old DC offset = " + str(self.cfg['multimodes'][self.mode]['dc_offset_freq']) + " Hz"
            self.cfg['multimodes'][self.mode]['dc_offset_freq'] = self.suggested_dc_offset_freq
            self.cfg['multimodes'][self.mode]['T2'] = fitdata[3]
            print "New DC offset = " + str(self.cfg['multimodes'][self.mode]['dc_offset_freq']) + " Hz"


class MultimodeQubitModeCrossKerrExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_Qubit_Mode_Cross_Kerr', config_file='..\\config.json', **kwargs):
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeQubitModeCrossKerrSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        pass


class MultimodeCalibrateEFSidebandExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_Calibrate_EF_Sideband_experiment', config_file='..\\config.json', **kwargs):
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
        if 'mode' in self.extra_args:
            self.exp = self.extra_args['exp']
            self.mode = self.extra_args['mode']
        self.dc_offset_guess_ef =  self.extra_args['dc_offset_guess_ef']

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeCalibrateEFSidebandSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):

        if self.exp =="multimode_ef_rabi":
            print "Analyzing EF Rabi Data"
            fitdata = fitdecaysin(expt_pts[2:], expt_avg_data[2:])

            if (-fitdata[2]%180 - 90)/(360*fitdata[1]) < 0:
                print fitdata[0]
                self.flux_pi_length_ef = (-fitdata[2]%180 + 90)/(360*fitdata[1])
                self.flux_2pi_length_ef = (-fitdata[2]%180 + 270)/(360*fitdata[1])
                print "Flux pi length EF =" + str(self.flux_pi_length_ef)
                print "Flux 2pi length EF =" + str(self.flux_2pi_length_ef)
                if self.cfg['multimode_calibrate_ef_sideband_experiment'][self.exp]['save_to_file']:
                    self.cfg['multimodes'][self.mode]['flux_pi_length_ef'] =   self.flux_pi_length_ef
                    self.cfg['multimodes'][self.mode]['flux_2pi_length_ef'] =  self.flux_2pi_length_ef
            else:
                print fitdata[0]
                self.flux_pi_length_ef = (-fitdata[2]%180 - 90)/(360*fitdata[1])
                self.flux_2pi_length_ef = (-fitdata[2]%180 + 90)/(360*fitdata[1])
                # self.flux_pi_length_ef = (-fitdata[2]%180 + 90)/(360*fitdata[1])
                # # self.flux_2pi_length_ef = (-fitdata[2]%180 + 270)/(360*fitdata[1])
                print "Flux pi length EF =" + str(self.flux_pi_length_ef)
                print "Flux 2pi length EF =" + str(self.flux_2pi_length_ef)
                if self.cfg['multimode_calibrate_ef_sideband_experiment'][self.exp]['save_to_file']:
                    self.cfg['multimodes'][self.mode]['flux_pi_length_ef'] =   self.flux_pi_length_ef
                    self.cfg['multimodes'][self.mode]['flux_2pi_length_ef'] =  self.flux_2pi_length_ef

        else:

            print "Analyzing Ramsey Data"
            fitdata = fitdecaysin(expt_pts, expt_avg_data)
            self.offset_freq =self.cfg['multimode_calibrate_ef_sideband_experiment'][self.exp]['ramsey_freq'] - fitdata[1] * 1e9
            self.suggested_dc_offset_freq_ef = self.dc_offset_guess_ef + self.offset_freq
            print "DC offset guess: " + str(self.dc_offset_guess_ef) + " Hz"
            # self.suggested_dc_offset_freq = self.offset - (fitdata[1] * 1e9 - self.cfg['multimode_ramsey']['ramsey_freq'])
            print "Suggested DC offset frequency: " + str(self.suggested_dc_offset_freq_ef)
            print "Oscillation frequency: " + str(fitdata[1] * 1e3) + " MHz"
            print "T2: " + str(fitdata[3]) + " ns"
            print self.cfg['multimode_calibrate_ef_sideband_experiment'][self.exp]
            print "Saving ef DC offset to config for mode " + str(self.mode)
            print "Old DC offset = " +str(self.cfg['multimodes'][self.mode]['dc_offset_freq_ef']) + " Hz"
            self.cfg['multimodes'][self.mode]['dc_offset_freq_ef'] = self.suggested_dc_offset_freq_ef
            print  "New DC offset = " +str(self.cfg['multimodes'][self.mode]['dc_offset_freq_ef']) + " Hz"


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
        pass


class MultimodeEFRabiSweepExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_EF_Rabi_Sweep', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeEFRabiSweepSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)



    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        #print self.data_file
        pass


class MultimodeT1Experiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_T1', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeT1Sequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

        if 'mode' in kwargs:
            self.id = self.extra_args['mode']
        else:
            self.id = self.cfg[self.expt_cfg_name]['id']

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        print "Analyzing mode T1 Data"
        fitdata = fitexp(expt_pts, expt_avg_data)
        print "T1: " + str(fitdata[3]) + " ns"
        self.cfg['multimodes'][self.id]['T1'] = fitdata[3]



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


class Multimode_Qubit_Mode_CZ_Offset_Experiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='multimode_qubit_mode_cz_offset', config_file='..\\config.json', **kwargs):

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=Multimode_Qubit_Mode_CZ_Offset_Sequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)
        if 'mode' in kwargs:
            self.id = self.extra_args['mode']
        else:
            self.id = self.cfg[self.expt_cfg_name]['id']

        if 'offset_exp' in self.extra_args:
            self.offset_exp = self.extra_args['offset_exp']
        else:
            self.offset_exp = self.cfg[self.expt_cfg_name]['offset_exp']


    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):

        expected_period = 360.
        if self.offset_exp==0:
            find_phase = 'max' #'max' or 'min'
            x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
            print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
            self.cfg['multimodes'][self.id]['qubit_mode_ef_offset_0'] = x_at_extremum
        if self.offset_exp==1:
            find_phase = 'min' #'max' or 'min'
            x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
            print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
            self.cfg['multimodes'][self.id]['qubit_mode_ef_offset_1'] = x_at_extremum


class Multimode_Qubit_Mode_CZ_V2_Offset_Experiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='multimode_qubit_mode_cz_v2_offset', config_file='..\\config.json', **kwargs):

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=Multimode_Qubit_Mode_CZ_V2_Offset_Sequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)
        if 'mode' in kwargs:
            self.id = self.extra_args['mode']
        else:
            self.id = self.cfg[self.expt_cfg_name]['id']

        if 'offset_exp' in self.extra_args:
            self.offset_exp = self.extra_args['offset_exp']
        else:
            self.offset_exp = self.cfg[self.expt_cfg_name]['offset_exp']


    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):

        expected_period = 360.
        if self.offset_exp==0:
            find_phase = 'max' #'max' or 'min'
            x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
            print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
            self.cfg['multimodes'][self.id]['cz_dc_phase'] = x_at_extremum

            if self.data_file:
                slab_file = SlabFile(self.data_file)
                with slab_file as f:
                    f.append_pt('offset_exp_0_phase', x_at_extremum)
                    f.close()

        if self.offset_exp==1:
            find_phase = 'min' #'max' or 'min'
            x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
            print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
            self.cfg['multimodes'][self.id]['cz_phase'] = x_at_extremum

            if self.data_file:
                slab_file = SlabFile(self.data_file)
                with slab_file as f:
                    f.append_pt('offset_exp_1_phase', x_at_extremum)
                    f.close()


class Multimode_Mode_Mode_CNOT_V2_Offset_Experiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='multimode_mode_mode_cnot_v2_offset', config_file='..\\config.json', **kwargs):

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=Multimode_Mode_Mode_CNOT_V2_Offset_Sequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)
        if 'mode' in kwargs:
            self.id = self.extra_args['mode']
        else:
            self.id = self.cfg[self.expt_cfg_name]['id']

        if 'mode2' in kwargs:
            self.id2 = self.extra_args['mode2']
        else:
            self.id2 = self.cfg[self.expt_cfg_name]['id2']


        if 'offset_exp' in self.extra_args:
            self.offset_exp = self.extra_args['offset_exp']
        else:
            self.offset_exp = self.cfg[self.expt_cfg_name]['offset_exp']

        if 'load_photon' in self.extra_args:
            self.load_photon = self.extra_args['load_photon']
        else:
            self.load_photon = self.cfg[self.expt_cfg_name]['load_photon']

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):


        if self.offset_exp==0:
            find_phase = 'max' #'max' or 'min'
            expected_period = 360.
            x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
            print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
            self.cfg['mode_mode_offset']['cnot_dc_phase'][self.id][self.id2] = x_at_extremum

            if self.data_file:
                slab_file = SlabFile(self.data_file)
                with slab_file as f:
                    f.append_pt('offset_exp_0_phase', x_at_extremum)
                    f.close()

        if self.offset_exp==1:
            find_phase = 'min' #'max' or 'min'
            expected_period = 180.0
            x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
            print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
            self.cfg['mode_mode_offset']['cnot_phase'][self.id][self.id2] = x_at_extremum

            if self.data_file:
                slab_file = SlabFile(self.data_file)
                with slab_file as f:
                    f.append_pt('offset_exp_1_phase', x_at_extremum)
                    f.close()

        if self.offset_exp==4:

            if self.load_photon:
                find_phase = 'min' #'max' or 'min'
                expected_period = 360.0
                x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
                print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
                self.cfg['mode_mode_offset']['cnot_phase'][self.id][self.id2] = x_at_extremum

                print "saving CNOT phase"

                if self.data_file:
                    slab_file = SlabFile(self.data_file)
                    with slab_file as f:
                        f.append_pt('offset_exp_1_phase', x_at_extremum)
                        f.close()
            else:
                find_phase = 'min' #'max' or 'min'
                expected_period = 360.0
                x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
                print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
                self.cfg['mode_mode_offset']['cnot_dc_phase'][self.id][self.id2] = x_at_extremum

                print "saving CNOT DC phase"

                if self.data_file:
                    slab_file = SlabFile(self.data_file)
                    with slab_file as f:
                        f.append_pt('offset_exp_0_phase', x_at_extremum)
                        f.close()

        if self.offset_exp==5:


            find_phase = 'min' #'max' or 'min'
            expected_period = 180.0
            x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
            print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
            self.cfg['mode_mode_offset']['cnot_phase2'][self.id][self.id2] = x_at_extremum

            print "saving CNOT phase"

            if self.data_file:
                slab_file = SlabFile(self.data_file)
                with slab_file as f:
                    f.append_pt('offset_exp_2_phase', x_at_extremum)
                    f.close()


class Multimode_Mode_Mode_CNOT_V3_Offset_Experiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='multimode_mode_mode_cnot_v3_offset', config_file='..\\config.json', **kwargs):

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=Multimode_Mode_Mode_CNOT_V3_Offset_Sequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)
        if 'mode' in kwargs:
            self.id = self.extra_args['mode']
        else:
            self.id = self.cfg[self.expt_cfg_name]['id']

        if 'mode2' in kwargs:
            self.id2 = self.extra_args['mode2']
        else:
            self.id2 = self.cfg[self.expt_cfg_name]['id2']


        if 'offset_exp' in self.extra_args:
            self.offset_exp = self.extra_args['offset_exp']
        else:
            self.offset_exp = self.cfg[self.expt_cfg_name]['offset_exp']

        if 'load_photon' in self.extra_args:
            self.load_photon = self.extra_args['load_photon']
        else:
            self.load_photon = self.cfg[self.expt_cfg_name]['load_photon']

        if 'include_cz_correction' in self.extra_args:
            self.include_cz_correction = self.extra_args['include_cz_correction']
        else:
            self.include_cz_correction = True

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):



        if self.offset_exp==0:

            if self.load_photon:
                find_phase = 'min' #'max' or 'min'
                expected_period = 360.0
                x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
                print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
                if self.include_cz_correction:
                    self.cfg['mode_mode_offset']['cnot_prep_phase'][self.id][self.id2] = x_at_extremum
                    print "saving CNOT phase that includes state dependent phase from state preperation sequence"
                else:
                    self.cfg['mode_mode_offset']['cnot_phase'][self.id][self.id2] = x_at_extremum
                    print "CNOT phase: No correction for state dependent phase in the input state"

                if self.data_file:
                    slab_file = SlabFile(self.data_file)
                    with slab_file as f:
                        f.append_pt('offset_exp_1_phase', x_at_extremum)
                        f.close()
            else:
                find_phase = 'min' #'max' or 'min'
                expected_period = 360.0
                x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
                print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
                self.cfg['mode_mode_offset']['cnot_dc_phase'][self.id][self.id2] = x_at_extremum

                print "saving CNOT DC phase"

                if self.data_file:
                    slab_file = SlabFile(self.data_file)
                    with slab_file as f:
                        f.append_pt('offset_exp_0_phase', x_at_extremum)
                        f.close()

        if self.offset_exp==1:


            find_phase = 'min' #'max' or 'min'
            expected_period = 180.0
            x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
            print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
            self.cfg['mode_mode_offset']['cnot_phase2'][self.id][self.id2] = x_at_extremum

            print "saving CNOT phase"

            if self.data_file:
                slab_file = SlabFile(self.data_file)
                with slab_file as f:
                    f.append_pt('offset_exp_2_phase', x_at_extremum)
                    f.close()


class Multimode_Mode_Mode_CZ_V2_Offset_Experiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='multimode_mode_mode_cz_v2_offset', config_file='..\\config.json', **kwargs):

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=Multimode_Mode_Mode_CZ_V2_Offset_Sequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)
        if 'mode' in kwargs:
            self.id = self.extra_args['mode']
        else:
            self.id = self.cfg[self.expt_cfg_name]['id']

        if 'mode2' in kwargs:
            self.id2 = self.extra_args['mode2']
        else:
            self.id2 = self.cfg[self.expt_cfg_name]['id2']

        if 'offset_exp' in self.extra_args:
            self.offset_exp = self.extra_args['offset_exp']
        else:
            self.offset_exp = self.cfg[self.expt_cfg_name]['offset_exp']


    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):

        expected_period = 360.
        if self.offset_exp==0:
            find_phase = 'max' #'max' or 'min'
            x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
            print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
            self.cfg['mode_mode_offset']['cz_dc_phase'][self.id][self.id2] = x_at_extremum

            if self.data_file:
                slab_file = SlabFile(self.data_file)
                with slab_file as f:
                    f.append_pt('2modes_offset_exp_0_phase', x_at_extremum)
                    f.close()

        if self.offset_exp==1:
            find_phase = 'min' #'max' or 'min'
            x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
            print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
            self.cfg['mode_mode_offset']['cz_phase'][self.id][self.id2] = x_at_extremum

            if self.data_file:
                slab_file = SlabFile(self.data_file)
                with slab_file as f:
                    f.append_pt('2modes_offset_exp_1_phase', x_at_extremum)
                    f.close()

class Multimode_Mode_Mode_CZ_V3_Offset_Experiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='multimode_mode_mode_cz_v3_offset', config_file='..\\config.json', **kwargs):

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=Multimode_Mode_Mode_CZ_V3_Offset_Sequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)
        if 'mode' in kwargs:
            self.id = self.extra_args['mode']
        else:
            self.id = self.cfg[self.expt_cfg_name]['id']

        if 'mode2' in kwargs:
            self.id2 = self.extra_args['mode2']
        else:
            self.id2 = self.cfg[self.expt_cfg_name]['id2']


        if 'offset_exp' in self.extra_args:
            self.offset_exp = self.extra_args['offset_exp']
        else:
            self.offset_exp = self.cfg[self.expt_cfg_name]['offset_exp']

        if 'load_photon' in self.extra_args:
            self.load_photon = self.extra_args['load_photon']
        else:
            self.load_photon = self.cfg[self.expt_cfg_name]['load_photon']

        if 'include_cz_correction' in self.extra_args:
            self.include_cz_correction = self.extra_args['include_cz_correction']
        else:
            self.include_cz_correction = True

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):

        if self.offset_exp==0:
            expected_period = 360.0
            if self.load_photon:
                if self.include_cz_correction:
                    find_phase = 'max' #'max' or 'min'
                    x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
                    print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
                    self.cfg['mode_mode_offset']['cz_prep_phase'][self.id][self.id2] = x_at_extremum
                    print "Testing for state |10> + |11>"
                    print "saving CZ_prep_phase (includes state dependent phase from state preperation sequence)"
                else:
                    find_phase = 'min' #'max' or 'min'
                    x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
                    print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
                    self.cfg['mode_mode_offset']['cz_phase'][self.id][self.id2] = x_at_extremum
                    print "Testing for state |10> + |11>"
                    print "saving CZ_phase (No correction for state dependent phase in the input state)"

                if self.data_file:
                    slab_file = SlabFile(self.data_file)
                    with slab_file as f:
                        f.append_pt('offset_exp_1_phase', x_at_extremum)
                        f.close()
            else:
                find_phase = 'min' #'max' or 'min'
                expected_period = 360.0
                x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
                print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
                self.cfg['mode_mode_offset']['cz_dc_phase'][self.id][self.id2] = x_at_extremum

                print "saving CZ DC phase: correction for |00> + |01>"

                if self.data_file:
                    slab_file = SlabFile(self.data_file)
                    with slab_file as f:
                        f.append_pt('offset_exp_0_phase', x_at_extremum)
                        f.close()

        if self.offset_exp==1:


            find_phase = 'min' #'max' or 'min'
            expected_period = 360.0
            x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
            print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
            self.cfg['mode_mode_offset']['cz_phase2'][self.id][self.id2] = x_at_extremum

            print "Testing for state: |00> + |10>"
            print "saving CZ_phase2"

            if self.data_file:
                slab_file = SlabFile(self.data_file)
                with slab_file as f:
                    f.append_pt('offset_exp_2_phase', x_at_extremum)
                    f.close()


class Multimode_AC_Stark_Shift_Offset_Experiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='multimode_ac_stark_shift', config_file='..\\config.json', **kwargs):

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=Multimode_AC_Stark_Shift_Offset_Sequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)
        if 'mode' in kwargs:
            self.id = self.extra_args['mode']
        else:
            self.id = self.cfg[self.expt_cfg_name]['id']

        if 'mode2' in kwargs:
            self.id2 = self.extra_args['mode2']
        else:
            self.id2 = self.cfg[self.expt_cfg_name]['id2']

        if 'offset_exp' in self.extra_args:
            self.offset_exp = self.extra_args['offset_exp']
        else:
            self.offset_exp = self.cfg[self.expt_cfg_name]['offset_exp']


    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):

        expected_period = 360.
        if self.offset_exp==0:
            find_phase = 'max' #'max' or 'min'
            x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
            print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
            #self.cfg['mode_mode_offset']['cz_dc_phase'][self.id][self.id2] = x_at_extremum

            if self.data_file:
                slab_file = SlabFile(self.data_file)
                with slab_file as f:
                    f.append_pt('offset_exp_0_phase', x_at_extremum)
                    f.close()

        if self.offset_exp==1:
            find_phase = 'max' #'max' or 'min'
            x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
            print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
            #self.cfg['mode_mode_offset']['cz_phase'][self.id][self.id2] = x_at_extremum

            if self.data_file:
                slab_file = SlabFile(self.data_file)
                with slab_file as f:
                    f.append_pt('offset_exp_1_phase', x_at_extremum)
                    f.close()

class MultimodePi_PiExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_Pi_Pi_Experiment', config_file='..\\config.json', **kwargs):

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodePi_PiSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)
        if 'mode' in kwargs:
            self.id = self.extra_args['mode']
        else:
            self.id = self.cfg[self.expt_cfg_name]['id']

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        if self.cfg[self.expt_cfg_name]['half_pi_sb']:
            x_at_extremum = float(expt_pts[argmax(expt_avg_data)])
            print 'Phase of maximum amplitude: %s degrees' %(x_at_extremum)
            self.cfg['multimodes'][self.id]['pi_pi_offset_phase'] = x_at_extremum
        else:
            expected_period = 360.
            find_phase = 'min' #'max' or 'min'
            x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
            print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
            self.cfg['multimodes'][self.id]['pi_pi_offset_phase'] = x_at_extremum



class Multimode_ef_Pi_PiExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_ef_Pi_Pi_Experiment', config_file='..\\config.json', **kwargs):

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=Multimode_ef_Pi_PiSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)
        if 'mode_1' in kwargs:
            self.id1 = self.extra_args['mode_1']
        else:
            self.id1 = self.cfg[self.expt_cfg_name]['id1']

        if 'mode_2' in kwargs:
            self.id2 = self.extra_args['mode_2']
        else:
            self.id2 = self.cfg[self.expt_cfg_name]['id2']

    def pre_run(self):

        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):

        expected_period = 360.
        find_phase = 'min' #'max' or 'min'
        x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
        print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
        self.cfg['multimodes'][self.id2]['ef_pi_pi_offset_phase'] = x_at_extremum

class Multimode_ef_2pi_Experiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_ef_2pi_Experiment', config_file='..\\config.json', **kwargs):

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=Multimode_ef_2pi_Sequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)
        if 'mode_1' in kwargs:
            self.id1 = self.extra_args['mode_1']
        else:
            self.id1 = self.cfg[self.expt_cfg_name]['id1']

        if 'mode_2' in kwargs:
            self.id2 = self.extra_args['mode_2']
        else:
            self.id2 = self.cfg[self.expt_cfg_name]['id2']

    def pre_run(self):

        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        expected_period = 360.
        find_phase = 'min' #'max' or 'min'
        x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)
        print 'Phase at %s: %s degrees' %(find_phase,x_at_extremum)
        self.cfg['multimodes'][self.id2]['ef_2pi_offset_phase'] = x_at_extremum



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

class MultimodeSingleResonatorRandomizedBenchmarkingExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='multimode_single_resonator_randomized_benchmarking', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeSingleResonatorRandomizedBenchmarkingSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        pass

class MultimodeQubitModeRandomizedBenchmarkingExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='multimode_qubit_mode_randomized_benchmarking', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeQubitModeRandomizedBenchmarkingSequence, pre_run=self.pre_run,
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
                                                    PulseSequence=MultimodeTwoResonatorTomographyPhaseSweepSequenceNEW, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        pass

class MultimodeProcessTomographyPhaseSweepExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='multimode_process_tomography_phase_sweep', config_file='..\\config.json', **kwargs):
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
        self.tomography_num = self.extra_args['tomography_num']
        self.state_num = self.extra_args['state_num']

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeProcessTomographyPhaseSweepSequenceDebug, pre_run=self.pre_run,
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

class MultimodeCPhaseAmplificationExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_CPhase_Amplification_Experiment', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeCPhaseAmplificationSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        pass


class MultimodeCNOTAmplificationExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_CNOT_Amplification_Experiment', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeCNOTAmplificationSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):
        pass


class Multimode_State_Dep_Shift_Experiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_State_Dep_Shift', config_file='..\\config.json', **kwargs):

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value


        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=Multimode_State_Dep_Shift_Sequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)


        if 'mode' in kwargs:
            self.mode = self.extra_args['mode']
        else:
            self.mode = self.cfg[self.expt_cfg_name]['id']

        if 'exp' in kwargs:
            self.exp = self.extra_args['exp']
        else:
            self.exp = self.cfg[self.expt_cfg_name]['exp']

        if 'qubit_shift_ge' in kwargs:
            self.qubit_shift_ge = self.extra_args['qubit_shift_ge']
        else:
            self.qubit_shift_ge = self.cfg[self.expt_cfg_name]['qubit_shift_ge']

        if 'qubit_shift_ef' in kwargs:
            self.qubit_shift_ef = self.extra_args['qubit_shift_ef']
        else:
            self.qubit_shift_ef = self.cfg[self.expt_cfg_name]['qubit_shift_ef']


    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):

        if self.exp == 1 or self.exp==2:

            print "Analyzing Rabi Data"

            xdata = expt_pts
            ydata = expt_avg_data
            FFT=scipy.fft(ydata)
            fft_freqs=scipy.fftpack.fftfreq(len(ydata),xdata[1]-xdata[0])
            max_ind=np.argmax(abs(FFT[2:len(ydata)/2.]))+2
            fft_val=FFT[max_ind]

            fitparams=[0,0,0,0,0]
            fitparams[4]=np.mean(ydata)
            fitparams[0]=(max(ydata)-min(ydata))/2.#2*abs(fft_val)/len(fitdatay)
            fitparams[1]=fft_freqs[max_ind]
            fitparams[2]=-90.0
            fitparams[3]=(max(xdata)-min(xdata))
            fitdata=fitdecaysin(xdata[:],ydata[:],fitparams=fitparams,showfit=False)

            self.pi_length = around(1/fitdata[1]/2,decimals=2)
            self.half_pi_length =around(1/fitdata[1]/4,decimals=2)

            if self.qubit_shift_ge == 1:
                print 'Rabi pi: %s ns' % (self.pi_length)
                print 'Rabi pi/2: %s ns' % (self.half_pi_length)
                print 'T1*: %s ns' % (fitdata[3])
            elif self.qubit_shift_ef == 1:
                print 'Rabi ef pi: %s ns' % (self.pi_length)
                print 'Rabi ef pi/2: %s ns' % (self.half_pi_length)
                print 'T1*: %s ns' % (fitdata[3])

        elif self.exp == 6 or self.exp ==7 or self.exp ==8:
            print "Analyzing multimode Rabi Data"
            fitdata = fitdecaysin(expt_pts[2:], expt_avg_data[2:])

            if (-fitdata[2]%180 - 90)/(360*fitdata[1]) < 0:
                print fitdata[0]
                self.flux_pi_length = (-fitdata[2]%180 + 90)/(360*fitdata[1])
                self.flux_2pi_length = (-fitdata[2]%180 + 270)/(360*fitdata[1])
                print "pi length =" + str(self.flux_pi_length)
                print "2pi length =" + str(self.flux_2pi_length)


        elif self.exp == 3 or self.exp ==4 or self.exp ==5:

            print "Analyzing offset phase in presence of photon in mode %s" %(self.mode)

            xdata = expt_pts
            ydata = expt_avg_data
            fitparams = [(max(ydata)-min(ydata))/(2.0),1/360.0,90,mean(ydata)]
            fitdata=fitsin(xdata[:],ydata[:],fitparams=fitparams,showfit=False)
            if self.exp == 3:
                self.cfg['multimodes'][self.mode]['qubit_offset_phase'] = around((-(fitdata[2]%180) + 90),2)
                print "Offset Phase = %s" %(self.cfg['multimodes'][self.mode]['qubit_offset_phase'])
            elif self.exp ==4:
                self.cfg['multimodes'][self.mode]['qubit_offset_phase_2'] = around((-(fitdata[2]%180) + 90),2)
                print "Offset Phase = %s" %(self.cfg['multimodes'][self.mode]['qubit_offset_phase_2'])
            else:
                print "Offset Phase = %s" %(around((-(fitdata[2]%180) + 90),2))

        else:

            print "Analyzing Ramsey Data"
            fitdata = fitdecaysin(expt_pts, expt_avg_data)

            self.offset_freq =self.cfg['multimode_state_dep_shift']['ramsey_freq'] - fitdata[1] * 1e9


            print "State dependent shift = " + str(self.offset_freq) + "MHz"
            print "Oscillation frequency: " + str(fitdata[1] * 1e3) + " MHz"
            print "T2*: " + str(fitdata[3]) + " ns"

            if self.qubit_shift_ge == 1:
                self.cfg['multimodes'][self.mode]['shift'] =   self.offset_freq
            elif self.qubit_shift_ef ==1:
                self.cfg['multimodes'][self.mode]['shift_ef'] =   self.offset_freq


class MultimodeEchoSidebandExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Multimode_Echo_Sideband_Experiment', config_file='..\\config.json', **kwargs):

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value


        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                                    PulseSequence=MultimodeEchoSidebandSequence, pre_run=self.pre_run,
                                                    post_run=self.post_run, prep_tek2= True,**kwargs)


        if 'mode' in kwargs:
            self.mode = self.extra_args['mode']
        else:
            self.mode = self.cfg[self.expt_cfg_name]['id']

        if 'exp' in kwargs:
            self.exp = self.extra_args['exp']
        else:
            self.exp = self.cfg[self.expt_cfg_name]['exp']

    def pre_run(self):
        self.tek2 = InstrumentManager()["TEK2"]

    def post_run(self, expt_pts, expt_avg_data):

        if self.exp ==2:
            pass
        else:
            if self.exp == 0:

                print "pi/2 + pi/2(dphi) sideband expt for mode: %s" %(self.mode)
                expected_period = 360.0

            else:

                print "pi/2 + pi(dphi) +  pi/2(phi_0) sideband expt for mode: %s" %(self.mode)
                expected_period = 180.0


            find_phase = 'min' #'max' or 'min'
            x_at_extremum = sin_phase(expt_pts,expt_avg_data,expected_period,find_phase)

            if self.exp == 0:
                self.cfg['multimodes'][self.mode]['piby2_piby2_off_phase_0'] = x_at_extremum
                print "Offset Phase = %s" %(x_at_extremum)
            else:
                self.cfg['multimodes'][self.mode]['piby2_piby2_off_phase_1'] =x_at_extremum
                print "Offset Phase = %s" %(x_at_extremum)

