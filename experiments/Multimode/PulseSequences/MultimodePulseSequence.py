__author__ = 'Nelson'

from slab.instruments.awg.PulseSequence import *
from numpy import arange, linspace, sin, pi, sign, append, around, mean, delete,arcsin,sqrt,load,save
from slab.experiments.ExpLib.PulseSequenceBuilder import *
from slab.experiments.ExpLib.QubitPulseSequence import *
from slab.experiments.ExpLib.QubitPulseSequence_SB_cool import *
from slab.experiments.ExpLib.PulseSequenceGroup import *
import random
# from numpy import around, mean, delete,arcsin
from liveplot import LivePlotClient

class MultimodeRabiSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.multimode_cfg = cfg['multimodes']
        # QubitPulseSequenceSBcool.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)
        # self.pulse_cfg = cfg['pulse_info']


    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        #self.id = self.expt_cfg['id']
        #self.flux_pulse_type = self.multimode_cfg[int(self.id)]['flux_pulse_type']

    def define_pulses(self,pt):
        # # original
        # self.psb.append('q','pi', self.pulse_type)
        # self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length=pt,phase=self.multimode_cfg[int(self.id)]['pi_pi_offset_phase']/(2.0))
        # #

        # # # DC offset
        # self.ramsey = 0e6#100e6
        # self.freq = 3.3718e9
        # self.psb.append('q','half_pi', self.pulse_type)
        # self.psb.append('q','pi_q_ef', self.pulse_type)
        # self.psb.append('q:mm','general', 'square', amp=0.0, length=pt,freq=self.freq)
        # self.psb.append('q','pi_q_ef', self.pulse_type)
        # self.psb.append('q','half_pi', self.pulse_type, phase = 360.0*self.ramsey*pt/(1.0e9))


        # self.psb.append('q','pi', self.pulse_type)
        # self.psb.append('q','half_pi_q_ef', self.pulse_type)
        # self.psb.idle(pt)
        # self.psb.append('q','half_pi_q_ef', self.pulse_type,phase = 360.0*self.ramsey*pt/(1.0e9))
        # self.psb.append('q','pi', self.pulse_type)
        # self.psb.append('q','pi_q_ef', self.pulse_type)

        self.psb.append('q','pi', self.pulse_type)
        self.psb.append('q:mm','general', 'square', amp=self.expt_cfg['a'], length=1000,freq=pt)


class MultimodeEFRabiSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)
        self.id = self.expt_cfg['id']

    def define_pulses(self,pt):
        self.psb.append('q','pi', self.pulse_type)
        self.psb.append('q,mm'+str(6),'pi_ge')
        self.psb.append('q','pi', self.pulse_type)
        self.psb.append('q','pi_q_ef')
        self.psb.append('q:mm','general', self.flux_pulse_type, amp=self.multimode_cfg[self.id]['a_ef'], length=pt,freq=self.multimode_cfg[self.id]['flux_pulse_freq_ef'])
        self.psb.append('q','pi', self.pulse_type)
        if self.expt_cfg['cal_ef']:
            self.psb.append('q','pi_q_ef')

class MultimodeRamseySequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.flux_pulse_cfg = cfg['flux_pulse_info']
        self.multimode_cfg = cfg['multimodes']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])


    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.offset_phase = self.pulse_cfg['gauss']['offset_phase']
        self.id = self.expt_cfg['id']
        self.flux_pulse_type = self.multimode_cfg[int(self.id)]['flux_pulse_type']
        if self.flux_pulse_cfg['fix_phase']:
            self.phase_freq = self.expt_cfg['ramsey_freq']
        else:
            self.phase_freq = self.multimode_cfg[int(self.id)]['dc_offset_freq'] + self.expt_cfg['ramsey_freq']


    def define_pulses(self,pt):


        self.psb.append('q','half_pi', self.pulse_type)
        self.psb.append('q,mm'+str(self.id),'pi_ge')
        self.psb.idle(pt)
        self.psb.append('q,mm'+str(self.id),'pi_ge', phase = self.multimode_cfg[self.id]['pi_pi_offset_phase'] + 360.0*self.phase_freq*pt/(1.0e9))
        self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase)

class MultimodeCalibrateOffsetSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.flux_pulse_cfg = cfg['flux_pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)
        self.exp = self.extra_args['exp']
        self.mode = self.extra_args['mode']
        self.dc_offset_guess =  self.extra_args['dc_offset_guess']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses,sb_cool=False)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg[self.exp]['start'], self.expt_cfg[self.exp]['stop'], self.expt_cfg[self.exp]['step'])


    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.id = self.mode
        self.flux_pulse_type = self.multimode_cfg[int(self.id)]['flux_pulse_type']
        self.offset_phase = self.pulse_cfg['gauss']['offset_phase']
        if self.exp == 'multimode_rabi':
            pass
        else:
            if self.flux_pulse_cfg['fix_phase']:
                self.phase_freq = self.expt_cfg[self.exp]['ramsey_freq']
            else:
                self.phase_freq = self.dc_offset_guess+ self.expt_cfg[self.exp]['ramsey_freq']
                print "DC offset guess (pulse sequence)= " + str(self.dc_offset_guess) + " Hz"


    def define_pulses(self,pt):
        if self.exp == 'multimode_rabi':

            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length=pt,phase=self.multimode_cfg[int(self.id)]['pi_pi_offset_phase']/(2.0))

        else:
            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=self.multimode_cfg[int(self.id)]['pi_pi_offset_phase']/(2.0)-self.offset_phase)
            self.psb.idle(pt)
            self.psb.append('q,mm'+str(self.id),'pi_ge', phase = 360.0*self.phase_freq*pt/(1.0e9)-self.multimode_cfg[int(self.id)]['pi_pi_offset_phase']/(2.0)+180.0)
            self.psb.append('q','half_pi', self.pulse_type)

class MultimodeEchoSidebandSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.flux_pulse_cfg = cfg['flux_pulse_info']
        self.multimode_cfg = cfg['multimodes']

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        if 'modelist' in self.extra_args:
            self.modelist = self.extra_args['modelist']
        else:
            self.modelist = arange(11)


        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses,sb_cool = False)

    def define_points(self):

        if 'exp' in self.extra_args:
            self.exp = self.extra_args['exp']
        else:
            self.exp = self.expt_cfg['exp']

        if self.exp==0 or self.exp == 1:

            self.expt_pts = arange(self.expt_cfg['start_phase'],self.expt_cfg['stop_phase'],self.expt_cfg['step_phase'])

        if self.exp == 2:

            self.expt_pts = arange(16)

        if self.exp == 3 or self.exp == 4:

            self.expt_pts = arange(len(self.modelist))


    def define_parameters(self):
        if 'mode' in self.extra_args:
            self.id = self.extra_args['mode']
        else:
            self.id = self.expt_cfg['id']

        if 'mode2' in self.extra_args:
            self.id2 = self.extra_args['mode2']
        else:
            self.id2 = self.expt_cfg['id2']


        if 'load_photon' in self.extra_args:
            self.load_photon = self.extra_args['load_photon']
        else:
            self.load_photon = self.expt_cfg['load_photon']


        print "Target id: " +str(self.id)

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.offset_phase = self.pulse_cfg['gauss']['offset_phase']



    def define_pulses(self,pt):

        if self.exp ==2:

            # Echo sideband testing: loads photon in the modes one by one and compares echo/regular sideband for a given mode

            modelist = array([0,1,3,4,5,6,7,9,10])
            mode2list=[]
            for mode in modelist:
                if mode == self.id:
                    pass
                else:
                    mode2list.append(mode)

            modelist = array(mode2list)
            # modelist = delete(modelist,self.id)
            self.id2 = modelist[pt/2]


            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(int(self.id2)),'pi_ge')

            half_pi_length = self.multimode_cfg[int(self.id)]['flux_pi_length'] -  (self.multimode_cfg[int(self.id)]['flux_2pi_length']-self.multimode_cfg[int(self.id)]['flux_pi_length'])/2.0
            if self.load_photon:
                self.freq_shift_q = +self.multimode_cfg[self.id2]['shift']
            else:
                self.freq_shift_q = 0
            self.psb.append('q','pi', self.pulse_type,add_freq=self.freq_shift_q)
            if pt%2 == 1:
                self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length= half_pi_length)
                self.psb.append('q,mm'+str(int(self.id)),'pi_ge',phase=self.multimode_cfg[self.id]['piby2_piby2_off_phase_1'])
                self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length= half_pi_length,phase = self.multimode_cfg[self.id]['piby2_piby2_off_phase_0'])
            else:
                self.psb.append('q,mm'+str(int(self.id)),'pi_ge')

        elif self.exp ==3:

            self.id = self.modelist[pt]
            half_pi_length = self.multimode_cfg[int(self.id)]['flux_pi_length'] -  (self.multimode_cfg[int(self.id)]['flux_2pi_length']-self.multimode_cfg[int(self.id)]['flux_pi_length'])/2.0
            self.psb.append('q','pi', self.pulse_type)

            self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length= half_pi_length)
            self.psb.append('q,mm'+str(int(self.id)),'pi_ge',phase=self.multimode_cfg[self.id]['piby2_piby2_off_phase_1'])
            self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length= half_pi_length,phase = self.multimode_cfg[self.id]['piby2_piby2_off_phase_0'])

        elif self.exp ==4:
            self.id = self.modelist[pt]
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(int(self.id)),'pi_ge')
        else:

            # Echo sideband calibration: First pi/2 phase and then pi phase

            self.psb.append('q','pi',self.pulse_type)
            if self.exp==0:
                phasepi = 0
                phasepiby2 = pt

            if self.exp==1:
                phasepi = pt
                phasepiby2 = self.multimode_cfg[self.id]['piby2_piby2_off_phase_0']

            half_pi_length = self.multimode_cfg[int(self.id)]['flux_pi_length'] -  (self.multimode_cfg[int(self.id)]['flux_2pi_length']-self.multimode_cfg[int(self.id)]['flux_pi_length'])/2.0

            self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length= half_pi_length)

            if self.exp == 1:
                self.psb.append('q,mm'+str(int(self.id)),'pi_ge',phase=phasepi)
            else:
                pass

            self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length= half_pi_length,phase = phasepiby2)

class MultimodeQubitModeCrossKerrSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.flux_pulse_cfg = cfg['flux_pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'],self.expt_cfg['stop'],self.expt_cfg['step'])


    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        if 'mode' in self.extra_args:
            self.exp = self.extra_args['exp']
            self.id = self.extra_args['mode']
            self.id2 = self.extra_args['mode2']

        else:
            self.id = self.expt_cfg['id']
            self.id2 = self.expt_cfg['id2']
            self.exp = self.expt_cfg['exp']




    def define_pulses(self,pt):
        self.psb.append('q','pi', self.pulse_type)
        self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(+self.id)]['a'], length= self.multimode_cfg[int(self.id)]['flux_pi_length'])
        if self.exp == 1:
            self.psb.append('q','pi', self.pulse_type)
        self.psb.idle(pt)
        if self.exp == 1:
            self.psb.append('q','pi', self.pulse_type)
        self.psb.append('q,mm'+str(self.id2),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(+self.id2)]['a'], length= self.multimode_cfg[int(self.id2)]['flux_pi_length'])

class MultimodeCalibrateEFSidebandSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.flux_pulse_cfg=cfg['flux_pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)
        if 'mode' in self.extra_args:
            self.exp = self.extra_args['exp']
            self.mode = self.extra_args['mode']

        if 'sb_cool' in self.extra_args:
            sb_cool = self.extra_args['sb_cool']
        else:
            sb_cool = False


        if 'add_freq' in self.extra_args:
            self.add_freq = self.extra_args['add_freq']
        else:
            self.add_freq = 0

        self.dc_offset_guess_ef =  self.extra_args['dc_offset_guess_ef']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses,sb_cool=False)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg[self.exp]['start'], self.expt_cfg[self.exp]['stop'], self.expt_cfg[self.exp]['step'])


    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']

        self.flux_pulse_type = self.multimode_cfg[int(self.mode)]['flux_pulse_type']
        if 'mode' in self.extra_args:
            pass
        else:
            self.exp = self.expt_cfg['exp']
            self.mode = self.expt_cfg['mode']

        self.id = self.mode
        if 'shift' in self.extra_args:
            self.shift = self.extra_args['shift']
        else:
            self.shift = self.expt_cfg['shift']


        if self.exp == 'multimode_ef_rabi':
            pass
        else:
            if self.flux_pulse_cfg['fix_phase']:
                self.phase_freq = self.expt_cfg[self.exp]['ramsey_freq']
            else:
                self.phase_freq = self.dc_offset_guess_ef + self.expt_cfg[self.exp]['ramsey_freq']

                print "ef DC offset guess (pulse sequence) = "+str(self.dc_offset_guess_ef) + " Hz"

    def define_pulses(self,pt):

        if self.exp == 'multimode_ef_rabi':
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q','pi_q_ef')
            # self.psb.append('q:mm','general', self.flux_pulse_type, amp=self.multimode_cfg[self.mode]['a_ef'], length=pt,freq=self.multimode_cfg[self.mode]['flux_pulse_freq_ef'])
            self.psb.append('q,mm'+str(self.mode),'general_f', self.flux_pulse_type, amp=self.multimode_cfg[self.mode]['a_ef'], length=pt,freq=self.multimode_cfg[self.mode]['flux_pulse_freq_ef'])
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q','pi_q_ef')

        else:

            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q','half_pi_q_ef')
            self.psb.append('q,mm'+str(self.id),'pi_ef')
            self.psb.idle(pt)
            self.psb.append('q,mm'+str(self.id),'pi_ef', phase = 360.0*self.phase_freq*pt/(1.0e9))
            self.psb.append('q','half_pi_q_ef')
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q','pi_q_ef')

class MultimodeEFRamseySequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.flux_pulse_cfg=cfg['flux_pulse_info']
        self.multimode_cfg = cfg['multimodes']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq+ self.expt_cfg['ramsey_freq'])
        self.id = self.expt_cfg['id']
        if self.flux_pulse_cfg['fix_phase']:
            self.phase_freq = self.expt_cfg['ramsey_freq']
        else:
            self.phase_freq = self.multimode_cfg[int(self.id)]['dc_offset_freq_ef'] + self.expt_cfg['ramsey_freq']

    def define_pulses(self,pt):
        self.psb.append('q','pi', self.pulse_type)
        self.psb.append('q','half_pi_q_ef')
        self.psb.append('q,mm'+str(self.id),'pi_ef')
        # self.psb.append('q:mm','general', self.flux_pulse_type, amp=self.multimode_cfg[self.id]['a_ef'], length=self.multimode_cfg[self.id]['flux_pi_length_ef'],freq=self.multimode_cfg[self.id]['flux_pulse_freq_ef'])
        self.psb.idle(pt)
        self.psb.append('q,mm'+str(self.id),'pi_ef', phase = 360.0*self.phase_freq*pt/(1.0e9))
        # self.psb.append('q:mm','general', self.flux_pulse_type, amp=self.multimode_cfg[self.id]['a_ef'], length=self.multimode_cfg[self.id]['flux_pi_length_ef'],freq=self.multimode_cfg[self.id]['flux_pulse_freq_ef'],phase=360.0*self.phase_freq*pt/(1.0e9))
        self.psb.append('q','half_pi_q_ef')
        self.psb.append('q','pi', self.pulse_type)
        if self.expt_cfg['cal_ef']:
            self.psb.append('q','pi_q_ef')

class MultimodeRabiSweepSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg, **kwargs):
        self.extra_args={}
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']

        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)

        self.flux_freq = self.extra_args['flux_freq']
        print "flux_freq: " +str(self.flux_freq)

        if 'amp' in kwargs:
            self.amp = kwargs['amp']
        else:
            self.amp = self.expt_cfg['a']

        if 'blue_sideband' in kwargs:
            self.blue_sideband = kwargs['blue_sideband']
        else:
            self.blue_sideband = False
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses, **kwargs)


    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']

    def define_pulses(self,pt):
        # self.psb.append('q','general', self.pulse_type, amp=0.0, length=50.0,freq=self.pulse_cfg['gauss']['iq_freq'])
        if self.blue_sideband:
            pass
        else:
            self.psb.append('q','pi', self.pulse_type)
        self.psb.append('q:mm','general', self.flux_pulse_type, amp=self.amp, length=pt,freq=self.flux_freq)

class MultimodeBlueSidebandSweepSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg, **kwargs):
        self.extra_args={}
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value


        if 'flux_freq' in kwargs:
            self.flux_freq = self.extra_args['flux_freq']
        else:
            self.flux_freq =cfg['multimode_bluesideband_sweep']['flux_freq']

        if 'add_freq' in kwargs:
            self.add_freq = self.extra_args['add_freq']
        else:
            self.add_freq =-55e6

        if 'qubit_delay' in kwargs:
            self.qubit_delay = self.extra_args['qubit_delay']
        else:
            self.qubit_delay =93
            # self.qubit_delay =4

        if 'predrive' in kwargs:
            self.predrive = self.extra_args['predrive']
        else:
            self.predrive =0

        if 'pi_pulse_delay' in kwargs:
            self.pi_pulse_delay = self.extra_args['pi_pulse_delay']
        else:
            self.pi_pulse_delay =0

        if 'pi_pulse_phase' in kwargs:
            self.pi_pulse_phase = self.extra_args['pi_pulse_phase']
        else:
            self.pi_pulse_phase =cfg['multimode_bluesideband_sweep']['pi_pulse_phase']


        # print "flux_freq: " +str(self.flux_freq)


        if 'flux_amp' in self.extra_args:
            self.flux_amp = self.extra_args['flux_amp']
        else:
            self.flux_amp = cfg['multimode_bluesideband_sweep']['flux_a']

        if 'qubit_amp' in self.extra_args:
            self.qubit_amp = self.extra_args['qubit_amp']
        else:
            self.qubit_amp = cfg['multimode_bluesideband_sweep']['qubit_a']

        self.ramsey = cfg['multimode_bluesideband_sweep']['ramsey_freq']

        self.ramp_sigma = cfg['multimode_bluesideband_sweep']['ramp_sigma']

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses, **kwargs)



    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])
        # self.expt_pts = append(arange(0,self.ramp_sigma,self.expt_cfg['step']/4.),arange(self.expt_cfg['start']+self.ramp_sigma,self.expt_cfg['stop']+self.ramp_sigma,self.expt_cfg['step']))

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']

    def define_pulses(self,pt):
        # if pt < self.ramp_sigma:
        #
        #     self.psb.append('q','general', 'gauss', amp=self.qubit_amp, length=pt, freq=self.pulse_cfg['gauss']['iq_freq']+self.add_freq, delay=pt+self.qubit_delay)
        #
        #     self.psb.append('q:mm','general', 'gauss', amp=self.flux_amp, length=pt, freq=self.flux_freq)
        #
        #     self.psb.idle(self.pi_pulse_delay)
        #
        #     self.psb.append('q','half_pi', 'square', phase = self.pi_pulse_phase)
        #
        # elif pt >= self.ramp_sigma:
        # #self.psb.append('q','general', self.pulse_type, amp=self.qubit_amp, length=pt,freq=self.pulse_cfg[self.pulse_type]['iq_freq'], add_freq=self.add_freq, delay=pt+self.qubit_delay)
        #
        # self.psb.append('q','pi', self.pulse_type)
        # self.psb.idle(pt)
        #
        #     self.psb.append('q','general', 'square', amp=self.qubit_amp, length=pt-self.ramp_sigma, freq=self.pulse_cfg['square']['iq_freq']+self.add_freq, delay=pt+self.qubit_delay)
        #
        #     self.psb.append('q:mm','general', 'square', amp=self.flux_amp, length=pt-self.ramp_sigma, freq=self.flux_freq)
        #
        #     self.psb.idle(self.pi_pulse_delay)
        #
        #     self.psb.append('q','half_pi', 'square', phase = self.pi_pulse_phase)

        # if self.pulse_type == 'gauss':
        #     self.psb.append('q','general', self.pulse_type, amp=self.qubit_amp, length=pt, freq=self.pulse_cfg[self.pulse_type]['iq_freq']+self.add_freq, delay=4*pt+self.qubit_delay)
        #
        #     self.psb.append('q:mm','general', self.flux_pulse_type, amp=self.flux_amp, length=pt, freq=self.flux_freq)
        #
        #     self.psb.idle(self.pi_pulse_delay)
        #
        #     # self.psb.append('q','half_pi', 'square', phase = self.pi_pulse_phase)
        #
        # else:
            # self.psb.append('q','general', self.pulse_type, amp=self.qubit_amp, length=self.predrive, freq=self.pulse_cfg[self.pulse_type]['iq_freq'], delay=0)

            # self.psb.append('q','general', self.pulse_type, amp=self.qubit_amp, length=pt, freq=self.pulse_cfg[self.pulse_type]['iq_freq']+self.add_freq, delay=pt+self.qubit_delay)
        self.psb.append('q','pi', self.pulse_type)
        self.psb.append('q:mm','general', self.flux_pulse_type, amp=self.flux_amp, length=pt, freq=self.flux_freq)

            # self.psb.idle(self.pi_pulse_delay)

            # self.psb.append('q','half_pi', 'square', phase = self.pi_pulse_phase)
            #
            # self.psb.append('q','general', 'square', amp=self.qubit_amp, length=pt, freq=self.pulse_cfg[self.pulse_type]['iq_freq'],phase = self.pi_pulse_phase)

        # if pt == 0:
        #
        #     self.psb.append('q','general', self.pulse_type, amp=self.qubit_amp, length=pt, freq=self.pulse_cfg[self.pulse_type]['iq_freq']+self.add_freq, delay=pt+self.qubit_delay)
        #
        #     self.psb.append('q:mm','general', self.flux_pulse_type, amp=self.flux_amp, length=pt, freq=self.flux_freq)
        #
        #     self.psb.idle(self.pi_pulse_delay)
        #
        #     self.psb.append('q','half_pi', 'square', phase = 38.8-90)
        #
        # elif pt > 0:
        #
        #     self.psb.append('q','general', self.pulse_type, amp=self.qubit_amp, length=pt, freq=self.pulse_cfg[self.pulse_type]['iq_freq']+self.add_freq, delay=pt+self.qubit_delay)
        #
        #     self.psb.append('q:mm','general', self.flux_pulse_type, amp=self.flux_amp, length=pt, freq=self.flux_freq)
        #
        #     self.psb.idle(self.pi_pulse_delay)
        #
        #     self.psb.append('q','half_pi', 'square', phase = self.pi_pulse_phase)



        # self.psb.append('q','half_pi', 'square', phase = self.pi_pulse_phase)

        # self.psb.append('q','half_pi', self.pulse_type)
        # self.psb.idle(pt)
        # self.psb.append('q','half_pi', self.pulse_type, phase = 360.0*self.ramsey*pt/(1.0e9))

        # self.psb.append('q','half_pi', self.pulse_type)
        # self.psb.append('q:mm','general', self.flux_pulse_type, amp=self.flux_amp, length=pt,freq=self.flux_freq)
        # self.psb.append('q','half_pi', self.pulse_type, phase = 360.0*self.ramsey*pt/(1.0e9))

        # self.psb.append('q','pi', self.pulse_type)
        # self.psb.idle(pt)

class MultimodeDCOffsetSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        self.multimode_cfg = cfg['multimodes']

        if 'mode_calibration' in self.extra_args:
            self.mode_calibration = self.extra_args['mode_calibration']
            self.mode = self.extra_args['mode']
            self.sideband = self.extra_args['sideband']
            if self.sideband == "ef":
                self.amp = self.multimode_cfg[self.mode]['a_ef']
                self.freq =self.multimode_cfg[self.mode]['flux_pulse_freq_ef']

            else:
                self.amp = self.multimode_cfg[self.mode]['a']
                self.freq =self.multimode_cfg[self.mode]['flux_pulse_freq']

        else:
            self.amp = self.extra_args['amp']
            self.freq = self.extra_args['freq']

        if 'sideband' in self.extra_args:
            self.sideband = self.extra_args['sideband']
        else:
            self.sideband = "ef"
        if 'ramsey_freq' in self.extra_args:
            self.ramsey = self.extra_args['ramsey_freq']
        else:
            self.ramsey = cfg['multimode_dc_offset_experiment']['ramsey_freq']

        # self.pulse_cfg = cfg['pulse_info']

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):

        if 'timelist' in self.extra_args:
            self.expt_pts = self.extra_args['timelist']
        else:
            self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']

    def define_pulses(self,pt):

        if self.sideband == "ge":
            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q','pi_q_ef')
            self.psb.append('q:mm','general', self.flux_pulse_type, amp=self.amp, length=pt,freq=self.freq)
            self.psb.append('q','pi_q_ef')
            self.psb.append('q','half_pi', self.pulse_type, phase = 360.0*self.ramsey*pt/(1.0e9))
        else:
            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q:mm','general', self.flux_pulse_type, amp=self.amp, length=pt,freq=self.freq)
            self.psb.append('q','half_pi', self.pulse_type, phase = 360.0*self.ramsey*pt/(1.0e9))

class MultimodeEFRabiSweepSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg, **kwargs):
        self.extra_args={}
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)
        self.flux_freq = self.extra_args['flux_freq']
        if 'amp' in kwargs:
            self.amp = kwargs['amp']
        else:
            self.amp = self.expt_cfg['a']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses,sb_cool=False)


    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)


    def define_pulses(self,pt):
        self.psb.append('q','pi', self.pulse_type)
        self.psb.append('q','pi_q_ef')
        self.psb.append('q:mm','general', self.flux_pulse_type, amp=self.amp, length=pt,freq=self.flux_freq)
        self.psb.append('q','pi', self.pulse_type)
        self.psb.append('q','pi_q_ef', self.pulse_type)

class MultimodeT1Sequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']

        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value



        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):

        self.pulse_type =  self.expt_cfg['pulse_type']

        if 'mode' in self.extra_args:
            self.id = self.extra_args['mode']
        else:
            self.id = self.expt_cfg['id']

        self.flux_pulse_type = self.multimode_cfg[int(self.id)]['flux_pulse_type']



    def define_pulses(self,pt):
        self.psb.append('q','pi', self.pulse_type)
        self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(+self.id)]['a'], length= self.multimode_cfg[int(self.id)]['flux_pi_length'])
        self.psb.idle(pt)
        self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(+self.id)]['a'], length= self.multimode_cfg[int(self.id)]['flux_pi_length'],phase= 360.0*self.multimode_cfg[int(self.id)]['dc_offset_freq']*pt/(1.0e9))

class MultimodeGeneralEntanglementSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        if 'id1' in self.extra_args:
            self.id1 = self.extra_args['id1']
        if 'id2' in self.extra_args:
            self.id2 = self.extra_args['id2']
        if 'id3' in self.extra_args:
            self.id3 = self.extra_args['id3']
        if 'id4' in self.extra_args:
            self.id4 = self.extra_args['id4']
        if 'id5' in self.extra_args:
            self.id5 = self.extra_args['id5']
        if 'id6' in self.extra_args:
            self.id6 = self.extra_args['id6']
        if 'id7' in self.extra_args:
            self.id7 = self.extra_args['id7']
        if 'id8' in self.extra_args:
            self.id8 = self.extra_args['id8']
        if 'id9' in self.extra_args:
            self.id9 = self.extra_args['id9']
        if 'idm' in self.extra_args:
            self.idm = self.extra_args['idm']

        if 'GHZ' in self.extra_args:
            self.ghz = self.extra_args['GHZ']
        else:
            self.ghz = True

        print "id1 = "  + str(self.id1)
        print "id2 = "  + str(self.id2)
        print "id3 = "  + str(self.id3)
        print "idm = "  + str(self.idm)


        self.number = self.extra_args['number']
        if 'tomography' in self.extra_args:
            self.tomography = True
            self.tom_pulse = self.extra_args['tom_pulse']

        #self.multimode_cfg = cfg['multimodes']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        self.ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']


    def define_pulses(self,pt):

        if self.ghz:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= pt,phase=-90)

            self.psb.append('q','pi', self.pulse_type)
            for ii in np.arange(2,self.number):
                self.psb.append('q','pi_q_ef')
                self.psb.append('q,mm'+str(getattr(self, "id"+str(ii))),'pi_ef')
            self.psb.append('q,mm'+str(getattr(self, "id"+str(self.number))),'pi_ge',phase=-90)
            self.psb.append('q,mm'+str(self.idm),'pi_ge',phase=+90)
        else:
            self.psb.append('q','pi', self.pulse_type)
            flux_two_third_length = self.multimode_cfg[int(self.id1)]['flux_pi_length'] + (2/pi)*arcsin(sqrt(2.0/3.0))*( self.multimode_cfg[int(self.id1)]['flux_2pi_length'] -  self.multimode_cfg[int(self.id1)]['flux_pi_length'])
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= flux_two_third_length,phase=-90)
            self.psb.append('q,mm'+str(self.id2),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id2)]['a'], length= pt,phase=-90)
            self.psb.append('q,mm'+str(self.id3),'pi_ge',phase=-90)
            self.psb.append('q,mm'+str(self.idm),'pi_ge',phase=+90)
        if self.tomography:
            if self.tom_pulse == 0:
                # <X>
                self.psb.append('q','half_pi_y', self.pulse_type)
            elif self.tom_pulse == 1:
                # <Y>
                self.psb.append('q','half_pi', self.pulse_type)
            else:
                pass

class MultimodeEntanglementScalingSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
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
        #self.multimode_cfg = cfg['multimodes']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']


    def define_pulses(self,pt):
        # if self.expt_cfg['2_mode']:
        if self.number ==2:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= pt, phase=180)
            if self.expt_cfg['GHZ']:
               self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id2)]['a'], length=  self.multimode_cfg[int(self.id2)]['flux_pi_length'], phase=180)
            self.psb.append('q,mm'+str(self.idm),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.idm)]['a'], length=  self.multimode_cfg[int(self.idm)]['flux_pi_length'])

        elif self.number ==3:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= pt)
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q','pi_q_ef')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q,mm'+str(self.id3),'pi_ge')
            self.psb.append('q,mm'+str(self.idm),'pi_ge')

        elif self.number ==4:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= pt)
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q','pi_q_ef')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q','pi_q_ef')
            self.psb.append('q,mm'+str(self.id3),'pi_ef')
            self.psb.append('q,mm'+str(self.id4),'pi_ge')
            self.psb.append('q,mm'+str(self.idm),'pi_ge')

        elif self.number ==5:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= pt)
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q','pi_q_ef')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q','pi_q_ef')
            self.psb.append('q,mm'+str(self.id3),'pi_ef')
            self.psb.append('q','pi_q_ef')
            self.psb.append('q,mm'+str(self.id4),'pi_ef')
            self.psb.append('q,mm'+str(self.id5),'pi_ge')
            self.psb.append('q,mm'+str(self.idm),'pi_ge')

        elif self.number == 6:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= pt)
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q','pi_q_ef')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q','pi_q_ef')
            self.psb.append('q,mm'+str(self.id3),'pi_ef')
            self.psb.append('q','pi_q_ef')
            self.psb.append('q,mm'+str(self.id4),'pi_ef')
            self.psb.append('q','pi_q_ef')
            self.psb.append('q,mm'+str(self.id5),'pi_ef')
            self.psb.append('q,mm'+str(self.id6),'pi_ge')
            self.psb.append('q,mm'+str(self.idm),'pi_ge')

class MultimodeEntanglementSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']

        if 'id1' in self.extra_args:
            self.id1 = self.extra_args['id1']
        else:
            self.id1 = self.expt_cfg['id1']

        if 'id2' in self.extra_args:
            self.id2 = self.extra_args['id2']
        else:
            self.id2 = self.expt_cfg['id2']

        if 'id3' in self.extra_args:
            self.id3 = self.extra_args['id3']
        else:
            self.id3 = self.expt_cfg['id3']

        if 'idm' in self.extra_args:
            self.idm = self.extra_args['idm']
        else:
            self.idm = self.expt_cfg['idm']




        if '2_mode' in self.extra_args:
            self.two_mode = self.extra_args['2_mode']
            self.ghz =  self.extra_args['GHZ']
        else:
            self.two_mode = self.expt_cfg['2_mode']
            self.ghz =  self.expt_cfg['GHZ']

        if '3_mode' in self.extra_args:
            self.three_mode = self.extra_args['3_mode']
        else:
            self.three_mode = self.expt_cfg['3_mode']


        if 'shift_freq' in self.extra_args:
            self.shift_freq = self.extra_args['shift_freq']
        else:
            self.shift_freq = False

        if self.two_mode:

             print "(id1, id2, idm) = (" + str(self.id1) + ", " + str(self.id2) + ", " + str(self.idm) +")"

    def define_pulses(self,pt):
        #
        # if self.shift_freq:
        #     self.add_freq_q = self.multimode_cfg[self.id1]['shift']/(2.0)
        #     self.add_freq_sb_1 = -self.multimode_cfg[self.id1]['shift']/(2.0)
        #     if self.ghz:
        #         self.add_freq_sb_2 = -self.multimode_cfg[self.id1]['shift']/(2.0)


        if self.two_mode:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= pt, phase=180)
            if self.ghz:
               self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id2)]['a'], length=  self.multimode_cfg[int(self.id2)]['flux_pi_length'], phase=180)
            self.psb.append('q,mm'+str(self.idm),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.idm)]['a'], length=  self.multimode_cfg[int(self.idm)]['flux_pi_length'])


        elif self.three_mode:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= pt)
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q','pi_q_ef')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q,mm'+str(self.id3),'pi_ge')
            self.psb.append('q,mm'+str(self.idm),'pi_ge')

class MultimodePi_PiSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        if 'sweep_time' in self.extra_args:
            self.sweep_time = self.extra_args['sweep_time']
            self.time = self.extra_args['time']

        else:
            self.sweep_time = False
            self.time = 0

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses,sb_cool=False)


    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):


        if 'mode' in self.extra_args:
            self.id = self.extra_args['mode']
        else:
            self.id = self.expt_cfg['id']

        print "Target id: " +str(self.id)

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)

        self.offset_phase = self.pulse_cfg['gauss']['offset_phase']

    def define_pulses(self,pt):


        if self.expt_cfg['try_echo']:
            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'echo_pi_ge')
            if self.sweep_time:
                self.psb.idle(self.time)
            else:
                pass
            self.psb.append('q,mm'+str(self.id),'echo_pi_ge')
            self.psb.append('q','half_pi', self.pulse_type,phase = self.offset_phase + pt )


        else:

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase= pt/2.0 - self.offset_phase)
            if self.sweep_time:
                self.psb.idle(self.time)
            else:
                pass
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase= -pt/2.0 )
            self.psb.append('q','half_pi', self.pulse_type)

            # self.psb.append('q','half_pi', self.pulse_type)
            # self.psb.idle(self.time/2.0)
            # self.psb.append('q,mm'+str(self.id),'pi_ge',phase= pt/2.0 - self.offset_phase)
            # self.psb.append('q,mm'+str(self.id),'pi_ge',phase= -pt/2.0 )
            # self.psb.idle(self.time/2.0)
            # self.psb.append('q','half_pi', self.pulse_type)


            #
            # self.psb.append('q','half_pi', self.pulse_type)
            # self.psb.append('q,mm'+str(self.id),'pi_ge')
            # self.psb.idle(pt)
            # # self.psb.append('q,mm'+str(self.id),'pi_ge',phase= -pt/2.0 )
            # self.psb.append('q','half_pi', self.pulse_type)

class MultimodePi_PiTestSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value


        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        if 'mode' in self.extra_args:
            self.id = self.extra_args['mode']
        else:
            self.id = self.expt_cfg['id']

        print "Target id: " +str(self.id)

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)

        self.offset_phase = self.pulse_cfg['gauss']['offset_phase']


    def define_pulses(self,pt):

# Current pi-pi with phase of multimode state fixed

        # self.psb.append('q','half_pi', self.pulse_type)
        # self.psb.append('q,mm'+str(self.id),'pi_ge',phase= pt/2.0 + self.offset_phase)
        # self.psb.append('q,mm'+str(self.id),'pi_ge',phase= -pt/2.0 )
        # self.psb.append('q','half_pi', self.pulse_type)

        self.psb.append('q','half_pi', self.pulse_type)
        self.psb.append('q,mm'+str(self.id),'pi_ge',phase= pt/2.0 - self.offset_phase)
        self.psb.append('q,mm'+str(self.id),'pi_ge',phase= -pt/2.0 )
        self.psb.append('q','half_pi', self.pulse_type)

        # self.psb.append('q','half_pi', self.pulse_type)
        # self.psb.append('q,mm'+str(self.id),'pi_ge',phase= pt/2.0 - self.offset_phase)
        # self.psb.append('q,mm'+str(self.id),'pi_ge',phase= -pt/2.0 )
        # self.psb.append('q,mm'+str(self.id),'pi_ge',phase= self.multimode_cfg[self.id]['pi_pi_offset_phase']/2.0)
        # self.psb.append('q,mm'+str(self.id),'pi_ge',phase= -self.multimode_cfg[self.id]['pi_pi_offset_phase']/2.0 )
        # self.psb.append('q','half_pi', self.pulse_type)

        # self.psb.append('q','half_pi', self.pulse_type)
        # self.psb.append('q','half_pi', self.pulse_type,phase=self.offset_phase)
        # self.psb.append('q','half_pi', self.pulse_type,phase=2*self.offset_phase)
        # self.psb.append('q,mm'+str(self.id),'pi_ge',phase= pt/2.0 - 3*self.offset_phase)
        # self.psb.append('q,mm'+str(self.id),'pi_ge',phase= -pt/2.0 )
        # self.psb.append('q,mm'+str(self.id),'pi_ge',phase= self.multimode_cfg[self.id]['pi_pi_offset_phase']/2.0)
        # self.psb.append('q,mm'+str(self.id),'pi_ge',phase= -self.multimode_cfg[self.id]['pi_pi_offset_phase']/2.0 )
        # self.psb.append('q','half_pi', self.pulse_type)



# Older pi-pi with phase of multimode state fixed

        # self.psb.append('q','half_pi', self.pulse_type)
        # self.psb.append('q,mm'+str(self.id),'pi_ge')
        # self.psb.append('q,mm'+str(self.id),'pi_ge',phase = pt)
        # self.psb.append('q','half_pi', self.pulse_type,phase= self.offset_phase)


        # self.psb.append('q','half_pi', self.pulse_type)
        # self.psb.append('q,mm'+str(self.id),'pi_ge')
        # self.psb.append('q,mm'+str(self.id),'pi_ge')
        # self.psb.append('q','half_pi', self.pulse_type,phase=pt + self.offset_phase)

        # self.psb.append('q','half_pi', self.pulse_type)
        # self.psb.append('q,mm'+str(self.id),'pi_ge')
        # self.psb.append('q,mm'+str(self.id),'pi_ge')
        # self.psb.append('q','half_pi', self.pulse_type,phase=pt + self.offset_phase)


        # self.psb.append('q','half_pi', self.pulse_type)
        # self.psb.append('q,mm'+str(self.id),'pi_ge',phase= pt/2.0 )
        # self.psb.append('q,mm'+str(self.id),'pi_ge',phase= -pt/2.0 )
        # self.psb.append('q','half_pi', self.pulse_type, phase = + self.offset_phase)

class Multimode_Qubit_Mode_CZ_Offset_Sequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.flux_pulse_cfg=cfg['flux_pulse_info']
        self.multimode_cfg = cfg['multimodes']

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value


        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        if self.extra_args['offset_exp']==3 or self.extra_args['offset_exp']== 4 :

            self.expt_pts = arange(0,100,1)
        else:
            self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):

        if 'mode' in self.extra_args:
            self.id = self.extra_args['mode']
        else:
            self.id = self.expt_cfg['id']

        if 'mode2' in self.extra_args:
            self.id2 = self.extra_args['mode2']
        else:
            self.id2 = self.expt_cfg['id2']

        if 'offset_exp' in self.extra_args:
            self.offset_exp = self.extra_args['offset_exp']
        else:
            self.offset_exp = self.expt_cfg['offset_exp']

        if 'load_photon' in self.extra_args:
            self.load_photon = self.extra_args['load_photon']
        else:
            self.load_photon = self.expt_cfg['load_photon']

        if 'fix_qubit_freq' in self.extra_args:
            self.fix_qubit_freq = self.extra_args['fix_qubit_freq']
        else:
            self.fix_qubit_freq = self.expt_cfg['fix_qubit_freq']
        print "Target id: " +str(self.id)

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)

        self.offset_phase = self.pulse_cfg['gauss']['offset_phase']


    def define_pulses(self,pt):
        #Calibration of DC offset due to the 2pi ef sideband
        if self.offset_exp == 0:
            self.psb.append('q','half_pi', self.pulse_type)

            if self.expt_cfg['mode_mode']:
                self.psb.append('q,mm'+str(self.id2),'pi_ge')
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            self.psb.append('q,mm'+str(self.id),'pi_ef')
            self.psb.append('q,mm'+str(self.id),'pi_ef')

            if self.expt_cfg['mode_mode']:
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=pt)
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase)

        #State dependent phase shift cal
        elif self.offset_exp == 1:

            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','half_pi', self.pulse_type)

            if self.expt_cfg['mode_mode']:

                self.psb.append('q,mm'+str(self.id2),'pi_ge')
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            self.psb.append('q,mm'+str(self.id),'pi_ef')
            self.psb.append('q,mm'+str(self.id),'pi_ef',phase=pt)

            if self.expt_cfg['mode_mode']:

                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id]['qubit_mode_ef_offset_0'])
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase)

        elif self.offset_exp == 2:
            if self.load_photon == 1:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','half_pi', self.pulse_type)

            if self.expt_cfg['mode_mode']:

                self.psb.append('q,mm'+str(self.id2),'pi_ge')
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            self.psb.append('q,mm'+str(self.id),'pi_ef')
            self.psb.append('q,mm'+str(self.id),'pi_ef',phase=self.multimode_cfg[self.id]['qubit_mode_ef_offset_1'] )

            if self.expt_cfg['mode_mode']:
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id]['qubit_mode_ef_offset_0'])
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            self.psb.append('q','half_pi', self.pulse_type, phase = pt + self.offset_phase)


        elif self.offset_exp == 3:

            if self.load_photon == 1:

                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a'], length=pt,freq=self.pulse_cfg['gauss']['iq_freq'])

            if self.expt_cfg['mode_mode']:

                self.psb.append('q,mm'+str(self.id2),'pi_ge')
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            self.psb.append('q,mm'+str(self.id),'pi_ef')
            self.psb.append('q,mm'+str(self.id),'pi_ef',phase=self.multimode_cfg[self.id]['qubit_mode_ef_offset_1'] )

            if self.expt_cfg['mode_mode']:

                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id]['qubit_mode_ef_offset_0'])
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a'], length=pt,freq=self.pulse_cfg['gauss']['iq_freq'],phase =  self.offset_phase)


        elif self.offset_exp == 4:
            #
            if self.load_photon==1:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a'], length=pt,freq=self.pulse_cfg['gauss']['iq_freq'])

            self.psb.append('q,mm'+str(self.id2),'pi_ge')

            cphase_len = 2*(self.multimode_cfg[self.id2]['flux_pi_length'] + self.multimode_cfg[self.id]['flux_pi_length_ef']) + 4*self.flux_pulse_cfg['spacing']
            self.psb.idle(cphase_len)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a'], length=pt,freq=self.pulse_cfg['gauss']['iq_freq'],phase = self.multimode_cfg[self.id]['qubit_mode_ef_offset_0'] + self.offset_phase)

        elif self.offset_exp == 5:


            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','pi', self.pulse_type)

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id]['pi_pi_offset_phase'])


            self.psb.append('q,mm'+str(self.id),'pi_ef')
            self.psb.append('q,mm'+str(self.id),'pi_ef',phase=self.multimode_cfg[self.id]['qubit_mode_ef_offset_1'] )


            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id]['pi_pi_offset_phase'])

            self.psb.append('q','half_pi', self.pulse_type, phase = pt + self.multimode_cfg[self.id]['qubit_mode_ef_offset_0'] + self.offset_phase)

class Multimode_Qubit_Mode_CZ_V2_Offset_Sequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.flux_pulse_cfg=cfg['flux_pulse_info']
        self.multimode_cfg = cfg['multimodes']

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value


        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        if 'offset_exp' in self.extra_args:
            if self.extra_args['offset_exp']==3 :
                self.expt_pts = arange(0,100,1)
            else:
                self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])
        else:
            if self.expt_cfg['offset_exp']==3 :
                self.expt_pts = arange(0,100,1)
            else:
                self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):

        if 'mode' in self.extra_args:
            self.id = self.extra_args['mode']
        else:
            self.id = self.expt_cfg['id']

        if 'offset_exp' in self.extra_args:
            self.offset_exp = self.extra_args['offset_exp']
        else:
            self.offset_exp = self.expt_cfg['offset_exp']

        if 'load_photon' in self.extra_args:
            self.load_photon = self.extra_args['load_photon']
        else:
            self.load_photon = self.expt_cfg['load_photon']

        print "Target id: " +str(self.id)

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)

        self.offset_phase = self.pulse_cfg['gauss']['offset_phase']

        self.cz_dc_phase = self.multimode_cfg[self.id]['cz_dc_phase']



    def define_pulses(self,pt):
        #Calibration of DC offset due to the 2pi ge sideband
        if self.offset_exp == 0:
            # self.psb.append('q','half_pi', self.pulse_type)
            # cphase_v2(self.psb,'q',self.id,cz_dc_phase=pt, cz_phase = 0)
            # self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase)

            self.psb.append('q','half_pi', self.pulse_type)
            cphase_v1(self.psb,'q',self.id,cz_phase = 0)
            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase + pt)

        #Calibration of phase for second conditional sideband
        elif self.offset_exp == 1:
            # self.psb.append('q','pi', self.pulse_type)
            # self.psb.append('q,mm'+str(self.id),'pi_ge')
            #
            # self.psb.append('q','half_pi', self.pulse_type)
            # cphase_v2(self.psb,'q',self.id,cz_dc_phase=self.cz_dc_phase,cz_phase = pt)
            # self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase)


            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','half_pi', self.pulse_type)
            cphase_v1(self.psb,'q',self.id,cz_phase = pt)
            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase + self.cz_dc_phase)

        # Verify that the phases works
        elif self.offset_exp == 2:
            # if self.load_photon:
            #     self.psb.append('q','pi', self.pulse_type)
            #     self.psb.append('q,mm'+str(self.id),'pi_ge')
            #
            # self.psb.append('q','half_pi', self.pulse_type)
            # cphase_v2(self.psb,'q',self.id,cz_dc_phase= self.cz_dc_phase, cz_phase = self.multimode_cfg[self.id]['cz_phase'])
            # self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase + pt)


            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','half_pi', self.pulse_type)
            cphase_v1(self.psb,'q',self.id, cz_phase = self.multimode_cfg[self.id]['cz_phase'])
            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase + self.cz_dc_phase + pt)

        # replace half pi by varying length
        elif self.offset_exp == 3:
            # if self.load_photon:
            #     self.psb.append('q','pi', self.pulse_type)
            #     self.psb.append('q,mm'+str(self.id),'pi_ge')
            #
            # self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a']
            #                 , length=pt,freq=self.pulse_cfg['gauss']['iq_freq'])
            # cphase_v2(self.psb,'q',self.id,cz_dc_phase= self.cz_dc_phase, cz_phase = self.multimode_cfg[self.id]['cz_phase'])
            # self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a'],
            #                 length=pt,freq=self.pulse_cfg['gauss']['iq_freq'], phase= self.offset_phase)
            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a']
                            , length=pt,freq=self.pulse_cfg['gauss']['iq_freq'])
            cphase_v1(self.psb,'q',self.id, cz_phase = self.multimode_cfg[self.id]['cz_phase'])
            self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a'],
                            length=pt,freq=self.pulse_cfg['gauss']['iq_freq'], phase= self.offset_phase+ self.cz_dc_phase)

class Multimode_Mode_Mode_CNOT_V2_Offset_Sequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.cfg = cfg
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.flux_pulse_cfg=cfg['flux_pulse_info']
        self.multimode_cfg = cfg['multimodes']

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value


        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses, sb_cool = False)


    def define_points(self):

        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):

        if 'mode' in self.extra_args:
            self.id = self.extra_args['mode']
        else:
            self.id = self.expt_cfg['id']

        if 'mode2' in self.extra_args:
            self.id2 = self.extra_args['mode2']
        else:
            self.id2 = self.expt_cfg['id2']

        if 'offset_exp' in self.extra_args:
            self.offset_exp = self.extra_args['offset_exp']
        else:
            self.offset_exp = self.expt_cfg['offset_exp']

        if 'load_photon' in self.extra_args:
            self.load_photon = self.extra_args['load_photon']
        else:
            self.load_photon = self.expt_cfg['load_photon']

        if 'number' in self.extra_args:
            self.number = self.extra_args['number']
        else:
            self.number = self.expt_cfg['number']

        print "Target id: " +str(self.id)
        print "Target id2: " +str(self.id2)

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)

        self.offset_phase = self.pulse_cfg['gauss']['offset_phase']

        self.mode_mode_cnot_dc_phase = self.cfg['mode_mode_offset']['cnot_dc_phase']
        self.mode_mode_cnot_phase = self.cfg['mode_mode_offset']['cnot_phase']
        self.mode_mode_cnot_phase2 = self.cfg['mode_mode_offset']['cnot_phase2']


    def define_pulses(self,pt):
        #Calibration of DC offset due to the 2pi ge sideband
        if self.offset_exp == 0:

            self.psb.append('q','half_pi', self.pulse_type)

            self.psb.append('q,mm'+str(self.id),'pi_ge')

            cnot_v1(self.psb,'q',self.id,cnot_phase = 0)

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=self.multimode_cfg[self.id]['pi_pi_offset_phase'])

            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase + pt)

        #Calibration of phase for second conditional sideband
        elif self.offset_exp == 1:

            self.psb.append('q','half_pi', self.pulse_type)

            self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','pi', self.pulse_type)

            cnot_v1(self.psb,'q',self.id,cnot_phase = pt)

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=self.multimode_cfg[self.id]['pi_pi_offset_phase'])

            #print self.mode_mode_cnot_dc_phase[self.id][self.id2]
            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase + self.mode_mode_cnot_dc_phase[self.id][self.id2])

        # Verify that the phases works
        elif self.offset_exp == 2:


            self.psb.append('q','half_pi', self.pulse_type)

            self.psb.append('q,mm'+str(self.id),'pi_ge')

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)

            cnot_v1(self.psb,'q',self.id,cnot_phase = self.mode_mode_cnot_phase[self.id][self.id2])

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=self.multimode_cfg[self.id]['pi_pi_offset_phase'])

            #print self.mode_mode_cnot_dc_phase[self.id][self.id2]
            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase + self.mode_mode_cnot_dc_phase[self.id][self.id2]+pt)

        # replace load photon pi to half pi
        elif self.offset_exp == 3:

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge')

            if self.load_photon:
                self.psb.append('q','half_pi', self.pulse_type)

            cnot_v1(self.psb,'q',self.id,cnot_phase = self.mode_mode_cnot_phase[self.id][self.id2])

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=self.multimode_cfg[self.id]['pi_pi_offset_phase'])

            #print self.mode_mode_cnot_dc_phase[self.id][self.id2]
            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase + self.mode_mode_cnot_dc_phase[self.id][self.id2]+pt)

        # Mode mode CNOT gate phase tests

        elif self.offset_exp == 4:

        # Offset Phase Calibration Experiment
            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge')

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id2),'pi_ge')
                cnot_phase = pt
                sideband_phase = self.mode_mode_cnot_dc_phase[self.id][self.id2]

            else:
                cnot_phase = 0
                sideband_phase = pt
            # cnot_v1(self.psb,self.id2,self.id,cnot_phase=cnot_phase)
            cnot_v2(self.psb,self.id2,self.id,cnot_phase=0,efsbphase_0=cnot_phase)

            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=self.multimode_cfg[self.id]['pi_pi_offset_phase'])
            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase + sideband_phase)

        elif self.offset_exp == 5:
            cnot_phase = self.mode_mode_cnot_phase[self.id][self.id2]
            sideband_phase = self.mode_mode_cnot_dc_phase[self.id][self.id2]

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge')

            cnot_v2(self.psb,self.id2,self.id,cnot_phase=0,efsbphase_0=cnot_phase,efsbphase_1=pt)
            cnot_v2(self.psb,self.id2,self.id,cnot_phase=0,efsbphase_0=cnot_phase,efsbphase_1=pt)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])
            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase)

        elif self.offset_exp == 6:
            cnot_phase = self.mode_mode_cnot_phase[self.id][self.id2]
            cnot_phase2 = self.mode_mode_cnot_phase2[self.id][self.id2]
            sideband_phase = self.mode_mode_cnot_dc_phase[self.id][self.id2]

            if 'number' not in self.extra_args:
                self.number=1

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge')

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id2),'pi_ge')
                prefactor = self.number
            else:
                prefactor = self.number

            for i in arange(self.number):
                cnot_v2(self.psb,self.id2,self.id,cnot_phase=0,efsbphase_0=cnot_phase,efsbphase_1=cnot_phase2)

            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=self.multimode_cfg[self.id]['pi_pi_offset_phase']+180.0)
            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase + prefactor*sideband_phase + pt)

        elif self.offset_exp == 7:
            cnot_phase = self.mode_mode_cnot_phase[self.id][self.id2]
            cnot_phase2 = self.mode_mode_cnot_phase2[self.id][self.id2]
            sideband_phase = self.mode_mode_cnot_dc_phase[self.id][self.id2] +pt

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge')

            for i in arange(self.number):
                cnot_v2(self.psb,self.id2,self.id,cnot_phase=0,efsbphase_0=cnot_phase,efsbphase_1=cnot_phase2)
                cnot_v2(self.psb,self.id2,self.id,cnot_phase=0,efsbphase_0=cnot_phase,efsbphase_1=cnot_phase2)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase']+180.0)
            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase + pt)

        elif self.offset_exp == 9:


            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge')

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id),'pi_ge')

            cnot_v1(self.psb,self.id2,self.id,cnot_phase=0)

            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=self.multimode_cfg[self.id]['pi_pi_offset_phase']+pt)

        elif self.offset_exp == 10:
            self.psb.append('q,mm1','echo_pi_ge_test')

class Multimode_Mode_Mode_CZ_V3_Offset_Sequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.cfg = cfg
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.flux_pulse_cfg=cfg['flux_pulse_info']
        self.multimode_cfg = cfg['multimodes']

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        if 'sb_cool' in self.extra_args:
            sb_cool = self.extra_args['sb_cool']
        else:
            sb_cool = True


        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses,sb_cool = False)


    def define_points(self):

        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):

        if 'mode' in self.extra_args:
            self.id = self.extra_args['mode']
        else:
            self.id = self.expt_cfg['id']

        if 'mode2' in self.extra_args:
            self.id2 = self.extra_args['mode2']
        else:
            self.id2 = self.expt_cfg['id2']

        if 'offset_exp' in self.extra_args:
            self.offset_exp = self.extra_args['offset_exp']
        else:
            self.offset_exp = self.expt_cfg['offset_exp']

        if 'load_photon' in self.extra_args:
            self.load_photon = self.extra_args['load_photon']
        else:
            self.load_photon = self.expt_cfg['load_photon']

        if 'number' in self.extra_args:
            self.number = self.extra_args['number']
        else:
            self.number = self.expt_cfg['number']

        if 'include_cz_correction' in self.extra_args:
            self.include_cz_correction = self.extra_args['include_cz_correction']
        else:
            self.include_cz_correction = True

        print "Target id: " +str(self.id)
        print "Target id2: " +str(self.id2)

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)

        self.offset_phase = self.pulse_cfg['gauss']['offset_phase']

        self.mode_mode_cz_dc_phase = self.cfg['mode_mode_offset']['cz_dc_phase']
        self.mode_mode_cz_phase = self.cfg['mode_mode_offset']['cz_phase']
        self.mode_mode_cz_prep_phase = self.cfg['mode_mode_offset']['cz_prep_phase']
        self.mode_mode_cz_phase2 = self.cfg['mode_mode_offset']['cz_phase2']


    def define_pulses(self,pt):

        if self.offset_exp == 0:

        # Offset Phase Calibration Experiment
            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge', phase = self.multimode_cfg[self.id]['pi_pi_offset_phase']/2.0 - self.offset_phase)

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

                if self.include_cz_correction:
                    cz_prep_phase = pt
                    # sideband_phase = self.mode_mode_cz_dc_phase[self.id][self.id2]
                    sideband_phase = 0
                    cz_phase2 = self.mode_mode_cz_phase2[self.id][self.id2]

                else:
                    cz_prep_phase = self.mode_mode_cz_prep_phase[self.id][self.id2]
                    cz_phase = pt
                    sideband_phase=0
                    # sideband_phase = 2*self.mode_mode_cz_dc_phase[self.id][self.id2]
                    cz_phase2 = self.mode_mode_cz_phase2[self.id][self.id2]

            else:
                cz_phase = 0
                cz_prep_phase=0
                sideband_phase = pt
                cz_phase2= self.mode_mode_cz_phase2[self.id][self.id2]

            cphase_v3(self.psb,self.id2,self.id,efsbphase_0=cz_prep_phase,efsbphase_1=cz_phase2,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
            if not self.include_cz_correction:
                cphase_v3(self.psb,self.id2,self.id,efsbphase_0=cz_phase,efsbphase_1=cz_phase2,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=-self.multimode_cfg[self.id]['pi_pi_offset_phase']/(2.0))
            self.psb.append('q','half_pi', self.pulse_type,phase=sideband_phase)

        elif self.offset_exp == 1:
            cz_phase = self.mode_mode_cz_phase[self.id][self.id2]
            sideband_phase = self.mode_mode_cz_dc_phase[self.id][self.id2]

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge', phase= self.multimode_cfg[self.id2]['pi_pi_offset_phase']/(2.0) - self.offset_phase)

            cphase_v3(self.psb,self.id2,self.id,efsbphase_0=cz_phase,efsbphase_1=pt,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/(2.0))
            self.psb.append('q','half_pi', self.pulse_type,phase=sideband_phase)

        elif self.offset_exp == 2:
            cz_prep_phase = self.mode_mode_cz_prep_phase[self.id][self.id2]
            cz_phase = self.mode_mode_cz_phase[self.id][self.id2]
            cz_phase2 = self.mode_mode_cz_phase2[self.id][self.id2]
            #
            if 'number' not in self.extra_args:
                self.number=1

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge', phase = self.multimode_cfg[self.id]['pi_pi_offset_phase']/2.0 - self.offset_phase)

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
                prefactor = (self.number + 1)
                sideband_phase =0
                # sideband_phase = self.mode_mode_cz_dc_phase[self.id][self.id2]
            else:
                prefactor = (self.number + 1)
                sideband_phase = self.mode_mode_cz_dc_phase[self.id][self.id2]

            cphase_v3(self.psb,self.id2,self.id,efsbphase_0=cz_prep_phase,efsbphase_1=cz_phase2,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
            for i in arange(self.number):
                cphase_v3(self.psb,self.id2,self.id,efsbphase_0=cz_phase,efsbphase_1=cz_phase2,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)


            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=-self.multimode_cfg[self.id]['pi_pi_offset_phase']/(2.0))
            self.psb.append('q','half_pi', self.pulse_type, phase= prefactor*sideband_phase + pt)

        elif self.offset_exp == 3:

            cz_prep_phase = self.mode_mode_cz_prep_phase[self.id][self.id2]
            cz_phase = self.mode_mode_cz_phase[self.id][self.id2]
            cz_phase2 = self.mode_mode_cz_phase2[self.id][self.id2]
            # sideband_phase = self.mode_mode_cnot_dc_phase[self.id][self.id2]
            sideband_phase =0
            prefactor = self.number

            if 'number' not in self.extra_args:
                self.number=1

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge', phase = self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 - self.offset_phase)

            for i in arange(self.number):
                cphase_v3(self.psb,self.id2,self.id,efsbphase_0=cz_phase,efsbphase_1=cz_phase2,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/(2.0))
            self.psb.append('q','half_pi', self.pulse_type, phase= prefactor*sideband_phase + pt)

        elif self.offset_exp == 4:

            # puts the MM in state |11> or |01>, measures id, state = |id2, id>

            cz_prep_phase = self.mode_mode_cz_prep_phase[self.id][self.id2]
            cz_phase = self.mode_mode_cz_phase[self.id][self.id2]
            cz_phase2 = self.mode_mode_cz_phase2[self.id][self.id2]
            #
            if 'number' not in self.extra_args:
                self.number=1

            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge', phase = self.multimode_cfg[self.id]['pi_pi_offset_phase']/2.0)

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
                prefactor = self.number
                sideband_phase =0
            else:
                prefactor = self.number
                sideband_phase = self.mode_mode_cz_dc_phase[self.id][self.id2]

            cphase_v3(self.psb,self.id2,self.id,efsbphase_0=cz_prep_phase,efsbphase_1=cz_phase2,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
            for i in arange(self.number):
                cphase_v3(self.psb,self.id2,self.id,efsbphase_0=cz_phase,efsbphase_1=cz_phase2,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)


            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=-self.multimode_cfg[self.id]['pi_pi_offset_phase']/(2.0) - pt)

        elif self.offset_exp ==5:
            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge', phase = self.multimode_cfg[self.id]['pi_pi_offset_phase']/2.0 - self.offset_phase)

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=-self.multimode_cfg[self.id]['pi_pi_offset_phase']/(2.0))
            self.psb.append('q','half_pi', self.pulse_type,phase= pt)

class Multimode_Mode_Mode_CZ_V3_Offset_Sequence_Debug(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.cfg = cfg
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.flux_pulse_cfg=cfg['flux_pulse_info']
        self.multimode_cfg = cfg['multimodes']

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        if 'sb_cool' in self.extra_args:
            sb_cool = self.extra_args['sb_cool']
        else:
            sb_cool = True


        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses,sb_cool = False)


    def define_points(self):

        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):

        if 'mode' in self.extra_args:
            self.id = self.extra_args['mode']
        else:
            self.id = self.expt_cfg['id']

        if 'mode2' in self.extra_args:
            self.id2 = self.extra_args['mode2']
        else:
            self.id2 = self.expt_cfg['id2']

        if 'offset_exp' in self.extra_args:
            self.offset_exp = self.extra_args['offset_exp']
        else:
            self.offset_exp = self.expt_cfg['offset_exp']

        if 'load_photon' in self.extra_args:
            self.load_photon = self.extra_args['load_photon']
        else:
            self.load_photon = self.expt_cfg['load_photon']

        if 'number' in self.extra_args:
            self.number = self.extra_args['number']
        else:
            self.number = self.expt_cfg['number']

        if 'include_cz_correction' in self.extra_args:
            self.include_cz_correction = self.extra_args['include_cz_correction']
        else:
            self.include_cz_correction = True

        print "Target id: " +str(self.id)
        print "Target id2: " +str(self.id2)

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)

        self.offset_phase = self.pulse_cfg['gauss']['offset_phase']

        self.mode_mode_cz_dc_phase = self.cfg['mode_mode_offset']['cz_dc_phase']
        self.mode_mode_cz_phase = self.cfg['mode_mode_offset']['cz_phase']
        self.mode_mode_cz_prep_phase = self.cfg['mode_mode_offset']['cz_prep_phase']
        self.mode_mode_cz_phase2 = self.cfg['mode_mode_offset']['cz_phase2']


    def define_pulses(self,pt):

        if self.offset_exp == 0:

        # Offset Phase Calibration Experiment
            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge', phase = self.multimode_cfg[self.id]['pi_pi_offset_phase']/2.0 - self.offset_phase)

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

                if self.include_cz_correction:
                    cz_prep_phase = 0
                    # sideband_phase = self.mode_mode_cz_dc_phase[self.id][self.id2]
                    sideband_phase = pt
                    # cz_phase2 = self.mode_mode_cz_phase2[self.id][self.id2]

                    cz_phase2 = 0

                else:
                    cz_prep_phase = self.mode_mode_cz_prep_phase[self.id][self.id2]
                    cz_phase = pt
                    sideband_phase=0
                    # sideband_phase = 2*self.mode_mode_cz_dc_phase[self.id][self.id2]
                    cz_phase2 = self.mode_mode_cz_phase2[self.id][self.id2]

            else:
                cz_phase = 0
                cz_prep_phase=0
                sideband_phase = pt
                cz_phase2= self.mode_mode_cz_phase2[self.id][self.id2]

            cphase_v3(self.psb,self.id2,self.id,efsbphase_0=cz_prep_phase,efsbphase_1=cz_phase2,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
            if not self.include_cz_correction:
                cphase_v3(self.psb,self.id2,self.id,efsbphase_0=cz_phase,efsbphase_1=cz_phase2,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=-self.multimode_cfg[self.id]['pi_pi_offset_phase']/(2.0))
            self.psb.append('q','half_pi', self.pulse_type,phase=sideband_phase)

        elif self.offset_exp == 1:
            cz_phase = self.mode_mode_cz_phase[self.id][self.id2]
            sideband_phase = self.mode_mode_cz_dc_phase[self.id][self.id2]

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge', phase= self.multimode_cfg[self.id2]['pi_pi_offset_phase']/(2.0) - self.offset_phase)

            cphase_v3(self.psb,self.id2,self.id,efsbphase_0=cz_phase,efsbphase_1=pt,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/(2.0))
            self.psb.append('q','half_pi', self.pulse_type,phase=sideband_phase)

class Multimode_Mode_Mode_CZ_V4_Offset_Sequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.cfg = cfg
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.flux_pulse_cfg=cfg['flux_pulse_info']
        self.multimode_cfg = cfg['multimodes']

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value


        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):

        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):

        if 'mode' in self.extra_args:
            self.id = self.extra_args['mode']
        else:
            self.id = self.expt_cfg['id']

        if 'mode2' in self.extra_args:
            self.id2 = self.extra_args['mode2']
        else:
            self.id2 = self.expt_cfg['id2']

        if 'offset_exp' in self.extra_args:
            self.offset_exp = self.extra_args['offset_exp']
        else:
            self.offset_exp = self.expt_cfg['offset_exp']

        if 'load_photon' in self.extra_args:
            self.load_photon = self.extra_args['load_photon']
        else:
            self.load_photon = self.expt_cfg['load_photon']

        if 'number' in self.extra_args:
            self.number = self.extra_args['number']
        else:
            self.number = self.expt_cfg['number']

        if 'include_cz_correction' in self.extra_args:
            self.include_cz_correction = self.extra_args['include_cz_correction']
        else:
            self.include_cz_correction = True

        print "Target id: " +str(self.id)
        print "Target id2: " +str(self.id2)

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)

        self.offset_phase = self.pulse_cfg['gauss']['offset_phase']

        self.mode_mode_cz_dc_phase = self.cfg['mode_mode_offset']['cz_dc_phase']
        self.mode_mode_cz_phase = self.cfg['mode_mode_offset']['cz_phase']
        self.mode_mode_cz_prep_phase = self.cfg['mode_mode_offset']['cz_prep_phase']
        self.mode_mode_cz_phase2 = self.cfg['mode_mode_offset']['cz_phase2']

        if self.load_photon:
            self.disp_id = self.multimode_cfg[self.id]['shift']
            self.disp_id2 = self.multimode_cfg[self.id2]['shift']
        else:
            self.disp_id = 0
            self.disp_id2 = 0


    def define_pulses(self,pt):

        if self.offset_exp == 0:

        # Offset Phase Calibration Experiment
            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge', phase = self.multimode_cfg[self.id]['pi_pi_offset_phase']/2.0 - self.offset_phase)

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type,add_freq=self.disp_id/2.0)
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,add_freq=-self.disp_id/(2.0) )

                if self.include_cz_correction:
                    cz_prep_phase = pt
                    # sideband_phase = self.mode_mode_cz_dc_phase[self.id][self.id2]
                    sideband_phase = 0
                    cz_phase2 = self.mode_mode_cz_phase2[self.id][self.id2]

                else:
                    cz_prep_phase = self.mode_mode_cz_prep_phase[self.id][self.id2]
                    cz_phase = pt
                    sideband_phase=0
                    # sideband_phase = 2*self.mode_mode_cz_dc_phase[self.id][self.id2]
                    cz_phase2 = self.mode_mode_cz_phase2[self.id][self.id2]

            else:
                disp_id = 0
                disp_id2 = 0
                cz_phase = 0
                cz_prep_phase=0
                sideband_phase = pt
                cz_phase2= self.mode_mode_cz_phase2[self.id][self.id2]

            cphase_v3(self.psb,self.id2,self.id,efsbphase_0=cz_prep_phase,efsbphase_1=cz_phase2,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,add_freq= -self.disp_id/(2.0))
            if not self.include_cz_correction:
                cphase_v3(self.psb,self.id2,self.id,efsbphase_0=cz_phase,efsbphase_1=cz_phase2,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,add_freq= -self.disp_id/(2.0))

            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=-self.multimode_cfg[self.id]['pi_pi_offset_phase']/(2.0),add_freq= -self.disp_id2)
            self.psb.append('q','half_pi', self.pulse_type,phase=sideband_phase,add_freq = +self.disp_id2)

        elif self.offset_exp == 1:
            cz_phase = self.mode_mode_cz_phase[self.id][self.id2]
            sideband_phase = self.mode_mode_cz_dc_phase[self.id][self.id2]

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge', phase= self.multimode_cfg[self.id2]['pi_pi_offset_phase']/(2.0) - self.offset_phase)

            cphase_v3(self.psb,self.id2,self.id,efsbphase_0=cz_phase,efsbphase_1=pt,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/(2.0))
            self.psb.append('q','half_pi', self.pulse_type,phase=sideband_phase)

        elif self.offset_exp == 2:
            cz_prep_phase = self.mode_mode_cz_prep_phase[self.id][self.id2]
            cz_phase = self.mode_mode_cz_phase[self.id][self.id2]
            cz_phase2 = self.mode_mode_cz_phase2[self.id][self.id2]
            #
            if 'number' not in self.extra_args:
                self.number=1

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge', phase = self.multimode_cfg[self.id]['pi_pi_offset_phase']/2.0 - self.offset_phase)

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type,add_freq=self.disp_id/(2.0))
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,add_freq=-self.disp_id/(2.0))
                prefactor = (self.number + 1)
                sideband_phase =0
                # sideband_phase = self.mode_mode_cz_dc_phase[self.id][self.id2]
            else:
                prefactor = (self.number + 1)
                sideband_phase = self.mode_mode_cz_dc_phase[self.id][self.id2]

            cphase_v3(self.psb,self.id2,self.id,efsbphase_0=cz_prep_phase,efsbphase_1=cz_phase2,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,add_freq=-self.disp_id/2.0)
            for i in arange(self.number):
                cphase_v3(self.psb,self.id2,self.id,efsbphase_0=cz_phase,efsbphase_1=cz_phase2,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,add_freq=-self.disp_id/2.0)


            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=-self.multimode_cfg[self.id]['pi_pi_offset_phase']/(2.0),add_freq=-self.disp_id2/2.0)
            self.psb.append('q','half_pi', self.pulse_type, phase= prefactor*sideband_phase + pt,add_freq= + self.disp_id2/2.0)

        elif self.offset_exp == 3:

            cz_prep_phase = self.mode_mode_cz_prep_phase[self.id][self.id2]
            cz_phase = self.mode_mode_cz_phase[self.id][self.id2]
            cz_phase2 = self.mode_mode_cz_phase2[self.id][self.id2]
            # sideband_phase = self.mode_mode_cnot_dc_phase[self.id][self.id2]
            sideband_phase =0
            prefactor = self.number

            if 'number' not in self.extra_args:
                self.number=1

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge', phase = self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 - self.offset_phase)

            for i in arange(self.number):
                cphase_v3(self.psb,self.id2,self.id,efsbphase_0=cz_phase,efsbphase_1=cz_phase2,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/(2.0))
            self.psb.append('q','half_pi', self.pulse_type, phase= prefactor*sideband_phase + pt)

        elif self.offset_exp == 4:

            # puts the MM in state |11> or |01>, measures id, state = |id2, id>

            cz_prep_phase = self.mode_mode_cz_prep_phase[self.id][self.id2]
            cz_phase = self.mode_mode_cz_phase[self.id][self.id2]
            cz_phase2 = self.mode_mode_cz_phase2[self.id][self.id2]
            #
            if 'number' not in self.extra_args:
                self.number=1

            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge', phase = self.multimode_cfg[self.id]['pi_pi_offset_phase']/2.0)

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type,add_freq=self.disp_id/2.0)
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,add_freq=-self.disp_id/2.0)
                prefactor = self.number
                sideband_phase =0
            else:
                prefactor = self.number
                sideband_phase = self.mode_mode_cz_dc_phase[self.id][self.id2]

            cphase_v3(self.psb,self.id2,self.id,efsbphase_0=cz_prep_phase,efsbphase_1=cz_phase2,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,add_freq=-self.disp_id/2.0)
            for i in arange(self.number):
                cphase_v3(self.psb,self.id2,self.id,efsbphase_0=cz_phase,efsbphase_1=cz_phase2,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,add_freq=-self.disp_id/2.0)


            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=-self.multimode_cfg[self.id]['pi_pi_offset_phase']/(2.0) - pt,add_freq=self.disp_id2/2.0)

        elif self.offset_exp ==5:
            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge', phase = self.multimode_cfg[self.id]['pi_pi_offset_phase']/2.0 - self.offset_phase)

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type,self.disp_id/2.0)
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,add_freq=-self.disp_id/2.0)

            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=-self.multimode_cfg[self.id]['pi_pi_offset_phase']/(2.0),add_freq=-self.disp_id2/2.0)
            self.psb.append('q','half_pi', self.pulse_type,phase= pt,add_freq=self.disp_id2/2.0)

class Multimode_Mode_Mode_CNOT_V3_Offset_Sequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.cfg = cfg
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.flux_pulse_cfg=cfg['flux_pulse_info']
        self.multimode_cfg = cfg['multimodes']

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        if 'sb_cool' in self.extra_args:
            sb_cool = self.extra_args['sb_cool']
        else:
            sb_cool = True

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses,sb_cool = False)


    def define_points(self):

        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):

        if 'mode' in self.extra_args:
            self.id = self.extra_args['mode']
        else:
            self.id = self.expt_cfg['id']

        if 'mode2' in self.extra_args:
            self.id2 = self.extra_args['mode2']
        else:
            self.id2 = self.expt_cfg['id2']

        if 'offset_exp' in self.extra_args:
            self.offset_exp = self.extra_args['offset_exp']
        else:
            self.offset_exp = self.expt_cfg['offset_exp']

        if 'load_photon' in self.extra_args:
            self.load_photon = self.extra_args['load_photon']
        else:
            self.load_photon = self.expt_cfg['load_photon']

        if 'number' in self.extra_args:
            self.number = self.extra_args['number']
        else:
            self.number = self.expt_cfg['number']

        if 'include_cz_correction' in self.extra_args:
            self.include_cz_correction = self.extra_args['include_cz_correction']
        else:
            self.include_cz_correction = True

        print "Target id: " +str(self.id)
        print "Target id2: " +str(self.id2)

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)

        self.offset_phase = self.pulse_cfg['gauss']['offset_phase']

        self.mode_mode_cnot_dc_phase = self.cfg['mode_mode_offset']['cnot_dc_phase']
        self.mode_mode_cnot_prep_phase = self.cfg['mode_mode_offset']['cnot_prep_phase']
        self.mode_mode_cnot_phase = self.cfg['mode_mode_offset']['cnot_phase']
        self.mode_mode_cnot_phase2 = self.cfg['mode_mode_offset']['cnot_phase2']


    def define_pulses(self,pt):
        #Calibration of DC offset due to the 2pi ge sideband

        if self.offset_exp == 0:

        # Offset Phase Calibration Experiment
            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge', phase = self.multimode_cfg[self.id]['pi_pi_offset_phase']/2.0 - self.offset_phase)

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
                #
                if self.include_cz_correction:
                    cnot_prep_phase = pt
                    # sideband_phase = self.mode_mode_cnot_dc_phase[self.id][self.id2]
                    sideband_phase = 0
                else:
                    cnot_prep_phase = self.mode_mode_cnot_prep_phase[self.id][self.id2]
                    cnot_phase = pt
                    # sideband_phase = 2*self.mode_mode_cnot_dc_phase[self.id][self.id2]
                    sideband_phase = 0
            else:
                cnot_phase = 0
                cnot_prep_phase=0
                sideband_phase = pt

            cnot_v2(self.psb,self.id2,self.id,cnot_phase=0,efsbphase_0=cnot_prep_phase,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
            if not self.include_cz_correction:
                cnot_v2(self.psb,self.id2,self.id,cnot_phase=0,efsbphase_0=cnot_phase,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=-self.multimode_cfg[self.id]['pi_pi_offset_phase']/(2.0))
            self.psb.append('q','half_pi', self.pulse_type,phase=sideband_phase)

        elif self.offset_exp == 1:
            cnot_phase = self.mode_mode_cnot_phase[self.id][self.id2]

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge', phase= self.multimode_cfg[self.id2]['pi_pi_offset_phase']/(2.0) - self.offset_phase)

            for i in arange(2):
                cnot_v2(self.psb,self.id2,self.id,cnot_phase=0,efsbphase_0=cnot_phase,efsbphase_1=pt,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/(2.0))
            self.psb.append('q','half_pi', self.pulse_type)

        elif self.offset_exp == 2:
            cnot_prep_phase = self.mode_mode_cnot_prep_phase[self.id][self.id2]
            cnot_phase = self.mode_mode_cnot_phase[self.id][self.id2]
            cnot_phase2 = self.mode_mode_cnot_phase2[self.id][self.id2]
            sideband_phase =0

            if 'number' not in self.extra_args:
                self.number=1

            prefactor=self.number

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge', phase = self.multimode_cfg[self.id]['pi_pi_offset_phase']/2.0 - self.offset_phase)

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            cnot_v2(self.psb,self.id2,self.id,cnot_phase=0,efsbphase_0=cnot_prep_phase,efsbphase_1=cnot_phase2,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
            for i in arange(self.number):
                cnot_v2(self.psb,self.id2,self.id,cnot_phase=0,efsbphase_0=cnot_phase,efsbphase_1=cnot_phase2,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=-self.multimode_cfg[self.id]['pi_pi_offset_phase']/(2.0))
            self.psb.append('q','half_pi', self.pulse_type,phase=prefactor*sideband_phase + pt)


        elif self.offset_exp == 3:
            cnot_phase = self.mode_mode_cnot_phase[self.id][self.id2]
            cnot_phase2 = self.mode_mode_cnot_phase2[self.id][self.id2]
            sideband_phase = self.mode_mode_cnot_dc_phase[self.id][self.id2] +pt

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge')

            for i in arange(self.number):
                cnot_v2(self.psb,self.id2,self.id,cnot_phase=0,efsbphase_0=cnot_phase,efsbphase_1=cnot_phase2)
                cnot_v2(self.psb,self.id2,self.id,cnot_phase=0,efsbphase_0=cnot_phase,efsbphase_1=cnot_phase2)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])
            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase + pt)

        elif self.offset_exp == 4:

            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id),'pi_ge')

            cnot_v1(self.psb,self.id2,self.id,cnot_phase=0)
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=self.multimode_cfg[self.id]['pi_pi_offset_phase']+pt)

class Multimode_Mode_Mode_CZ_V2_Offset_Sequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.cfg = cfg
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.flux_pulse_cfg=cfg['flux_pulse_info']
        self.multimode_cfg = cfg['multimodes']

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value


        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        if 'offset_exp' in self.extra_args:
            if self.extra_args['offset_exp']==3 :
                self.expt_pts = arange(0,100,1)
            else:
                self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])
        else:
            if self.expt_cfg['offset_exp']==3 :
                self.expt_pts = arange(0,100,1)
            else:
                self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])


    def define_parameters(self):

        if 'mode' in self.extra_args:
            self.id = self.extra_args['mode']
        else:
            self.id = self.expt_cfg['id']

        if 'mode2' in self.extra_args:
            self.id2 = self.extra_args['mode2']
        else:
            self.id2 = self.expt_cfg['id2']

        if 'offset_exp' in self.extra_args:
            self.offset_exp = self.extra_args['offset_exp']
        else:
            self.offset_exp = self.expt_cfg['offset_exp']

        if 'load_photon' in self.extra_args:
            self.load_photon = self.extra_args['load_photon']
        else:
            self.load_photon = self.expt_cfg['load_photon']

        print "Target id: " +str(self.id)
        print "Target id2: " +str(self.id2)

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)

        self.offset_phase = self.pulse_cfg['gauss']['offset_phase']

        self.mode_mode_cz_dc_phase = self.cfg['mode_mode_offset']['cz_dc_phase']
        self.mode_mode_cz_phase = self.cfg['mode_mode_offset']['cz_phase']



    def define_pulses(self,pt):
        #Calibration of DC offset due to the 2pi ge sideband
        if self.offset_exp == 0:
            # self.psb.append('q','half_pi', self.pulse_type)
            #
            # self.psb.append('q,mm'+str(self.id2),'pi_ge')
            # self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])
            #
            # cphase_v2(self.psb,'q',self.id,cz_dc_phase= pt, cz_phase = 0)
            #
            # self.psb.append('q,mm'+str(self.id2),'pi_ge')
            # self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])
            #
            #
            # self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase)

            self.psb.append('q','half_pi', self.pulse_type)

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            cphase_v1(self.psb,'q',self.id,cz_phase = 0)

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])


            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase + pt)



                #Calibration of phase for second conditional sideband
        elif self.offset_exp == 1:
            # self.psb.append('q','pi', self.pulse_type)
            # self.psb.append('q,mm'+str(self.id),'pi_ge')
            #
            # self.psb.append('q','half_pi', self.pulse_type)
            #
            # self.psb.append('q,mm'+str(self.id2),'pi_ge')
            # self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])
            #
            # cphase_v2(self.psb,'q',self.id
            #           ,cz_dc_phase=self.mode_mode_cz_dc_phase[self.id][self.id2]
            #           ,cz_phase = pt)
            #
            # self.psb.append('q,mm'+str(self.id2),'pi_ge')
            # self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])
            #
            # self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase)
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','half_pi', self.pulse_type)

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            cphase_v1(self.psb,'q',self.id,cz_phase = pt)

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])
            #print self.mode_mode_cz_dc_phase[self.id][self.id2]
            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase + self.mode_mode_cz_dc_phase[self.id][self.id2])

        # Verify that the phases works
        elif self.offset_exp == 2:
            # if self.load_photon:
            #     self.psb.append('q','pi', self.pulse_type)
            #     self.psb.append('q,mm'+str(self.id),'pi_ge')
            #
            # self.psb.append('q','half_pi', self.pulse_type)
            #
            # self.psb.append('q,mm'+str(self.id2),'pi_ge')
            # self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])
            #
            # cphase_v2(self.psb,'q',self.id
            #           ,cz_dc_phase=self.mode_mode_cz_dc_phase[self.id][self.id2]
            #           ,cz_phase = self.mode_mode_cz_phase[self.id][self.id2])
            #
            # self.psb.append('q,mm'+str(self.id2),'pi_ge')
            # self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])
            #
            # self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase + pt)

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','half_pi', self.pulse_type)

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            cphase_v1(self.psb,'q',self.id,cz_phase = self.mode_mode_cz_phase[self.id][self.id2])

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase + self.mode_mode_cz_dc_phase[self.id][self.id2] + pt)

        # replace half pi by varying length
        elif self.offset_exp == 3:
            # if self.load_photon:
            #     self.psb.append('q','pi', self.pulse_type)
            #     self.psb.append('q,mm'+str(self.id),'pi_ge')
            #
            # self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a']
            #                 , length=pt,freq=self.pulse_cfg['gauss']['iq_freq'])
            #
            # self.psb.append('q,mm'+str(self.id2),'pi_ge')
            # self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])
            #
            # cphase_v2(self.psb,'q',self.id
            #           ,cz_dc_phase=self.mode_mode_cz_dc_phase[self.id][self.id2]
            #           ,cz_phase = self.mode_mode_cz_phase[self.id][self.id2])
            #
            # self.psb.append('q,mm'+str(self.id2),'pi_ge')
            # self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])
            #
            # self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a'],
            #                 length=pt,freq=self.pulse_cfg['gauss']['iq_freq'], phase= self.offset_phase)


            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a']
                            , length=pt,freq=self.pulse_cfg['gauss']['iq_freq'])

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            cphase_v1(self.psb,'q',self.id,cz_phase = self.mode_mode_cz_phase[self.id][self.id2])

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a'],
                            length=pt,freq=self.pulse_cfg['gauss']['iq_freq'], phase= self.offset_phase + self.mode_mode_cz_dc_phase[self.id][self.id2])

class Multimode_AC_Stark_Shift_Offset_Sequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.cfg = cfg
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.flux_pulse_cfg=cfg['flux_pulse_info']
        self.multimode_cfg = cfg['multimodes']

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value


        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        if 'offset_exp' in self.extra_args:
            if self.extra_args['offset_exp']==3 :
                self.expt_pts = arange(0,100,1)
            else:
                self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])
        else:
            if self.expt_cfg['offset_exp']==3 :
                self.expt_pts = arange(0,100,1)
            else:
                self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])


    def define_parameters(self):

        if 'mode' in self.extra_args:
            self.id = self.extra_args['mode']
        else:
            self.id = self.expt_cfg['id']

        if 'mode2' in self.extra_args:
            self.id2 = self.extra_args['mode2']
        else:
            self.id2 = self.expt_cfg['id2']

        if 'offset_exp' in self.extra_args:
            self.offset_exp = self.extra_args['offset_exp']
        else:
            self.offset_exp = self.expt_cfg['offset_exp']

        if 'load_photon' in self.extra_args:
            self.load_photon = self.extra_args['load_photon']
        else:
            self.load_photon = self.expt_cfg['load_photon']

        print "Target id: " +str(self.id)
        print "Target id2: " +str(self.id2)

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)

        self.offset_phase = self.pulse_cfg['gauss']['offset_phase']




    def define_pulses(self,pt):
        #Calibration of DC offset due to the 2pi ge sideband
        if self.offset_exp == 0:
            self.psb.append('q','half_pi', self.pulse_type)

            self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.idle(self.multimode_cfg[self.id2]['flux_pi_length']+self.flux_pulse_cfg['spacing'])

            self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase+pt)

                #Calibration of phase for second conditional sideband
        elif self.offset_exp == 1:
            self.psb.append('q','half_pi', self.pulse_type)

            self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q,mm'+str(self.id2),'pi_ge')

            self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase+pt)

class Multimode_ef_2pi_Sequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):

        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        if 'mode_1' in self.extra_args:
            self.id1 = self.extra_args['mode_1']
        else:
            self.id1 = self.expt_cfg['id1']

        if 'mode_2' in self.extra_args:
            self.id2 = self.extra_args['mode_2']
        else:
            self.id2 = self.expt_cfg['id2']

        print "ge mode id: " +str(self.id1)
        print "ef mode id: " +str(self.id2)

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)

        self.offset_phase = self.pulse_cfg['gauss']['offset_phase']


    def define_pulses(self,pt):

        self.psb.append('q','half_pi', self.pulse_type)
        self.psb.append('q,mm'+str(self.id1),'pi_ge')

        self.psb.append('q,mm'+str(self.id1),'pi_ge', phase = self.multimode_cfg[self.id1]['pi_pi_offset_phase'])
        self.psb.append('q,mm'+str(self.id2),'2pi_ef')
        self.psb.append('q,mm'+str(self.id1),'pi_ge',phase= pt)

        self.psb.append('q,mm'+str(self.id1),'pi_ge', phase = self.multimode_cfg[self.id1]['pi_pi_offset_phase'])
        self.psb.append('q','half_pi', self.pulse_type, phase = self.offset_phase)

class Multimode_ef_Pi_PiSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):

        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        if 'mode_1' in self.extra_args:
            self.id1 = self.extra_args['mode_1']
        else:
            self.id1 = self.expt_cfg['id1']

        if 'mode_2' in self.extra_args:
            self.id2 = self.extra_args['mode_2']
        else:
            self.id2 = self.expt_cfg['id2']

        print "ge mode id: " +str(self.id1)
        print "ef mode id: " +str(self.id2)

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)

        self.offset_phase = self.pulse_cfg['gauss']['offset_phase']


    def define_pulses(self,pt):


        self.psb.append('q','half_pi', self.pulse_type)
        self.psb.append('q,mm'+str(self.id1),'pi_ge')

        self.psb.append('q,mm'+str(self.id1),'pi_ge', phase = self.multimode_cfg[self.id1]['pi_pi_offset_phase'])
        self.psb.append('q,mm'+str(self.id2),'pi_ef')
        self.psb.append('q','pi_q_ef')
        self.psb.append('q,mm'+str(self.id2),'pi_ef')
        self.psb.append('q,mm'+str(self.id1),'pi_ge')

        self.psb.append('q,mm'+str(self.id1),'pi_ge', phase = pt + self.multimode_cfg[self.id1]['pi_pi_offset_phase'])
        self.psb.append('q,mm'+str(self.id2),'pi_ef')
        self.psb.append('q','pi_q_ef')
        self.psb.append('q','half_pi', self.pulse_type, phase = self.offset_phase)

        # Tomography at a given time slice

class CPhaseOptimizationSweepSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)
        self.idle_time = self.extra_args['idle_time']

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)
        self.id1 = self.expt_cfg['id1']
        self.id2 = self.expt_cfg['id2']
        self.id = self.expt_cfg['id']
        self.half_pi_offset = self.pulse_cfg[self.pulse_type]['offset_phase']


    def define_pulses(self,pt):
        if self.expt_cfg['pi_pi']:
            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.idle(self.idle_time)
            self.psb.append('q,mm'+str(self.id),'pi_ge', phase = pt)
            self.psb.append('q','half_pi', self.pulse_type, phase=self.half_pi_offset)
        elif self.expt_cfg['ef']:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q','half_pi_q_ef')
            self.psb.idle(self.idle_time)
            self.psb.append('q','half_pi_q_ef', phase = pt)
            self.psb.append('q','pi', self.pulse_type)
        elif self.expt_cfg['pi_pi_ef']:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q','half_pi_q_ef')
            self.psb.append('q,mm'+str(self.id),'pi_ef')
            self.psb.idle(self.idle_time)
            self.psb.append('q,mm'+str(self.id),'pi_ef', phase = pt)
            self.psb.append('q','half_pi_q_ef')
            self.psb.append('q','pi', self.pulse_type)

class MultimodeSingleResonatorTomographySequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        ## we define
        self.states_num = 4
        self.operations_num = 1
        ## automauted
        self.tomography_pulse_num = 3

        sequence_num = self.states_num*self.operations_num*self.tomography_pulse_num
        self.expt_pts = np.arange(0,sequence_num)

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.id = self.expt_cfg['id']
        self.half_pi_offset = self.pulse_cfg[self.pulse_type]['y_phase']-90
        self.op_index =  self.expt_cfg['op_index']

    def define_pulses(self,pt):

        ### Initiate states
        self.define_states(pt)


    def define_states(self,pt):
        state_index = (pt/(self.tomography_pulse_num*self.operations_num)) % self.states_num
        if state_index == 0:
            # I
            pass
        elif state_index == 1:
            # Pi/2 X
            self.psb.append('q','half_pi', self.pulse_type)
        elif state_index == 2:
            # Pi/2 Y
            self.psb.append('q','half_pi_y', self.pulse_type)
        elif state_index == 3:
            # Pi X
            self.psb.append('q','pi', self.pulse_type)

        self.define_operations_pulse(pt)

    def define_operations_pulse(self,pt):
        # op_index = (pt/self.tomography_pulse_num) % self.operations_num
        op_index = self.op_index
        if op_index ==0:
            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=35)
            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=35)

        if op_index ==1:
            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=35)
            self.psb.append('q','half_pi', self.pulse_type, phase= 0)
            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=35)

        if op_index ==2:
            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=35)
            self.psb.append('q','half_pi_y', self.pulse_type, phase = self.half_pi_offset+self.pulse_cfg[self.pulse_type]['y_phase'])
            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=35)



        self.define_tomography_pulse(pt)

    def define_tomography_pulse(self,pt):
        ### gates before measurement for tomography
        tomo_index = pt%self.tomography_pulse_num
        if tomo_index == 0:
            # <X>
            self.psb.append('q','half_pi', self.pulse_type, phase=self.half_pi_offset)
        elif tomo_index == 1:
            # <Y>
            self.psb.append('q','half_pi_y', self.pulse_type, phase = self.half_pi_offset+self.pulse_cfg[self.pulse_type]['y_phase']  )
        elif tomo_index == 2:
            # <Z>
            pass

class MultimodeTwoResonatorTomographySequence(QubitPulseSequence):

    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        ## we define
        self.states_num = 1
        ## automauted
        self.tomography_pulse_num = 15

        sequence_num = self.states_num*self.tomography_pulse_num
        self.expt_pts = np.arange(0,sequence_num)

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)
        self.id1 = self.expt_cfg['id1']
        self.id2 = self.expt_cfg['id2']
        self.id = self.expt_cfg['id']
        self.half_pi_offset = self.pulse_cfg[self.pulse_type]['offset_phase']
        self.state_index =  self.expt_cfg['state_index']

    def define_pulses(self,pt):

        ### Initiate states
        self.define_states(pt)

    def define_states(self,pt):
        state_index = self.state_index
        self.halfpicounter1=0
        self.halfpicounter2=0
        if state_index ==0:
            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=35)
            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=35)

        if state_index ==1:

            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=35)
            self.psb.append('q','half_pi', self.pulse_type)
            self.halfpicounter1+=1
            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=35)

        if state_index ==2:

            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=35)
            self.psb.append('q','half_pi_y', self.pulse_type)
            self.halfpicounter1+=1
            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=35)

        if state_index ==3:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= self.expt_cfg['time_slice'])
            self.psb.append('q,mm'+str(self.id2),'pi_ge')

        if state_index ==4:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= self.expt_cfg['time_slice'],phase=180)
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=180)

        if state_index ==5:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= self.expt_cfg['time_slice'],phase=180)
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=0)

        if state_index ==6:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= self.expt_cfg['time_slice'])
            self.psb.append('q,mm'+str(self.id2),'pi_ge', phase=180)

        self.define_tomography_pulse(pt)

    def define_tomography_pulse(self,pt):
        ### gates before measurement for two resonaotor tomography
        tomo_index = pt%self.tomography_pulse_num
        if tomo_index == 0:
            # -<IX>
            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q','half_pi_y', self.pulse_type, phase = 90 + self.halfpicounter2*self.half_pi_offset)
            # self.halfpicounter2+=1
        elif tomo_index == 1:
            # <IY>
            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q','half_pi', self.pulse_type, phase =  self.halfpicounter2*self.half_pi_offset)
            # self.halfpicounter2+=1
        elif tomo_index == 2:
            # <IZ>
            self.psb.append('q,mm'+str(self.id2),'pi_ge')
        elif tomo_index == 3:
            # -<XI>
            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q','half_pi_y', self.pulse_type, phase = 90 + self.halfpicounter1*self.half_pi_offset)
            # self.halfpicounter2+=1
        elif tomo_index == 4:
            # <XX>

            #CNOT
            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q','pi_q_ef', phase=self.expt_cfg['pi_ef_offset'])
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=180)

            self.psb.append('q,mm'+str(self.id1),'pi_ge', phase=self.expt_cfg['final_offset_not'])
            self.psb.append('q','half_pi_y', self.pulse_type, phase= self.halfpicounter1*self.half_pi_offset  + 90)

        elif tomo_index == 5:
            # -<XY>

            #CY
            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q','pi_q_ef', phase=self.expt_cfg['pi_ef_offset']+90)
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=180)


            self.psb.append('q,mm'+str(self.id1),'pi_ge', phase=self.expt_cfg['final_offset_not'])
            self.psb.append('q','half_pi_y', self.pulse_type, phase= self.halfpicounter1*self.half_pi_offset  + 90)


        elif tomo_index == 6:
            # <XZ>
            cphase(self.psb,self.id1,self.id2)
            self.psb.append('q,mm'+str(self.id1),'pi_ge', phase=self.expt_cfg['final_offset_z'])
            self.psb.append('q','half_pi_y', self.pulse_type, phase= self.halfpicounter1*self.half_pi_offset  + 90)

        elif tomo_index == 7:
            # <YI>
            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q','half_pi', self.pulse_type)
        elif tomo_index == 8:
            # -<YX>

            #CNOT
            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q','pi_q_ef', phase=self.expt_cfg['pi_ef_offset'])
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=180)


            self.psb.append('q,mm'+str(self.id1),'pi_ge', phase=self.expt_cfg['final_offset_not'])
            self.psb.append('q','half_pi', self.pulse_type, phase= self.halfpicounter1*self.half_pi_offset)


        elif tomo_index == 9:
            # <YY>

            #CY
            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q','pi_q_ef', phase=self.expt_cfg['pi_ef_offset']+90)
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=180)

            self.psb.append('q,mm'+str(self.id1),'pi_ge', phase=self.expt_cfg['final_offset_not'])
            self.psb.append('q','half_pi', self.pulse_type, phase= self.halfpicounter1*self.half_pi_offset)


        elif tomo_index == 10:
            # -<YZ>
            cphase(self.psb,self.id1,self.id2)
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=self.expt_cfg['final_offset_z'])
            self.psb.append('q','half_pi', self.pulse_type, phase= self.halfpicounter1*self.half_pi_offset)
        elif tomo_index == 11:
            # <ZI>
            self.psb.append('q,mm'+str(self.id1),'pi_ge')
        elif tomo_index == 12:
            # <ZX>
            cphase(self.psb,self.id1,self.id2)
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.expt_cfg['final_offset2_z'])
            self.psb.append('q','half_pi_y', self.pulse_type, phase= self.halfpicounter2*self.half_pi_offset + 90 )
        elif tomo_index == 13:
            # <ZY>
            cphase(self.psb,self.id1,self.id2)
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.expt_cfg['final_offset2_z'])
            self.psb.append('q','half_pi', self.pulse_type,phase= self.halfpicounter2*self.half_pi_offset)
        elif tomo_index == 14:
            # <ZZ>

            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q','pi_q_ef', phase=self.expt_cfg['pi_ef_offset'])
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=180)

            self.psb.append('q,mm'+str(self.id2),'pi_ge', phase=self.expt_cfg['final_offset_not'])

class MultimodeTwoResonatorTomographyPhaseSweepSequence(QubitPulseSequence):

    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)
        self.tomography_num = self.extra_args['tomography_num']
        self.state_num = self.extra_args['state_num']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        ## we define
        self.states_num = 1
        ## automauted
        self.tomography_pulse_num = 15

        sequence_num = self.states_num*self.tomography_pulse_num
        self.expt_pts = np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)
        self.id1 = self.expt_cfg['id1']
        self.id2 = self.expt_cfg['id2']
        self.id = self.expt_cfg['id']
        self.half_pi_offset = self.pulse_cfg[self.pulse_type]['offset_phase']
        self.state_index =  self.expt_cfg['state_index']
        self.halfpicounter1 = 0
        self.halfpicounter2 = 0

    def define_pulses(self,pt):

        ### Initiate states
        self.define_states(pt)

    def define_states(self,pt):
        state_index = self.state_index
        self.halfpicounter1=0
        self.halfpicounter2=0
        if self.state_num ==0:
            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=35)
            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=35)

        if self.state_num ==1:

            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=35)
            self.psb.append('q','half_pi', self.pulse_type)
            self.halfpicounter1+=1
            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=35)

        if self.state_num ==2:

            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=35)
            self.psb.append('q','half_pi_y', self.pulse_type)
            self.halfpicounter1+=1
            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=35)

        if self.state_num ==3:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= self.expt_cfg['time_slice'])
            self.psb.append('q,mm'+str(self.id2),'pi_ge')

        if self.state_num ==4:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= self.expt_cfg['time_slice'],phase=180)
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=180)

        if self.state_num ==5:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= self.expt_cfg['time_slice'],phase=180)
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=0)

        if self.state_num ==6:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= self.expt_cfg['time_slice'])
            self.psb.append('q,mm'+str(self.id2),'pi_ge', phase=180)

        self.define_tomography_pulse(pt)

    def define_tomography_pulse(self,pt):
        ### gates before measurement for two resonaotor tomography
        tomo_index = (self.tomography_num)%self.tomography_pulse_num
        if tomo_index == 0:
            # -<IX>
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=pt)
            self.psb.append('q','half_pi_y', self.pulse_type, phase = 90 + self.halfpicounter2*self.half_pi_offset)
            # self.halfpicounter2+=1
        elif tomo_index == 1:
            # <IY>
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=pt)
            self.psb.append('q','half_pi', self.pulse_type, phase =  self.halfpicounter2*self.half_pi_offset)
            # self.halfpicounter2+=1
        elif tomo_index == 2:
            # <IZ>
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=pt)
        elif tomo_index == 3:
            # -<XI>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=pt)
            self.psb.append('q','half_pi_y', self.pulse_type, phase = 90 + self.halfpicounter1*self.half_pi_offset)
            # self.halfpicounter2+=1
        elif tomo_index == 4:
            # <XX>

            #CNOT
            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q','pi_q_ef', phase=self.expt_cfg['pi_ef_offset'])
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=180)

            self.psb.append('q,mm'+str(self.id1),'pi_ge', phase=self.expt_cfg['final_offset_not']+pt)
            self.psb.append('q','half_pi_y', self.pulse_type, phase= self.halfpicounter1*self.half_pi_offset  + 90)

        elif tomo_index == 5:
            # -<XY>

            #CY
            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q','pi_q_ef', phase=self.expt_cfg['pi_ef_offset']+90)
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=180)


            self.psb.append('q,mm'+str(self.id1),'pi_ge', phase=self.expt_cfg['final_offset_not']+pt)
            self.psb.append('q','half_pi_y', self.pulse_type, phase= self.halfpicounter1*self.half_pi_offset  + 90)


        elif tomo_index == 6:
            # <XZ>
            cphase(self.psb,self.id1,self.id2)
            self.psb.append('q,mm'+str(self.id1),'pi_ge', phase=self.expt_cfg['final_offset_z']+pt)
            self.psb.append('q','half_pi_y', self.pulse_type, phase= self.halfpicounter1*self.half_pi_offset  + 90)

        elif tomo_index == 7:
            # <YI>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=pt)
            self.psb.append('q','half_pi', self.pulse_type)
        elif tomo_index == 8:
            # -<YX>

            #CNOT
            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q','pi_q_ef', phase=self.expt_cfg['pi_ef_offset'])
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=180)


            self.psb.append('q,mm'+str(self.id1),'pi_ge', phase=self.expt_cfg['final_offset_not'] + pt)
            self.psb.append('q','half_pi', self.pulse_type, phase= self.halfpicounter1*self.half_pi_offset)


        elif tomo_index == 9:
            # <YY>

            #CY
            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q','pi_q_ef', phase=self.expt_cfg['pi_ef_offset']+90)
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=180)

            self.psb.append('q,mm'+str(self.id1),'pi_ge', phase=self.expt_cfg['final_offset_not']+pt)
            self.psb.append('q','half_pi', self.pulse_type, phase= self.halfpicounter1*self.half_pi_offset)


        elif tomo_index == 10:
            # -<YZ>
            cphase(self.psb,self.id1,self.id2)
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=self.expt_cfg['final_offset_z']+pt)
            self.psb.append('q','half_pi', self.pulse_type, phase= self.halfpicounter1*self.half_pi_offset)
        elif tomo_index == 11:
            # <ZI>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=pt)
        elif tomo_index == 12:
            # <ZX>
            cphase(self.psb,self.id1,self.id2)
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.expt_cfg['final_offset2_z'] + pt)
            self.psb.append('q','half_pi_y', self.pulse_type, phase= self.halfpicounter2*self.half_pi_offset + 90 )
        elif tomo_index == 13:
            # <ZY>
            cphase(self.psb,self.id1,self.id2)
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.expt_cfg['final_offset2_z']+pt)
            self.psb.append('q','half_pi', self.pulse_type,phase= self.halfpicounter2*self.half_pi_offset)
        elif tomo_index == 14:
            # <ZZ>

            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q','pi_q_ef', phase=self.expt_cfg['pi_ef_offset'])
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=180)

            self.psb.append('q,mm'+str(self.id2),'pi_ge', phase=self.expt_cfg['final_offset_not']+pt)

class MultimodeTwoResonatorTomographyPhaseSweepSequenceNEW(QubitPulseSequence):

    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.cfg = cfg
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)
        self.tomography_num = self.extra_args['tomography_num']
        self.state_num = self.extra_args['state_num']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        ## we define
        self.states_num = 1
        ## automauted
        self.tomography_pulse_num = 15

        sequence_num = self.states_num*self.tomography_pulse_num

        self.expt_pts = np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']

        if 'id1' in self.extra_args:
            self.id1 = self.extra_args['id1']
        else:
            self.id1 = self.expt_cfg['id1']

        if 'id2' in self.extra_args:
            self.id2 = self.extra_args['id2']
        else:
            self.id2 = self.expt_cfg['id2']

        if 'do_length_sweep' in self.extra_args:
            self.do_length_sweep = self.extra_args['do_length_sweep']
            self.length = self.extra_args['length']
        else:
            self.do_length_sweep = False
            self.length =(self.multimode_cfg[self.id1]['flux_pi_length']+self.multimode_cfg[self.id1]['flux_2pi_length'])/(2.0)


        self.id = self.expt_cfg['id']

        self.offset_phase = self.pulse_cfg[self.pulse_type]['offset_phase']
        self.state_index =  self.expt_cfg['state_index']
        self.halfpicounter1 = 0
        self.halfpicounter2 = 0

        self.mode_mode_cnot_dc_phase = self.cfg['mode_mode_offset']['cnot_dc_phase']
        self.mode_mode_cnot_phase = self.cfg['mode_mode_offset']['cnot_phase']
        self.mode_mode_cnot_phase2 = self.cfg['mode_mode_offset']['cnot_phase2']

        self.mode_mode_cz_dc_phase = self.cfg['mode_mode_offset']['cz_dc_phase']
        self.mode_mode_cz_phase = self.cfg['mode_mode_offset']['cz_phase']
        self.mode_mode_cz_phase2 = self.cfg['mode_mode_offset']['cz_phase2']

        self.cz_phase_cz = self.mode_mode_cz_phase[self.id1][self.id2]
        self.cz_phase2_cz = self.mode_mode_cz_phase2[self.id1][self.id2]

        self.cz_phase_zc = self.mode_mode_cz_phase[self.id2][self.id1]
        self.cz_phase2_zc = self.mode_mode_cz_phase2[self.id2][self.id1]

        self.cnot_phase_cx = self.mode_mode_cnot_phase[self.id1][self.id2]
        self.cnot_phase2_cx = self.mode_mode_cnot_phase2[self.id1][self.id2]

        self.cnot_phase_xc = self.mode_mode_cnot_phase[self.id2][self.id1]
        self.cnot_phase2_xc = self.mode_mode_cnot_phase2[self.id2][self.id1]


    def define_pulses(self,pt):

        ### Initiate states
        self.define_states(pt)

        ### Tomography
        self.define_tomography_pulse(pt)

    def define_states(self,pt):
        state_index = self.state_index


        if self.state_num ==0:

            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'],
                            phase = self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0, length= self.length)
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

        if self.state_num == 1:

            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'],
                        phase = self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0, length= (self.multimode_cfg[self.id1]['flux_pi_length']+self.multimode_cfg[self.id1]['flux_2pi_length'])/(2.0))
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)


    def define_tomography_pulse(self,pt):
        ### Correlators for two mode tomography; while sweeping the phase of the final sideband pulse

        # State convention : |id2, id1 >

        # Gate convention CNOT/CZ(control_id,target_id)

        #CNOT(id2, id1) = CX
        #CZ(id2, id1) = CZ
        tomo_index = (self.tomography_num)%self.tomography_pulse_num

        if tomo_index == 0:

            # -<IX>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + pt)
            self.psb.append('q','half_pi_y', self.pulse_type)

            # self.halfpicounter2+=1

        elif tomo_index == 1:
            # <IY>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + pt)
            self.psb.append('q','half_pi', self.pulse_type)

            # self.halfpicounter2+=1

        elif tomo_index == 2:
            # <IZ>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + pt)

        elif tomo_index == 3:
            # -<XI>
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + pt)
            self.psb.append('q','half_pi_y', self.pulse_type)
            # self.halfpicounter2+=1
        elif tomo_index == 4:
            # <XX> = XI + CX

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=0,efsbphase_0=self.cnot_phase_cx,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + pt)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 5:
            # -<XY> = XI + CY

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase= 90.0,efsbphase_0=self.cnot_phase_cx,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + pt)
            self.psb.append('q','half_pi_y', self.pulse_type)


        elif tomo_index == 6:
            # <XZ>

            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.cz_phase_cz,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + pt)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 7:
            # <YI>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + pt)
            self.psb.append('q','half_pi', self.pulse_type)

        elif tomo_index == 8:
            # -<YX>

            #CNOT
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=0,efsbphase_0=self.cnot_phase_cx,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + pt)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 9:
            # <YY>

            #CY
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=90.0,efsbphase_0=self.cnot_phase_cx,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + pt)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 10:
            # -<YZ>

            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.cz_phase_cz,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + pt)
            self.psb.append('q','half_pi_y', self.pulse_type)


        elif tomo_index == 11:
            # <ZI>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + pt)

        elif tomo_index == 12:
            # <ZX>

            cphase_v3(self.psb,self.id1,self.id2,efsbphase_0=self.cz_phase_zc,efsbphase_1=self.cz_phase2_zc,gesbphase1=self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + pt)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 13:
            # <ZY>

            cphase_v3(self.psb,self.id1,self.id2,efsbphase_0=self.cz_phase_zc,efsbphase_1=self.cz_phase2_zc,gesbphase1=self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + pt)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 14:
            # <ZZ>

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=0,efsbphase_0=self.cnot_phase_cx,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + pt)

class MultimodeGHZEntanglementWitnessSequence(QubitPulseSequence):

    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.cfg = cfg
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)
        self.tomography_num = self.extra_args['tomography_num']
        self.state_num = self.extra_args['state_num']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        ## we define
        self.states_num = 1
        ## automauted
        self.tomography_pulse_num = 15

        sequence_num = self.states_num*self.tomography_pulse_num

        self.expt_pts = np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']

        if 'id1' in self.extra_args:
            self.id1 = self.extra_args['id1']
        else:
            self.id1 = self.expt_cfg['id1']

        if 'id2' in self.extra_args:
            self.id2 = self.extra_args['id2']
        else:
            self.id2 = self.expt_cfg['id2']


        if 'id3' in self.extra_args:
            self.id3 = self.extra_args['id3']
        else:
            self.id3 = self.expt_cfg['id3']

        if 'pi_q_ef_offset' in self.extra_args:
            self.pi_q_ef_offset= self.extra_args['pi_q_ef_offset']
        else:
            self.pi_q_ef_offset = 0.0


        if 'final_sb_offset' in self.extra_args:
            self.final_sb_offset= self.extra_args['final_sb_offset']
        else:
            self.final_sb_offset = 0.0

        if 'sweep_ef_qubit_phase' in self.extra_args:
            self.sweep_ef_qubit_phase= self.extra_args['sweep_ef_qubit_phase']
        else:
            self.sweep_ef_qubit_phase=False

        self.length =(self.multimode_cfg[self.id1]['flux_pi_length']+self.multimode_cfg[self.id1]['flux_2pi_length'])/(2.0)



        self.mode_mode_cnot_dc_phase = self.cfg['mode_mode_offset']['cnot_dc_phase']
        self.mode_mode_cnot_phase = self.cfg['mode_mode_offset']['cnot_phase']
        self.mode_mode_cnot_phase2 = self.cfg['mode_mode_offset']['cnot_phase2']

        self.mode_mode_cz_dc_phase = self.cfg['mode_mode_offset']['cz_dc_phase']
        self.mode_mode_cz_phase = self.cfg['mode_mode_offset']['cz_phase']
        self.mode_mode_cz_phase2 = self.cfg['mode_mode_offset']['cz_phase2']

        self.cz_phase_cz = self.mode_mode_cz_phase[self.id1][self.id2]
        self.cz_phase2_cz = self.mode_mode_cz_phase2[self.id1][self.id2]

        self.cz_phase_zc = self.mode_mode_cz_phase[self.id2][self.id1]
        self.cz_phase2_zc = self.mode_mode_cz_phase2[self.id2][self.id1]

        self.cnot_phase_cx = self.mode_mode_cnot_phase[self.id1][self.id2]
        self.cnot_phase2_cx = self.mode_mode_cnot_phase2[self.id1][self.id2]

        self.cnot_phase_xc = self.mode_mode_cnot_phase[self.id2][self.id1]
        self.cnot_phase2_xc = self.mode_mode_cnot_phase2[self.id2][self.id1]

    def define_pulses(self,pt):

        ### Initiate states
        self.define_states(pt)

        ### Tomography
        self.define_tomography_pulse(pt)

    def define_states(self,pt):

        if self.state_num ==0:

            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= self.length,phase=-90)

            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q','pi_q_ef')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q,mm'+str(self.id3),'pi_ge',phase=-90)


    def define_tomography_pulse(self,pt):
        ### Correlators for two mode tomography; while sweeping the phase of the final sideband pulse

        # State convention : |id2, id1 >

        # Gate convention CNOT/CZ(control_id,target_id)

        #CNOT(id2, id1) = CX
        #CZ(id2, id1) = CZ
        tomo_index = (self.tomography_num)%self.tomography_pulse_num

        if self.sweep_ef_qubit_phase:
            self.final_sb_phase = self.final_sb_offset
            self.ef_qubit_phase_2 = self.pi_q_ef_offset + pt
        else:
            self.final_sb_phase = self.final_sb_offset + pt
            self.ef_qubit_phase_2 = self.pi_q_ef_offset



        if tomo_index == 0:

            #<XXX> = + XII + CIX + CXI

            cnot_v2(self.psb,self.id3,self.id2,cnot_phase=0,efsbphase_0=self.mode_mode_cnot_phase[self.id2][self.id3],efsbphase_1=self.mode_mode_cnot_phase2[self.id2][self.id3],gesbphase1=self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0)
            cnot_v2(self.psb,self.id3,self.id1,cnot_phase=0 + self.ef_qubit_phase_2,efsbphase_0=self.mode_mode_cnot_phase[self.id1][self.id3],efsbphase_1=self.mode_mode_cnot_phase2[self.id1][self.id3],gesbphase1=self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id3),'pi_ge',phase=-self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)



        elif tomo_index == 1:
            #<YYX> = + YII + CIX + CYI


            cnot_v2(self.psb,self.id3,self.id2,cnot_phase=-90.0,efsbphase_0=self.mode_mode_cnot_phase[self.id2][self.id3],efsbphase_1=self.mode_mode_cnot_phase2[self.id2][self.id3],gesbphase1=self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0)
            cnot_v2(self.psb,self.id3,self.id1,cnot_phase=0+self.ef_qubit_phase_2,efsbphase_0=self.mode_mode_cnot_phase[self.id1][self.id3],efsbphase_1=self.mode_mode_cnot_phase2[self.id1][self.id3],gesbphase1=self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id3),'pi_ge',phase=-self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)

        elif tomo_index == 2:
            #<YXY> = + YII + CIY + CXI

            cnot_v2(self.psb,self.id3,self.id2,cnot_phase=0.0,efsbphase_0=self.mode_mode_cnot_phase[self.id2][self.id3],efsbphase_1=self.mode_mode_cnot_phase2[self.id2][self.id3],gesbphase1=self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0)
            cnot_v2(self.psb,self.id3,self.id1,cnot_phase=-90.0 + self.ef_qubit_phase_2,efsbphase_0=self.mode_mode_cnot_phase[self.id1][self.id3],efsbphase_1=self.mode_mode_cnot_phase2[self.id1][self.id3],gesbphase1=self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id3),'pi_ge',phase=-self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 3:
            #<XYY> = + XII + CIY + CYI

            cnot_v2(self.psb,self.id3,self.id2,cnot_phase=-90.0,efsbphase_0=self.mode_mode_cnot_phase[self.id2][self.id3],efsbphase_1=self.mode_mode_cnot_phase2[self.id2][self.id3],gesbphase1=self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0)
            cnot_v2(self.psb,self.id3,self.id1,cnot_phase=-90.0 + self.ef_qubit_phase_2,efsbphase_0=self.mode_mode_cnot_phase[self.id1][self.id3],efsbphase_1=self.mode_mode_cnot_phase2[self.id1][self.id3],gesbphase1=self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id3),'pi_ge',phase=-self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)


class MultimodeconcatenatedGHZEntanglementWitnessSequence(QubitPulseSequence):

    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.cfg = cfg
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)
        self.tomography_num = self.extra_args['tomography_num']

        self.state_num = self.extra_args['state_num']

        if 'echo_pi_time_sweep' in self.extra_args:
            self.echo_pi_time_sweep = self.extra_args['echo_pi_time_sweep']
        else:
            self.echo_pi_time_sweep = False

        if 'ramsey_freq' in self.extra_args:
            self.ramsey_freq = self.extra_args['ramsey_freq']
        else:
            self.ramsey_freq = 0.0e6


        if 'startstopstep' in self.extra_args:
            self.startstopstep = self.extra_args['startstopstep']
        else:
            self.startstopstep = [0.0,1500.0,15.0]


        print "Echo pi sweep = " + str(self.echo_pi_time_sweep)
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        ## we define
        self.states_num = 1
        ## automauted
        self.tomography_pulse_num = 15

        sequence_num = self.states_num*self.tomography_pulse_num
        if self.echo_pi_time_sweep:
            self.expt_pts = arange(self.startstopstep[0],self.startstopstep[1],self.startstopstep[2])
            print "Goes here"
        else:
            self.expt_pts = np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']

        if 'id1' in self.extra_args:
            self.id1 = self.extra_args['id1']
        else:
            self.id1 = self.expt_cfg['id1']

        if 'id2' in self.extra_args:
            self.id2 = self.extra_args['id2']
        else:
            self.id2 = self.expt_cfg['id2']


        if 'id3' in self.extra_args:
            self.id3 = self.extra_args['id3']
        else:
            self.id3 = self.expt_cfg['id3']

        if 'pi_q_ef_offset' in self.extra_args:
            self.pi_q_ef_offset= self.extra_args['pi_q_ef_offset']
        else:
            self.pi_q_ef_offset = 0.0


        if 'final_sb_offset' in self.extra_args:
            self.final_sb_offset= self.extra_args['final_sb_offset']
        else:
            self.final_sb_offset = 0.0

        if 'sweep_ef_qubit_phase' in self.extra_args:
            self.sweep_ef_qubit_phase= self.extra_args['sweep_ef_qubit_phase']
        else:
            self.sweep_ef_qubit_phase=False


        if 'number_pi_pulses' in self.extra_args:
            self.number_pi_pulses = self.extra_args['number_pi_pulses']
        else:
            self.number_pi_pulses = 1



        self.length =(self.multimode_cfg[self.id1]['flux_pi_length']+self.multimode_cfg[self.id1]['flux_2pi_length'])/(2.0)



        self.mode_mode_cnot_dc_phase = self.cfg['mode_mode_offset']['cnot_dc_phase']
        self.mode_mode_cnot_phase = self.cfg['mode_mode_offset']['cnot_phase']
        self.mode_mode_cnot_phase2 = self.cfg['mode_mode_offset']['cnot_phase2']

        self.mode_mode_cz_dc_phase = self.cfg['mode_mode_offset']['cz_dc_phase']
        self.mode_mode_cz_phase = self.cfg['mode_mode_offset']['cz_phase']
        self.mode_mode_cz_phase2 = self.cfg['mode_mode_offset']['cz_phase2']

        self.cz_phase_cz = self.mode_mode_cz_phase[self.id1][self.id2]
        self.cz_phase2_cz = self.mode_mode_cz_phase2[self.id1][self.id2]

        self.cz_phase_zc = self.mode_mode_cz_phase[self.id2][self.id1]
        self.cz_phase2_zc = self.mode_mode_cz_phase2[self.id2][self.id1]

        self.cnot_phase_cx = self.mode_mode_cnot_phase[self.id1][self.id2]
        self.cnot_phase2_cx = self.mode_mode_cnot_phase2[self.id1][self.id2]

        self.cnot_phase_xc = self.mode_mode_cnot_phase[self.id2][self.id1]
        self.cnot_phase2_xc = self.mode_mode_cnot_phase2[self.id2][self.id1]

    def define_pulses(self,pt):

        ### Initiate states
        self.define_states(pt)

        ### Tomography
        self.define_tomography_pulse(pt)

    def define_states(self,pt):

        if self.state_num ==0:

            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= self.length,phase=-90)

            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q','pi_q_ef')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q,mm'+str(self.id3),'pi_ge',phase=-90)


    def define_tomography_pulse(self,pt):
        ### Correlators for two mode tomography; while sweeping the phase of the final sideband pulse

        # State convention : |id2, id1 >

        # Gate convention CNOT/CZ(control_id,target_id)

        #CNOT(id2, id1) = CX
        #CZ(id2, id1) = CZ
        tomo_index = (self.tomography_num)%self.tomography_pulse_num

        if self.sweep_ef_qubit_phase:
            self.final_sb_phase = self.final_sb_offset
            self.ef_qubit_phase_2 = self.pi_q_ef_offset + pt
        elif self.echo_pi_time_sweep:
            self.final_sb_phase = self.final_sb_offset + 360.0*self.ramsey_freq*pt/1.0e9
            self.ef_qubit_phase_2 = self.pi_q_ef_offset
        else:
            self.final_sb_phase = self.final_sb_offset + pt
            self.ef_qubit_phase_2 = self.pi_q_ef_offset



        if tomo_index == 0:

            #<XXX> = + XII + CIX + CXI
            efsbphase_0=self.mode_mode_cnot_phase[self.id2][self.id3]
            efsbphase_1=self.mode_mode_cnot_phase2[self.id2][self.id3]
            gesbphase1=self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0
            efsbphase_1 = 0
            cnot_phase = 0
            efsbphase_2 = 0

            self.psb.append('q,mm'+str(self.id3),'pi_ge', phase = -gesbphase1)
            self.psb.append('q,mm'+str(self.id2),'pi_ef',phase= efsbphase_0 )
            self.psb.append('q','pi_q_ef', phase=cnot_phase)
            self.psb.append('q,mm'+str(self.id2),'pi_ef',phase=efsbphase_2)

            efsbphase_0=self.mode_mode_cnot_phase[self.id1][self.id3]
            efsbphase_1=self.mode_mode_cnot_phase2[self.id1][self.id3]
            gesbphase1=self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0
            cnot_phase = self.ef_qubit_phase_2

            self.psb.append('q,mm'+str(self.id1),'pi_ef',phase= efsbphase_0 )
            self.psb.append('q','pi_q_ef', phase=cnot_phase)
            self.psb.append('q,mm'+str(self.id1),'pi_ef',phase=efsbphase_2)

            if self.echo_pi_time_sweep:
                for i in arange(self.number_pi_pulses):
                    self.psb.append('q','pi_y',self.pulse_type,addphase=self.final_sb_offset+90.0)
                    self.psb.idle(pt/float(self.number_pi_pulses))


            self.psb.append('q','half_pi_y',self.pulse_type,addphase=self.final_sb_phase)



        elif tomo_index == 1:
            #<YYX> = + YII + CIX + CYI
            efsbphase_0=self.mode_mode_cnot_phase[self.id2][self.id3]
            efsbphase_1=self.mode_mode_cnot_phase2[self.id2][self.id3]
            gesbphase1=self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0
            efsbphase_1 = 0
            cnot_phase = -90.0
            efsbphase_2 = 0

            self.psb.append('q,mm'+str(self.id3),'pi_ge', phase = -gesbphase1)
            self.psb.append('q,mm'+str(self.id2),'pi_ef',phase= efsbphase_0 )
            self.psb.append('q','pi_q_ef', phase=cnot_phase)
            self.psb.append('q,mm'+str(self.id2),'pi_ef',phase=efsbphase_2)

            efsbphase_0=self.mode_mode_cnot_phase[self.id1][self.id3]
            efsbphase_1=self.mode_mode_cnot_phase2[self.id1][self.id3]
            gesbphase1=self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0
            cnot_phase = 0.0 + self.ef_qubit_phase_2

            self.psb.append('q,mm'+str(self.id1),'pi_ef',phase= efsbphase_0 )
            self.psb.append('q','pi_q_ef', phase=cnot_phase)
            self.psb.append('q,mm'+str(self.id1),'pi_ef',phase=efsbphase_2)

            if self.echo_pi_time_sweep:
                for i in arange(self.number_pi_pulses):
                    self.psb.append('q','pi',self.pulse_type,addphase=self.final_sb_offset+90.0)
                    self.psb.idle(pt/float(self.number_pi_pulses))

            self.psb.append('q','half_pi',self.pulse_type,addphase=self.final_sb_phase)


        elif tomo_index == 2:
            #<YXY> = + YII + CIY + CXI
            efsbphase_0=self.mode_mode_cnot_phase[self.id2][self.id3]
            efsbphase_1=self.mode_mode_cnot_phase2[self.id2][self.id3]
            gesbphase1=self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0
            efsbphase_1 = 0
            cnot_phase = 0.0
            efsbphase_2 = 0

            self.psb.append('q,mm'+str(self.id3),'pi_ge', phase = -gesbphase1)
            self.psb.append('q,mm'+str(self.id2),'pi_ef',phase= efsbphase_0 )
            self.psb.append('q','pi_q_ef', phase=cnot_phase)
            self.psb.append('q,mm'+str(self.id2),'pi_ef',phase=efsbphase_2)

            efsbphase_0=self.mode_mode_cnot_phase[self.id1][self.id3]
            efsbphase_1=self.mode_mode_cnot_phase2[self.id1][self.id3]
            gesbphase1=self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0
            cnot_phase = -90.0 + self.ef_qubit_phase_2

            self.psb.append('q,mm'+str(self.id1),'pi_ef',phase= efsbphase_0 )
            self.psb.append('q','pi_q_ef', phase=cnot_phase)
            self.psb.append('q,mm'+str(self.id1),'pi_ef',phase=efsbphase_2)

            if self.echo_pi_time_sweep:
                for i in arange(self.number_pi_pulses):
                    self.psb.append('q','pi',self.pulse_type,addphase=self.final_sb_offset+90.0)
                    self.psb.idle(pt/float(self.number_pi_pulses))

            self.psb.append('q','half_pi',self.pulse_type,addphase=self.final_sb_phase)



        elif tomo_index == 3:
            #<XYY> = + XII + CIY + CYI

            efsbphase_0=self.mode_mode_cnot_phase[self.id2][self.id3]
            efsbphase_1=self.mode_mode_cnot_phase2[self.id2][self.id3]
            gesbphase1=self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0
            efsbphase_1 = 0
            cnot_phase = -90.0
            efsbphase_2 = 0

            self.psb.append('q,mm'+str(self.id3),'pi_ge', phase = -gesbphase1)
            self.psb.append('q,mm'+str(self.id2),'pi_ef',phase= efsbphase_0 )
            self.psb.append('q','pi_q_ef', phase=cnot_phase)
            self.psb.append('q,mm'+str(self.id2),'pi_ef',phase=efsbphase_2)

            efsbphase_0=self.mode_mode_cnot_phase[self.id1][self.id3]
            efsbphase_1=self.mode_mode_cnot_phase2[self.id1][self.id3]
            gesbphase1=self.multimode_cfg[self.id3]['pi_pi_offset_phase']/2.0
            cnot_phase = -90.0 + self.ef_qubit_phase_2

            self.psb.append('q,mm'+str(self.id1),'pi_ef',phase= efsbphase_0 )
            self.psb.append('q','pi_q_ef', phase=cnot_phase)
            self.psb.append('q,mm'+str(self.id1),'pi_ef',phase=efsbphase_2)

            if self.echo_pi_time_sweep:
                for i in arange(self.number_pi_pulses):
                    self.psb.append('q','pi_y',self.pulse_type,addphase=self.final_sb_offset+90.0)
                    self.psb.idle(pt/float(self.number_pi_pulses))


            self.psb.append('q','half_pi_y',self.pulse_type,addphase= self.final_sb_phase)

class MultimodeThreeModeCorrelationSequence(QubitPulseSequence):

    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)
        self.tomo_index = self.extra_args['tomography_num']
        self.state_index = self.extra_args['state_num']

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        ## we define
        self.states_num = 1
        self.tomography_pulse_num = 15

        sequence_num = self.states_num*self.tomography_pulse_num
        self.expt_pts = np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)
        self.id1 = self.expt_cfg['id1']
        self.id2 = self.expt_cfg['id2']
        self.id3 = self.expt_cfg['id3']
        self.id = self.expt_cfg['id']
        self.half_pi_offset = self.pulse_cfg[self.pulse_type]['offset_phase']
        self.state_index =  self.expt_cfg['state_index']
        self.halfpicounter1 = 0
        self.halfpicounter2 = 0


# Figure out how to import config to psbuildergroup...and move these functions there

    def cxgate(self,control_id, x_id):
        self.psb.append('q,mm'+str(control_id),'pi_ge')
        self.psb.append('q,mm'+str(x_id),'pi_ef')
        self.psb.append('q','pi_q_ef',phase=0)
        self.psb.append('q,mm'+str(x_id),'pi_ef',phase=180)
        self.psb.append('q,mm'+str(control_id),'pi_ge',phase=180)

    def cygate(self,control_id, x_id):
        self.psb.append('q,mm'+str(control_id),'pi_ge')
        self.psb.append('q,mm'+str(x_id),'pi_ef')
        self.psb.append('q','pi_q_ef',phase=90)
        self.psb.append('q,mm'+str(x_id),'pi_ef',phase=180)
        self.psb.append('q,mm'+str(control_id),'pi_ge',phase=180)


    def define_pulses(self,pt):

        ### Initiate states
        self.define_states(pt)

    def define_states(self,pt):
        state_index = self.state_index
        self.halfpicounter1=0
        self.halfpicounter2=0

        if self.state_index== 0:

            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= self.expt_cfg['time_slice'])
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q','pi_q_ef')
            self.psb.append('q,mm'+str(self.id3),'pi_ef')
            self.psb.append('q,mm'+str(self.id2),'pi_ge')

        self.define_tomography_pulse(pt)

    def define_tomography_pulse(self,pt):
        ### gates before measurement for two resonaotor tomography
        if self.expt_cfg['test']:

            if self.tomo_index == 0:
                # <IZZ>

                self.cxgate(self.id2,self.id3)
                self.psb.append('q,mm'+str(self.id3),'pi_ge',phase=pt)

            if self.tomo_index == 1:
                # -<XXX>
                self.cxgate(self.id2,self.id1)
                self.cxgate(self.id2,self.id3)
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase= pt)
                self.psb.append('q','half_pi_y', self.pulse_type, phase = 90 + self.halfpicounter2*self.half_pi_offset)

            if self.tomo_index == 2:
                # <XYY>
                self.cxgate(self.id2,self.id1)
                self.cygate(self.id2,self.id3)
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=pt)
                self.psb.append('q','half_pi', self.pulse_type, phase = self.halfpicounter2*self.half_pi_offset)


            if self.tomo_index == 3:
                # <YXY>
                self.cygate(self.id2,self.id1)
                self.cygate(self.id2,self.id3)
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=pt)
                self.psb.append('q','half_pi_y', self.pulse_type, phase = 90 + self.halfpicounter2*self.half_pi_offset)


            if self.tomo_index == 4:
                # <YYX>
                self.cygate(self.id2,self.id1)
                self.cxgate(self.id2,self.id3)
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=pt)
                self.psb.append('q','half_pi', self.pulse_type, phase = self.halfpicounter2*self.half_pi_offset)


            if self.tomo_index == 5:
                # <ZIZ>

                self.cxgate(self.id1,self.id3)
                self.psb.append('q,mm'+str(self.id3),'pi_ge',phase=pt)

            if self.tomo_index == 6:
                # <ZZI>

                self.cxgate(self.id1,self.id2)
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=pt)



            if self.tomo_index == 7:
                # -<ZZZ>
                # self.psb.append('q','half_pi_y', self.pulse_type, phase = 90 + self.half_pi_offset)
                # self.psb.append('q,mm'+str(self.id1),'2pi_ef')
                # self.psb.append('q,mm'+str(self.id2),'2pi_ef')
                # self.psb.append('q,mm'+str(self.id3),'2pi_ef', phase=pt)
                # self.psb.append('q','half_pi_y', self.pulse_type, phase = 90 + self.half_pi_offset)
                cphase(self.psb,self.id2,self.id1)
                self.cxgate(self.id2,self.id3)
                self.psb.append('q,mm'+str(self.id3),'pi_ge', phase=pt)


            if self.tomo_index == 8:
                # -<YYY>
                self.cygate(self.id2,self.id1)
                self.cygate(self.id2,self.id3)
                self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=pt)
                self.psb.append('q','half_pi', self.pulse_type, phase = self.halfpicounter2*self.half_pi_offset)




        else:
            pass

class MultimodeSingleResonatorRandomizedBenchmarkingSequenceb(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.pulse_cfg = cfg['pulse_info']
        self.expt_cfg = expt_cfg
        self.multimode_cfg = cfg['multimodes']
        if 'mode' in kwargs:
            self.id = kwargs['mode']
        else:
            self.id = self.expt_cfg['id']

        print "Target id: " +str(self.id)

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg,self.define_points, self.define_parameters, self.define_pulses)

    def expmat(self, mat, theta):
        return np.cos(theta/2)*self.I - 1j*np.sin(theta/2)*mat

    def R_q(self,theta,phi):
        return np.cos(theta/2.0)*self.I -1j*np.sin(theta/2.0)*(np.cos(phi)*self.X + np.sin(phi)*self.Y)

    def define_points(self):
        if self.expt_cfg['knill_length_list']:
            self.expt_pts = np.array([2,3,4,5,6,8,10,12,16,20,24,32,40,48,64,80,96])
        else:
            self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.flux_pulse_type = self.multimode_cfg[int(self.id)]['flux_pulse_type']
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.clifford_pulse_1_list = ['0','half_pi_y','pi_y','neg_half_pi_y']
        self.clifford_pulse_2_list = ['0','half_pi','pi','neg_half_pi','half_pi_z','neg_half_pi_z']

        self.clifford_inv_pulse_1_list = ['0','half_pi_y','pi_y','neg_half_pi_y']
        self.clifford_inv_pulse_2_list = ['0','half_pi','pi','neg_half_pi','half_pi_z','neg_half_pi_z']

        ## Clifford and Pauli operators
        self.I = np.array([[1,0],[0,1]])
        self.X = np.array([[0,1],[1,0]])
        self.Y = np.array([[0,-1j],[1j,0]])
        self.Z = np.array([[1,0],[0,-1]])

        self.Pauli = [self.I,self.X,self.Y,self.Z]

        self.P_gen = np.empty([len(self.clifford_pulse_1_list),2,2],dtype=np.complex64)
        self.C_gen = np.empty([len(self.clifford_pulse_2_list),2,2],dtype=np.complex64)

        self.P_gen[0] = self.I
        self.P_gen[1] = self.expmat(self.Y,np.pi/2)
        self.P_gen[2] = self.expmat(self.Y, np.pi)
        self.P_gen[3] = self.expmat(self.Y,-np.pi/2)

        self.znumber=0
        self.xnumber=0

        if self.expt_cfg['phase_offset']:
            self.offset_phase = self.pulse_cfg['gauss']['offset_phase']
            if not self.expt_cfg['split_pi']:
                print "ERROR: Running offset phase correction without splitting pi pulse"
            else:
                pass
        else:
            self.offset_phase = 0

        print "Offset phase = %s"%(self.offset_phase)
        clist1 = [0,1,1,1,3,3] # index of IXYZ
        clist2 = [0, np.pi/2,np.pi,-np.pi/2,np.pi/2,-np.pi/2]

        for i in arange(len(self.clifford_pulse_2_list)):
            self.C_gen[i] = self.expmat(self.Pauli[clist1[i]],clist2[i])

        self.random_cliffords_1 = [random.randint(0,len(self.clifford_pulse_1_list)-1) for r in range(max(self.expt_pts))]
        self.random_cliffords_2 = [random.randint(0,len(self.clifford_pulse_2_list)-1) for r in range(max(self.expt_pts))]

        # self.random_cliffords_1 =  np.concatenate((np.array([0]),0*np.ones(max(self.expt_pts)-1)),axis=0).astype(int)
        # self.random_cliffords_2 =  np.concatenate((np.array([1]),1*np.ones(max(self.expt_pts)-1)),axis=0).astype(int)

        # print [self.clifford_pulse_1_list[jj] for jj in self.random_cliffords_1]
        # print [self.clifford_pulse_2_list[jj] for jj in self.random_cliffords_2]


    def define_pulses(self,pt):
        self.n = pt

        R = self.I
        self.znumber=0
        self.xnumber=0
        for jj in range(self.n):
            C1 = self.P_gen[self.random_cliffords_1[jj]]
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase = self.multimode_cfg[self.id]['pi_pi_offset_phase'])
            if (self.random_cliffords_1[jj] == 2):
                if self.expt_cfg['split_pi']:
                    self.psb.append('q','half_pi_y', self.pulse_type, addphase=self.xnumber*self.offset_phase + self.znumber*90)
                    self.xnumber+=1
                    self.psb.append('q','half_pi_y', self.pulse_type, addphase=self.xnumber*self.offset_phase + self.znumber*90)
                    self.xnumber+=1
                else:
                    self.psb.append('q',self.clifford_pulse_1_list[self.random_cliffords_1[jj]], self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
            else:
                self.psb.append('q',self.clifford_pulse_1_list[self.random_cliffords_1[jj]], self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                if (self.random_cliffords_1[jj] == 1) or (self.random_cliffords_1[jj] == 3):
                    self.xnumber +=1
            C2 = self.C_gen[self.random_cliffords_2[jj]]
            if self.random_cliffords_2[jj] == 4:
                if self.expt_cfg['z_phase']:
                    self.znumber-=1
                    self.psb.idle(self.pulse_cfg['gauss']['half_pi_length'])
                else:
                    self.psb.append('q','neg_half_pi', self.pulse_type)
                    self.psb.append('q','half_pi_y', self.pulse_type)
                    self.psb.append('q','half_pi', self.pulse_type)
            elif self.random_cliffords_2[jj] == 5:
                if self.expt_cfg['z_phase']:
                    self.znumber+=1
                    self.psb.idle(self.pulse_cfg['gauss']['half_pi_length'])
                else:
                    self.psb.append('q','neg_half_pi', self.pulse_type)
                    self.psb.append('q','neg_half_pi_y', self.pulse_type)
                    self.psb.append('q','half_pi', self.pulse_type)
            elif (self.random_cliffords_2[jj] == 2):
                if self.expt_cfg['split_pi']:
                    self.psb.append('q','half_pi', self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                    self.xnumber+=1
                    self.psb.append('q','half_pi', self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                    self.xnumber+=1
                else:
                    self.psb.append('q',self.clifford_pulse_2_list[self.random_cliffords_2[jj]], self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
            else:
                self.psb.append('q',self.clifford_pulse_2_list[self.random_cliffords_2[jj]], self.pulse_type, addphase=self.xnumber*self.offset_phase + self.znumber*90)
                if (self.random_cliffords_2[jj] == 1) or (self.random_cliffords_2[jj] == 3):
                    self.xnumber+=1

            C = np.dot(C2,C1)
            R = np.dot(C,R)
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=0)
        self.final_pulse_dictionary(R)

    def final_pulse_dictionary(self,R_input):
        g_e_random = random.randint(0,1)

        found = 0

        for zz in range(4):
            R = ((1j)**zz)*R_input
            # inversepulselist2 = []
            # inversepulselist1 = []
            for ii in range(len(self.clifford_inv_pulse_1_list)):
                for jj in range(len(self.clifford_inv_pulse_2_list)):
                    C1 = self.P_gen[ii]
                    C2 = self.C_gen[jj]
                    C = np.dot(C2,C1)

                    if np.allclose(np.real(self.I),np.real(np.dot(C,R))) and np.allclose(np.imag(self.I),np.imag(np.dot(C,R))):
                        found +=1
                        # print "---" + str(self.n)
                        # print "Number of z pulses in creation sequence %s" %(self.znumber)
                        # print self.clifford_inv_pulse_1_list[ii]
                        # print self.clifford_inv_pulse_2_list[jj]
                        self.psb.append('q,mm'+str(self.id),'pi_ge',phase= self.multimode_cfg[self.id]['pi_pi_offset_phase'])
                        if (ii == 2) and self.expt_cfg['split_pi']:
                            self.psb.append('q','half_pi_y', self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                            self.xnumber+=1
                            self.psb.append('q','half_pi_y', self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                            self.xnumber+=1
                        else:
                            self.psb.append('q',self.clifford_inv_pulse_1_list[ii], self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                            if (ii == 1) or (ii == 3):
                                self.xnumber+=1

                        if jj == 4:
                            if self.expt_cfg['z_phase']:
                                self.znumber-=1
                                self.psb.idle(self.pulse_cfg['gauss']['half_pi_length'])
                            else:
                                self.psb.append('q','neg_half_pi', self.pulse_type)
                                self.psb.append('q','half_pi_y', self.pulse_type)
                                self.psb.append('q','half_pi', self.pulse_type)
                        elif jj == 5:
                            if self.expt_cfg['z_phase']:
                                self.znumber+=1
                                self.psb.idle(self.pulse_cfg['gauss']['half_pi_length'])
                            else:
                                self.psb.append('q','neg_half_pi', self.pulse_type)
                                self.psb.append('q','neg_half_pi_y', self.pulse_type)
                                self.psb.append('q','half_pi', self.pulse_type)

                        elif (jj == 2) and self.expt_cfg['split_pi']:
                            self.psb.append('q','half_pi', self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                            self.xnumber+=1
                            self.psb.append('q','half_pi', self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                            self.xnumber+=1
                        else:
                            self.psb.append('q',self.clifford_inv_pulse_2_list[jj], self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                            if (jj==1) or (jj==3):
                                self.xnumber+=1

        if found == 0 :
            print "Error! Some pulse's inverse was not found."
        elif found > 1:
            print "Error! Non unique inverse."

class MultimodeSingleResonatorRandomizedBenchmarkingSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.pulse_cfg = cfg['pulse_info']
        self.expt_cfg = expt_cfg
        self.multimode_cfg = cfg['multimodes']
        if 'mode' in kwargs:
            self.id = kwargs['mode']
        else:
            self.id = self.expt_cfg['id']

        print "Target id: " +str(self.id)

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg,self.define_points, self.define_parameters, self.define_pulses)

    def expmat(self, mat, theta):
        return np.cos(theta/2)*self.I - 1j*np.sin(theta/2)*mat

    def R_q(self,theta,phi):
        return np.cos(theta/2.0)*self.I -1j*np.sin(theta/2.0)*(np.cos(phi)*self.X + np.sin(phi)*self.Y)

    def define_points(self):
        if self.expt_cfg['knill_length_list']:
            self.expt_pts = np.array([2,3,4,5,6,8,10,12,16,20,24,32,40,48,64,80,96])
        else:
            self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.flux_pulse_type = self.multimode_cfg[int(self.id)]['flux_pulse_type']
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.clifford_pulse_1_list = ['0','half_pi_y','pi_y','neg_half_pi_y']
        self.clifford_pulse_2_list = ['0','half_pi','pi','neg_half_pi','half_pi_z','neg_half_pi_z']

        self.clifford_inv_pulse_1_list = ['0','half_pi_y','pi_y','neg_half_pi_y']
        self.clifford_inv_pulse_2_list = ['0','half_pi','pi','neg_half_pi','half_pi_z','neg_half_pi_z']

        ## Clifford and Pauli operators
        self.I = np.array([[1,0],[0,1]])
        self.X = np.array([[0,1],[1,0]])
        self.Y = np.array([[0,-1j],[1j,0]])
        self.Z = np.array([[1,0],[0,-1]])

        self.Pauli = [self.I,self.X,self.Y,self.Z]

        self.P_gen = np.empty([len(self.clifford_pulse_1_list),2,2],dtype=np.complex64)
        self.C_gen = np.empty([len(self.clifford_pulse_2_list),2,2],dtype=np.complex64)

        self.P_gen[0] = self.I
        self.P_gen[1] = self.expmat(self.Y,np.pi/2)
        self.P_gen[2] = self.expmat(self.Y, np.pi)
        self.P_gen[3] = self.expmat(self.Y,-np.pi/2)

        self.znumber=0
        self.xnumber=0

        if self.expt_cfg['phase_offset']:
            self.offset_phase = self.pulse_cfg['gauss']['offset_phase']
            if not self.expt_cfg['split_pi']:
                print "ERROR: Running offset phase correction without splitting pi pulse"
            else:
                pass
        else:
            self.offset_phase = 0

        print "Offset phase = %s"%(self.offset_phase)
        clist1 = [0,1,1,1,3,3] # index of IXYZ
        clist2 = [0, np.pi/2,np.pi,-np.pi/2,np.pi/2,-np.pi/2]

        for i in arange(len(self.clifford_pulse_2_list)):
            self.C_gen[i] = self.expmat(self.Pauli[clist1[i]],clist2[i])

        self.random_cliffords_1 = [random.randint(0,len(self.clifford_pulse_1_list)-1) for r in range(max(self.expt_pts))]
        self.random_cliffords_2 = [random.randint(0,len(self.clifford_pulse_2_list)-1) for r in range(max(self.expt_pts))]

        # self.random_cliffords_1 =  np.concatenate((np.array([0]),0*np.ones(max(self.expt_pts)-1)),axis=0).astype(int)
        # self.random_cliffords_2 =  np.concatenate((np.array([1]),1*np.ones(max(self.expt_pts)-1)),axis=0).astype(int)

        # print [self.clifford_pulse_1_list[jj] for jj in self.random_cliffords_1]
        # print [self.clifford_pulse_2_list[jj] for jj in self.random_cliffords_2]


    def define_pulses(self,pt):
        self.n = pt

        R = self.I
        self.znumber=0
        self.xnumber=0
        for jj in range(self.n):
            C1 = self.P_gen[self.random_cliffords_1[jj]]
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase = -self.multimode_cfg[self.id]['pi_pi_offset_phase']/2.0)
            if (self.random_cliffords_1[jj] == 2):
                if self.expt_cfg['split_pi']:
                    self.psb.append('q','half_pi_y', self.pulse_type, addphase=self.xnumber*self.offset_phase + self.znumber*90)
                    self.xnumber+=1
                    self.psb.append('q','half_pi_y', self.pulse_type, addphase=self.xnumber*self.offset_phase + self.znumber*90)
                    self.xnumber+=1
                else:
                    self.psb.append('q',self.clifford_pulse_1_list[self.random_cliffords_1[jj]], self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
            else:
                self.psb.append('q',self.clifford_pulse_1_list[self.random_cliffords_1[jj]], self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                if (self.random_cliffords_1[jj] == 1) or (self.random_cliffords_1[jj] == 3):
                    self.xnumber +=1
            C2 = self.C_gen[self.random_cliffords_2[jj]]
            if self.random_cliffords_2[jj] == 4:
                if self.expt_cfg['z_phase']:
                    self.znumber-=1
                    self.psb.idle(self.pulse_cfg['gauss']['half_pi_length'])
                else:
                    self.psb.append('q','neg_half_pi', self.pulse_type)
                    self.psb.append('q','half_pi_y', self.pulse_type)
                    self.psb.append('q','half_pi', self.pulse_type)
            elif self.random_cliffords_2[jj] == 5:
                if self.expt_cfg['z_phase']:
                    self.znumber+=1
                    self.psb.idle(self.pulse_cfg['gauss']['half_pi_length'])
                else:
                    self.psb.append('q','neg_half_pi', self.pulse_type)
                    self.psb.append('q','neg_half_pi_y', self.pulse_type)
                    self.psb.append('q','half_pi', self.pulse_type)
            elif (self.random_cliffords_2[jj] == 2):
                if self.expt_cfg['split_pi']:
                    self.psb.append('q','half_pi', self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                    self.xnumber+=1
                    self.psb.append('q','half_pi', self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                    self.xnumber+=1
                else:
                    self.psb.append('q',self.clifford_pulse_2_list[self.random_cliffords_2[jj]], self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
            else:
                self.psb.append('q',self.clifford_pulse_2_list[self.random_cliffords_2[jj]], self.pulse_type, addphase=self.xnumber*self.offset_phase + self.znumber*90)
                if (self.random_cliffords_2[jj] == 1) or (self.random_cliffords_2[jj] == 3):
                    self.xnumber+=1

            C = np.dot(C2,C1)
            R = np.dot(C,R)
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=+self.multimode_cfg[self.id]['pi_pi_offset_phase']/2.0 + 180.0)
        self.final_pulse_dictionary(R)

    def final_pulse_dictionary(self,R_input):
        g_e_random = random.randint(0,1)

        found = 0

        for zz in range(4):
            R = ((1j)**zz)*R_input
            # inversepulselist2 = []
            # inversepulselist1 = []
            for ii in range(len(self.clifford_inv_pulse_1_list)):
                for jj in range(len(self.clifford_inv_pulse_2_list)):
                    C1 = self.P_gen[ii]
                    C2 = self.C_gen[jj]
                    C = np.dot(C2,C1)

                    if np.allclose(np.real(self.I),np.real(np.dot(C,R))) and np.allclose(np.imag(self.I),np.imag(np.dot(C,R))):
                        found +=1
                        # print "---" + str(self.n)
                        # print "Number of z pulses in creation sequence %s" %(self.znumber)
                        # print self.clifford_inv_pulse_1_list[ii]
                        # print self.clifford_inv_pulse_2_list[jj]
                        self.psb.append('q,mm'+str(self.id),'pi_ge',phase= -self.multimode_cfg[self.id]['pi_pi_offset_phase']/(2.0))
                        if (ii == 2) and self.expt_cfg['split_pi']:
                            self.psb.append('q','half_pi_y', self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                            self.xnumber+=1
                            self.psb.append('q','half_pi_y', self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                            self.xnumber+=1
                        else:
                            self.psb.append('q',self.clifford_inv_pulse_1_list[ii], self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                            if (ii == 1) or (ii == 3):
                                self.xnumber+=1

                        if jj == 4:
                            if self.expt_cfg['z_phase']:
                                self.znumber-=1
                                self.psb.idle(self.pulse_cfg['gauss']['half_pi_length'])
                            else:
                                self.psb.append('q','neg_half_pi', self.pulse_type)
                                self.psb.append('q','half_pi_y', self.pulse_type)
                                self.psb.append('q','half_pi', self.pulse_type)
                        elif jj == 5:
                            if self.expt_cfg['z_phase']:
                                self.znumber+=1
                                self.psb.idle(self.pulse_cfg['gauss']['half_pi_length'])
                            else:
                                self.psb.append('q','neg_half_pi', self.pulse_type)
                                self.psb.append('q','neg_half_pi_y', self.pulse_type)
                                self.psb.append('q','half_pi', self.pulse_type)

                        elif (jj == 2) and self.expt_cfg['split_pi']:
                            self.psb.append('q','half_pi', self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                            self.xnumber+=1
                            self.psb.append('q','half_pi', self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                            self.xnumber+=1
                        else:
                            self.psb.append('q',self.clifford_inv_pulse_2_list[jj], self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                            if (jj==1) or (jj==3):
                                self.xnumber+=1

        if found == 0 :
            print "Error! Some pulse's inverse was not found."
        elif found > 1:
            print "Error! Non unique inverse."


        #print "Total number of half pi pulses = %s"%(self.xnumber)

class MultimodeQubitModeRandomizedBenchmarkingSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.pulse_cfg = cfg['pulse_info']
        self.expt_cfg = expt_cfg
        self.multimode_cfg = cfg['multimodes']
        if 'mode' in kwargs:
            self.id = kwargs['mode']
        else:
            self.id = self.expt_cfg['id']

        print "Target id: " +str(self.id)

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg,self.define_points, self.define_parameters, self.define_pulses)

    def expmat(self, mat, theta):
        return np.cos(theta/2)*self.I - 1j*np.sin(theta/2)*mat

    def R_q(self,theta,phi):
        return np.cos(theta/2.0)*self.I -1j*np.sin(theta/2.0)*(np.cos(phi)*self.X + np.sin(phi)*self.Y)

    def define_points(self):
        if self.expt_cfg['knill_length_list']:
            self.expt_pts = np.array([2,3,4,5,6,8,10,12,16,20,24,32,40,48,64,80,96])
        else:
            self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.flux_pulse_type = self.multimode_cfg[int(self.id)]['flux_pulse_type']
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.clifford_pulse_1_list = ['0','half_pi_y','pi_y','neg_half_pi_y']
        self.clifford_pulse_2_list = ['0','half_pi','pi','neg_half_pi','half_pi_z','neg_half_pi_z']

        self.clifford_inv_pulse_1_list = ['0','half_pi_y','pi_y','neg_half_pi_y']
        self.clifford_inv_pulse_2_list = ['0','half_pi','pi','neg_half_pi','half_pi_z','neg_half_pi_z']

        ## Clifford and Pauli operators
        self.I = np.array([[1,0],[0,1]])
        self.X = np.array([[0,1],[1,0]])
        self.Y = np.array([[0,-1j],[1j,0]])
        self.Z = np.array([[1,0],[0,-1]])

        self.Pauli = [self.I,self.X,self.Y,self.Z]

        self.P_gen = np.empty([len(self.clifford_pulse_1_list),2,2],dtype=np.complex64)
        self.C_gen = np.empty([len(self.clifford_pulse_2_list),2,2],dtype=np.complex64)

        self.P_gen[0] = self.I
        self.P_gen[1] = self.expmat(self.Y,np.pi/2)
        self.P_gen[2] = self.expmat(self.Y, np.pi)
        self.P_gen[3] = self.expmat(self.Y,-np.pi/2)

        self.znumber=0
        self.xnumber=0

        if self.expt_cfg['phase_offset']:
            self.offset_phase = self.pulse_cfg['gauss']['offset_phase']
            if not self.expt_cfg['split_pi']:
                print "ERROR: Running offset phase correction without splitting pi pulse"
            else:
                pass
        else:
            self.offset_phase = 0

        print "Offset phase = %s"%(self.offset_phase)
        clist1 = [0,1,1,1,3,3] # index of IXYZ
        clist2 = [0, np.pi/2,np.pi,-np.pi/2,np.pi/2,-np.pi/2]

        for i in arange(len(self.clifford_pulse_2_list)):
            self.C_gen[i] = self.expmat(self.Pauli[clist1[i]],clist2[i])

        self.random_cliffords_1 = [random.randint(0,len(self.clifford_pulse_1_list)-1) for r in range(max(self.expt_pts))]
        self.random_cliffords_2 = [random.randint(0,len(self.clifford_pulse_2_list)-1) for r in range(max(self.expt_pts))]

        # self.random_cliffords_1 =  np.concatenate((np.array([0]),0*np.ones(max(self.expt_pts)-1)),axis=0).astype(int)
        # self.random_cliffords_2 =  np.concatenate((np.array([1]),1*np.ones(max(self.expt_pts)-1)),axis=0).astype(int)

        # print [self.clifford_pulse_1_list[jj] for jj in self.random_cliffords_1]
        # print [self.clifford_pulse_2_list[jj] for jj in self.random_cliffords_2]


    def define_pulses(self,pt):
        self.n = pt

        R = self.I
        self.znumber=0
        self.xnumber=0

        # Create excitation in mode

        # self.psb.append('q','pi',self.pulse_type)
        # self.psb.append('q,mm'+str(self.id),'pi_ge')

        for jj in range(self.n):
            C1 = self.P_gen[self.random_cliffords_1[jj]]
            if (self.random_cliffords_1[jj] == 2):
                if self.expt_cfg['split_pi']:
                    self.psb.append('q','half_pi_y', self.pulse_type, addphase=self.xnumber*self.offset_phase + self.znumber*90)
                    self.xnumber+=1
                    self.psb.append('q','half_pi_y', self.pulse_type, addphase=self.xnumber*self.offset_phase + self.znumber*90)
                    self.xnumber+=1
                else:
                    self.psb.append('q',self.clifford_pulse_1_list[self.random_cliffords_1[jj]], self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
            else:
                self.psb.append('q',self.clifford_pulse_1_list[self.random_cliffords_1[jj]], self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                if (self.random_cliffords_1[jj] == 1) or (self.random_cliffords_1[jj] == 3):
                    self.xnumber +=1
            C2 = self.C_gen[self.random_cliffords_2[jj]]
            if self.random_cliffords_2[jj] == 4:
                if self.expt_cfg['z_phase']:
                    self.znumber-=1
                    self.psb.idle(self.pulse_cfg['gauss']['half_pi_length'])
                else:
                    self.psb.append('q','neg_half_pi', self.pulse_type)
                    self.psb.append('q','half_pi_y', self.pulse_type)
                    self.psb.append('q','half_pi', self.pulse_type)
            elif self.random_cliffords_2[jj] == 5:
                if self.expt_cfg['z_phase']:
                    self.znumber+=1
                    self.psb.idle(self.pulse_cfg['gauss']['half_pi_length'])
                else:
                    self.psb.append('q','neg_half_pi', self.pulse_type)
                    self.psb.append('q','neg_half_pi_y', self.pulse_type)
                    self.psb.append('q','half_pi', self.pulse_type)
            elif (self.random_cliffords_2[jj] == 2):
                if self.expt_cfg['split_pi']:
                    self.psb.append('q','half_pi', self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                    self.xnumber+=1
                    self.psb.append('q','half_pi', self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                    self.xnumber+=1
                else:
                    self.psb.append('q',self.clifford_pulse_2_list[self.random_cliffords_2[jj]], self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
            else:
                self.psb.append('q',self.clifford_pulse_2_list[self.random_cliffords_2[jj]], self.pulse_type, addphase=self.xnumber*self.offset_phase + self.znumber*90)
                if (self.random_cliffords_2[jj] == 1) or (self.random_cliffords_2[jj] == 3):
                    self.xnumber+=1

            C = np.dot(C2,C1)
            # C = np.dot(self.Z,C)
            R = np.dot(C,R)
            # self.psb.append('q,mm'+str(self.id),'2pi_ef')
            # self.psb.idle(100)

        self.final_pulse_dictionary(R)

    def final_pulse_dictionary(self,R_input):
        g_e_random = random.randint(0,1)

        found = 0

        for zz in range(4):
            R = ((1j)**zz)*R_input
            # inversepulselist2 = []
            # inversepulselist1 = []
            for ii in range(len(self.clifford_inv_pulse_1_list)):
                for jj in range(len(self.clifford_inv_pulse_2_list)):
                    C1 = self.P_gen[ii]
                    C2 = self.C_gen[jj]
                    C = np.dot(C2,C1)

                    if np.allclose(np.real(self.I),np.real(np.dot(C,R))) and np.allclose(np.imag(self.I),np.imag(np.dot(C,R))):
                        found +=1

                        if (ii == 2) and self.expt_cfg['split_pi']:
                            self.psb.append('q','half_pi_y', self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                            self.xnumber+=1
                            self.psb.append('q','half_pi_y', self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                            self.xnumber+=1
                        else:
                            self.psb.append('q',self.clifford_inv_pulse_1_list[ii], self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                            if (ii == 1) or (ii == 3):
                                self.xnumber+=1

                        if jj == 4:
                            if self.expt_cfg['z_phase']:
                                self.znumber-=1
                                self.psb.idle(self.pulse_cfg['gauss']['half_pi_length'])
                            else:
                                self.psb.append('q','neg_half_pi', self.pulse_type)
                                self.psb.append('q','half_pi_y', self.pulse_type)
                                self.psb.append('q','half_pi', self.pulse_type)
                        elif jj == 5:
                            if self.expt_cfg['z_phase']:
                                self.znumber+=1
                                self.psb.idle(self.pulse_cfg['gauss']['half_pi_length'])
                            else:
                                self.psb.append('q','neg_half_pi', self.pulse_type)
                                self.psb.append('q','neg_half_pi_y', self.pulse_type)
                                self.psb.append('q','half_pi', self.pulse_type)

                        elif (jj == 2) and self.expt_cfg['split_pi']:
                            self.psb.append('q','half_pi', self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                            self.xnumber+=1
                            self.psb.append('q','half_pi', self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                            self.xnumber+=1
                        else:
                            self.psb.append('q',self.clifford_inv_pulse_2_list[jj], self.pulse_type, addphase=self.xnumber*self.offset_phase+self.znumber*90)
                            if (jj==1) or (jj==3):
                                self.xnumber+=1

        if found == 0 :
            print "Error! Some pulse's inverse was not found."
        elif found > 1:
            print "Error! Non unique inverse."

class MultimodeCPhaseAmplificationSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value


        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        if self.expt_cfg['sweep_phase']:
            self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])
        else:
            self.expt_pts = arange(self.expt_cfg['number'])

    def define_parameters(self):

        if 'mode_1' in self.extra_args:
            self.id1 = self.extra_args['mode_1']
        else:
            self.id1 = self.expt_cfg['id1']

        if 'mode_2' in self.extra_args:
            self.id2 = self.extra_args['mode_2']
        else:
            self.id2 = self.expt_cfg['id2']

        if 'number' in self.extra_args:
            self.number = self.extra_args['number']
        else:
            self.number = self.expt_cfg['number']

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.multimode_cfg[int(self.id1)]['flux_pulse_type']
        self.flux_pulse_type_ef = self.multimode_cfg[int(self.id2)]['flux_pulse_type_ef']

        self.offset_phase = self.pulse_cfg[self.pulse_type]['offset_phase']


    def define_pulses(self,pt):

        if self.expt_cfg['sweep_phase']:
            finalphase = pt +  self.multimode_cfg[self.id1]['pi_pi_offset_phase']
        else:
            ef_2pi_phase = self.multimode_cfg[self.id1]['pi_pi_offset_phase']


        self.psb.append('q','half_pi', self.pulse_type)
        self.psb.append('q,mm'+str(self.id1),'pi_ge')

        for i in arange(self.number+1):
            self.psb.append('q,mm'+str(self.id1),'pi_ge', phase = self.multimode_cfg[self.id1]['pi_pi_offset_phase'])
            self.psb.append('q,mm'+str(self.id2),'2pi_ef')
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase= self.multimode_cfg[self.id2]['ef_2pi_offset_phase'])


        self.psb.append('q,mm'+str(self.id1),'pi_ge', phase = finalphase)
        self.psb.append('q','half_pi', self.pulse_type, phase = self.offset_phase)

class MultimodeCNOTAmplificationSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.cfg = cfg
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value


        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        if self.expt_cfg['sweep_phase']:
            self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])
        else:
            self.expt_pts = arange(self.expt_cfg['number'])

    def define_parameters(self):

        if 'mode_1' in self.extra_args:
            self.id1 = self.extra_args['mode_1']
        else:
            self.id1 = self.expt_cfg['id1']

        if 'mode_2' in self.extra_args:
            self.id2 = self.extra_args['mode_2']
        else:
            self.id2 = self.expt_cfg['id2']

        if 'number' in self.extra_args:
            self.number = self.extra_args['number']
        else:
            self.number = self.expt_cfg['number']

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.multimode_cfg[int(self.id1)]['flux_pulse_type']
        self.flux_pulse_type_ef = self.multimode_cfg[int(self.id2)]['flux_pulse_type_ef']
        self.offset_phase = self.pulse_cfg[self.pulse_type]['offset_phase']
        self.mode_mode_cnot_dc_phase = self.cfg['mode_mode_offset']['cnot_dc_phase']
        self.mode_mode_cnot_phase = self.cfg['mode_mode_offset']['cnot_phase']
        self.mode_mode_cnot_phase2 = self.cfg['mode_mode_offset']['cnot_phase2']

    def define_pulses(self,pt):
        cnot_phase = self.mode_mode_cnot_phase[self.id2][self.id1]
        cnot_phase2 = self.mode_mode_cnot_phase2[self.id2][self.id1]
        sideband_phase = self.mode_mode_cnot_dc_phase[self.id2][self.id1] + pt


        self.psb.append('q','half_pi', self.pulse_type)
        self.psb.append('q,mm'+str(self.id1),'pi_ge')

        if self.number > 0:
            for i in arange(self.number):
                 cnot_v2(self.psb,self.id1,self.id2,cnot_phase=0,efsbphase_0=cnot_phase,efsbphase_1=cnot_phase2)
        if self.number%2==0:
            self.psb.append('q,mm'+str(self.id1),'pi_ge',self.multimode_cfg[int(self.id1)]['pi_pi_offset_phase'])
            self.psb.append('q','half_pi', self.pulse_type, phase = self.offset_phase + pt)
        else:
            self.psb.append('q,mm'+str(self.id1),'pi_ge', phase = pt + self.multimode_cfg[self.id1]['pi_pi_offset_phase'])
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q','pi_q_ef')
            self.psb.append('q','half_pi', self.pulse_type, phase = self.offset_phase)

class Multimode_State_Dep_Shift_Sequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.flux_pulse_cfg = cfg['flux_pulse_info']
        self.multimode_cfg = cfg['multimodes']

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value



        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):

        if 'exp' in self.extra_args:
            self.exp = self.extra_args['exp']
        else:
            self.exp = self.expt_cfg['exp']

        # 'exp' == 1 and 'exp' == 2 (drive at Chi shifted frequency) corresponding to qubit rabi with photon loaded

        if self.exp==1 or self.exp== 2 or self.exp == 6:
            self.expt_pts =arange(self.expt_cfg['start_rabi'], self.expt_cfg['stop_rabi'], self.expt_cfg['step_rabi'])

        # 'exp' == 3 and 'exp' == 4 (drive at Chi shifted frequency): shift in offset phase of qubit due to photon loaded

        elif self.exp==3 or self.exp== 4 or self.exp== 5:

            self.expt_pts = arange(self.expt_cfg['start_phase'],self.expt_cfg['stop_phase'],self.expt_cfg['step_phase'])
        else:
            self.expt_pts = arange(self.expt_cfg['start_ram'], self.expt_cfg['stop_ram'], self.expt_cfg['step_ram'])


    def define_parameters(self):
        if 'mode' in self.extra_args:
            self.id = self.extra_args['mode']
        else:
            self.id = self.expt_cfg['id']


        if 'mode2' in self.extra_args:
            self.id2 = self.extra_args['mode2']
        else:
            self.id2 = self.expt_cfg['id2']

        if 'load_photon' in self.extra_args:
            self.load_photon = self.extra_args['load_photon']
        else:
            self.load_photon = self.expt_cfg['load_photon']

        if 'qubit_shift_ge' in self.extra_args:
            self.qubit_shift_ge = self.extra_args['qubit_shift_ge']
        else:
            self.qubit_shift_ge = self.expt_cfg['qubit_shift_ge']

        if 'subexp' in self.extra_args:
            self.subexp = self.extra_args['subexp']
        else:
            self.subexp = 0

        if 'qubit_shift_ef' in self.extra_args:
            self.qubit_shift_ef = self.extra_args['qubit_shift_ef']
        else:
            self.qubit_shift_ef = self.expt_cfg['qubit_shift_ef']




        print "Target id: " +str(self.id)

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)
        self.offset_phase = self.pulse_cfg['gauss']['offset_phase']

        if self.flux_pulse_cfg['fix_phase']:
            self.phase_freq = self.expt_cfg['ramsey_freq']
        else:
            self.phase_freq = self.multimode_cfg[int(self.id)]['dc_offset_freq'] + self.expt_cfg['ramsey_freq']


    def define_pulses(self,pt):


        if self.qubit_shift_ge == 1:

            if self.exp==0:

                if self.load_photon:
                    self.psb.append('q','pi', self.pulse_type)
                    self.psb.append('q,mm'+str(self.id),'pi_ge')

                self.psb.append('q','half_pi', self.pulse_type)
                if self.expt_cfg['echo']:
                    self.psb.idle(pt/2.0)
                    self.psb.append('q','pi', self.pulse_type)
                    self.psb.idle(pt/2.0)
                else:
                    self.psb.idle(pt)
                self.psb.append('q','half_pi', self.pulse_type, phase = self.offset_phase + 360.0*self.expt_cfg['ramsey_freq']*pt/(1.0e9))

            elif self.exp==3:

                if self.load_photon:
                    self.psb.append('q','pi', self.pulse_type)
                    self.psb.append('q,mm'+str(self.id),'pi_ge')

                self.psb.append('q','half_pi', self.pulse_type)
                self.psb.append('q','half_pi', self.pulse_type, phase = pt)

            elif self.exp==4:

                if self.load_photon:
                    self.psb.append('q','pi', self.pulse_type)
                    self.psb.append('q,mm'+str(self.id),'pi_ge')

                self.psb.append('q','half_pi', self.pulse_type)
                self.psb.append('q','half_pi', self.pulse_type, phase = pt,freq=self.pulse_cfg['gauss']['iq_freq'] + self.multimode_cfg[self.id]['shift'])

            elif self.exp==5:

                if self.load_photon:
                    self.psb.append('q','pi', self.pulse_type)
                    self.psb.append('q,mm'+str(self.id2),'pi_ge')

                self.psb.append('q','half_pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id),'pi_ge')
                self.psb.append('q,mm'+str(self.id),'pi_ge',phase=pt)
                self.psb.append('q','half_pi', self.pulse_type, phase=self.offset_phase)


            else:

                if self.load_photon:
                    self.psb.append('q','pi', self.pulse_type)
                    self.psb.append('q,mm'+str(self.id),'pi_ge')

                if self.exp == 1:
                    self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a'], length=pt,freq=self.pulse_cfg['gauss']['iq_freq'])

                elif self.exp == 2:
                    self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a'], length=pt,freq=self.pulse_cfg['gauss']['iq_freq'] + self.multimode_cfg[self.id]['shift'])



        elif self.qubit_shift_ef == 1:

            if self.exp==0:

                if self.load_photon:
                    self.psb.append('q','pi', self.pulse_type)
                    self.psb.append('q,mm'+str(self.id),'pi_ge')


                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q','half_pi_q_ef')

                if self.expt_cfg['echo']:
                    self.psb.idle(pt/2.0)
                    self.psb.append('q','pi_q_ef')
                    self.psb.idle(pt/2.0)
                else:
                    self.psb.idle(pt)

                self.psb.append('q','half_pi_q_ef',phase = 360.0*self.expt_cfg['ramsey_freq']*pt/(1.0e9))
                self.psb.append('q','pi', self.pulse_type)
                # Calibrate ef Ramsey
                self.psb.append('q','pi_q_ef')

            elif self.exp==3:

                if self.load_photon:
                    self.psb.append('q','pi', self.pulse_type)
                    self.psb.append('q,mm'+str(self.id),'pi_ge')

                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q','half_pi_q_ef', self.pulse_type)
                self.psb.append('q','half_pi_q_ef', self.pulse_type, phase = pt)
                self.psb.append('q','pi', self.pulse_type)
                # Calibrate ef Ramsey
                self.psb.append('q','pi_q_ef')

            elif self.exp==6:

                if self.subexp==0:
                    self.psb.append('q','pi', self.pulse_type)
                    self.psb.append('q','pi_q_ef')
                elif self.subexp==1:
                    self.psb.append('q','pi', self.pulse_type)
                    self.psb.append('q,mm'+str(self.id2),'pi_ge')
                    self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])
                    self.psb.append('q','pi_q_ef')
                elif self.subexp==2:
                    self.psb.append('q','pi', self.pulse_type)
                    self.psb.append('q,mm'+str(self.id2),'pi_ge')
                    self.psb.append('q','pi', self.pulse_type)
                    self.psb.append('q','pi_q_ef')
                elif self.subexp==3:
                    self.psb.append('q','pi', self.pulse_type)
                    self.psb.append('q,mm'+str(self.id),'pi_ge')
                    self.psb.append('q','pi', self.pulse_type)

                elif self.subexp==4:
                    self.psb.append('q','pi', self.pulse_type)
                    self.psb.append('q,mm'+str(self.id),'pi_ge')
                    self.psb.append('q','pi', self.pulse_type)
                    self.psb.append('q,mm'+str(self.id2),'pi_ge')
                    self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])



                self.psb.append('q:mm','general', self.multimode_cfg[self.id]['flux_pulse_type'], amp=self.multimode_cfg[self.id]['a_ef'], length=pt,freq=self.multimode_cfg[self.id]['flux_pulse_freq_ef'])
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q','pi_q_ef')


            else:

                if self.load_photon:
                    self.psb.append('q','pi', self.pulse_type)
                    self.psb.append('q,mm'+str(self.id),'pi_ge')

                if self.exp == 1:
                    self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a'], length=pt,freq=self.pulse_cfg['gauss']['iq_freq'])

                elif self.exp == 2:
                    self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a'], length=pt,freq=self.pulse_cfg['gauss']['iq_freq'] + self.multimode_cfg[self.id]['shift'])




        else:

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge')

            if self.expt_cfg['excite_qubit']:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.idle(pt)
                self.psb.append('q','pi', self.pulse_type)
            else:
                 self.psb.idle(pt)

            self.psb.append('q,mm'+str(self.id),'pi_ge', phase = 360.0*self.phase_freq*pt/(1.0e9))
            self.psb.append('q','half_pi', self.pulse_type,phase=self.offset_phase)

class Multimode_State_Dep_Shift_Sequenceb(QubitPulseSequenceSBcool):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.flux_pulse_cfg = cfg['flux_pulse_info']
        self.multimode_cfg = cfg['multimodes']

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        QubitPulseSequenceSBcool.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):

        if 'exp' in self.extra_args:
            self.exp = self.extra_args['exp']
        else:
            self.exp = self.expt_cfg['exp']
        # 'exp' == 1 and 'exp' == 2 (drive at Chi shifted frequency) corresponding to qubit rabi with photon loaded

        if self.exp==1 or self.exp== 2 or self.exp == 6 or self.exp ==7 or self.exp == 8:
            self.expt_pts =arange(self.expt_cfg['start_rabi'], self.expt_cfg['stop_rabi'], self.expt_cfg['step_rabi'])

        # 'exp' == 3 and 'exp' == 4 (drive at Chi shifted frequency): shift in offset phase of qubit due to photon loaded

        elif self.exp==3 or self.exp== 4 or self.exp== 5 or self.exp == 11 or self.exp == 12:

            self.expt_pts = arange(self.expt_cfg['start_phase'],self.expt_cfg['stop_phase'],self.expt_cfg['step_phase'])

        elif self.exp==10 :

            self.expt_pts = arange(16)

        elif self.exp==13:

            self.expt_pts = arange(18)

        elif self.exp==14:

            self.expt_pts = arange(27)

        elif self.exp==15:

            self.expt_pts = arange(29)


        else:
            self.expt_pts = arange(self.expt_cfg['start_ram'], self.expt_cfg['stop_ram'], self.expt_cfg['step_ram'])


    def define_parameters(self):
        if 'mode' in self.extra_args:
            self.id = self.extra_args['mode']
        else:
            self.id = self.expt_cfg['id']


        if 'mode2' in self.extra_args:
            self.id2 = self.extra_args['mode2']
        else:
            self.id2 = self.expt_cfg['id2']

        if 'load_photon' in self.extra_args:
            self.load_photon = self.extra_args['load_photon']
        else:
            self.load_photon = self.expt_cfg['load_photon']

        if 'shift_freq' in self.extra_args:
            self.shift_freq = self.extra_args['shift_freq']
        else:
            self.shift_freq = False


        if 'shift_freq_q' in self.extra_args:
            self.shift_freq_q = self.extra_args['shift_freq_q']
        else:
            self.shift_freq_q = False

        if 'add_freq' in self.extra_args:
            self.add_freq = self.extra_args['add_freq']
        else:
            self.add_freq=0

        if 'qubit_shift_ge' in self.extra_args:
            self.qubit_shift_ge = self.extra_args['qubit_shift_ge']
        else:
            self.qubit_shift_ge = self.expt_cfg['qubit_shift_ge']

        if 'subexp' in self.extra_args:
            self.subexp = self.extra_args['subexp']
        else:
            self.subexp = 0

        if 'qubit_shift_ef' in self.extra_args:
            self.qubit_shift_ef = self.extra_args['qubit_shift_ef']
        else:
            self.qubit_shift_ef = self.expt_cfg['qubit_shift_ef']




        print "Target id: " +str(self.id)

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)
        self.offset_phase = self.pulse_cfg['gauss']['offset_phase']

        if self.flux_pulse_cfg['fix_phase']:
            self.phase_freq = self.expt_cfg['ramsey_freq']
        else:
            self.phase_freq = self.multimode_cfg[int(self.id)]['dc_offset_freq'] + self.expt_cfg['ramsey_freq']


    def define_pulses(self,pt):


        if self.exp==0:

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','half_pi', self.pulse_type)
            if self.expt_cfg['echo']:
                self.psb.idle(pt/2.0)
                self.psb.append('q','pi', self.pulse_type)
                self.psb.idle(pt/2.0)
            else:
                self.psb.idle(pt)
            self.psb.append('q','half_pi', self.pulse_type, phase = self.offset_phase + 360.0*self.expt_cfg['ramsey_freq']*pt/(1.0e9))

        elif self.exp==3:

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q','half_pi', self.pulse_type, phase = pt + self.offset_phase)

        elif self.exp==4:

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q','half_pi', self.pulse_type, phase = pt,freq=self.pulse_cfg['gauss']['iq_freq'] + self.multimode_cfg[self.id]['shift'])

        elif self.exp==5:

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id2),'pi_ge')

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.append('q,mm'+str(self.id),'pi_ge',phase=pt)
            self.psb.append('q','half_pi', self.pulse_type, phase=self.offset_phase)


        elif self.exp==6:

            # Multimode Rabi with photon loaded in another mode

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(int(self.id2)),'pi_ge')
                if self.shift_freq:
                    self.freq_shift = -self.multimode_cfg[self.id2]['shift'] + self.add_freq
                    if self.shift_freq_q:
                        self.freq_shift_q = +self.multimode_cfg[self.id2]['shift']
                    else:
                        self.freq_shift_q = 0
                else:
                    self.freq_shift = 0
                    self.freq_shift_q=0
            else:
                self.freq_shift=0
                self.freq_shift_q=0

            self.psb.append('q','pi', self.pulse_type,add_freq=self.freq_shift_q)
            self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length=pt, add_freq=self.freq_shift)

        elif self.exp==7:

            # Qubit Rabi with photon loaded in a mode
            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(int(self.id)),'pi_ge')

                if self.shift_freq:
                    self.freq_shift_q = +self.multimode_cfg[self.id]['shift']
                else:
                    self.freq_shift_q=0
            else:
                self.freq_shift_q=0

            self.psb.append('q','general', self.pulse_type,amp=self.pulse_cfg['gauss']['pi_a'],length=pt,freq = self.pulse_cfg['gauss']['iq_freq'],add_freq=self.freq_shift_q)

        elif self.exp==8:

            # Qubit Rabi with photon loaded in  a multimode: sweep frequency of qubit drive
            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(int(self.id)),'pi_ge')

            self.psb.append('q','general', self.pulse_type,amp=self.pulse_cfg['gauss']['pi_a'],length=pt,freq = self.pulse_cfg['gauss']['iq_freq'],add_freq=self.add_freq)

        elif self.exp==9:

            # Multimode Ramsey for the readout resonator with a photon loaded in a mode

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(int(self.id)),'pi_ge')

                if self.shift_freq:
                    self.freq_shift_q = -self.multimode_cfg[self.id]['shift']/2.0
                else:
                    self.freq_shift_q=0
            else:
                self.freq_shift_q=0

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[int(self.id2)]['pi_pi_offset_phase']/(2.0)-self.offset_phase)
            self.psb.idle(pt)
            self.psb.append('q,mm'+str(self.id2),'pi_ge', phase = 360.0*self.phase_freq*pt/(1.0e9)-self.multimode_cfg[int(self.id2)]['pi_pi_offset_phase']/(2.0)+180.0)
            self.psb.append('q','half_pi', self.pulse_type)

        elif self.exp==10:

            # Testing echo sideband pulse

            modelist = array([0,1,3,4,5,6,7,9,10])

            modelist = delete(modelist,self.id)
            self.id2 = modelist[pt/2]


            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(int(self.id2)),'pi_ge')
                if self.shift_freq:
                    self.freq_shift = -self.multimode_cfg[self.id2]['shift'] + self.add_freq
                    if self.shift_freq_q:
                        self.freq_shift_q = +self.multimode_cfg[self.id2]['shift']
                    else:
                        self.freq_shift_q = 0
                else:
                    self.freq_shift = 0
                    self.freq_shift_q=0
            else:
                self.freq_shift=0
                self.freq_shift_q=0

            half_pi_length = self.multimode_cfg[int(self.id)]['flux_pi_length'] -  (self.multimode_cfg[int(self.id)]['flux_2pi_length']-self.multimode_cfg[int(self.id)]['flux_pi_length'])/2.0
            print half_pi_length



            self.psb.append('q','pi', self.pulse_type,add_freq=self.freq_shift_q)
            if pt%2 == 1:
                self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length= half_pi_length)
                self.psb.append('q,mm'+str(int(self.id)),'pi_ge',phase=25.1)
                self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length= half_pi_length,phase = 106.69)
            else:
                self.psb.append('q,mm'+str(int(self.id)),'pi_ge')


        elif self.exp==11:

            # Calibrating echo sideband pulse



            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(int(self.id2)),'pi_ge')
                if self.shift_freq:
                    self.freq_shift = -self.multimode_cfg[self.id2]['shift'] + self.add_freq
                    if self.shift_freq_q:
                        self.freq_shift_q = +self.multimode_cfg[self.id2]['shift']
                    else:
                        self.freq_shift_q = 0
                else:
                    self.freq_shift = 0
                    self.freq_shift_q=0
            else:
                self.freq_shift=0
                self.freq_shift_q=0

            half_pi_length = self.multimode_cfg[int(self.id)]['flux_pi_length'] -  (self.multimode_cfg[int(self.id)]['flux_2pi_length']-self.multimode_cfg[int(self.id)]['flux_pi_length'])/2.0

            self.psb.append('q','pi', self.pulse_type,add_freq=self.freq_shift_q)

            self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length= half_pi_length)
            self.psb.append('q,mm'+str(int(self.id)),'pi_ge',phase=pt)
            self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length= half_pi_length,phase = 106.69)


        elif self.exp==12:

            # Calibrating echo qubit pi pulse...

            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(int(self.id2)),'pi_ge')
                if self.shift_freq:
                    self.freq_shift = -self.multimode_cfg[self.id2]['shift'] + self.add_freq
                    if self.shift_freq_q:
                        self.freq_shift_q = +self.multimode_cfg[self.id2]['shift']
                    else:
                        self.freq_shift_q = 0
                else:
                    self.freq_shift = 0
                    self.freq_shift_q=0
            else:
                self.freq_shift=0
                self.freq_shift_q=0


            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q','pi', self.pulse_type,phase=25.1)
            self.psb.append('q','half_pi', self.pulse_type,phase=self.offset_phase)


        elif self.exp == 13:

            # Testing echo qubit pulse

            modelist = array([0,1,3,4,5,6,7,9,10])

            self.id2 = modelist[pt/2]


            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(int(self.id2)),'pi_ge')
                if self.shift_freq:
                    self.freq_shift = -self.multimode_cfg[self.id2]['shift'] + self.add_freq
                    if self.shift_freq_q:
                        self.freq_shift_q = +self.multimode_cfg[self.id2]['shift']
                    else:
                        self.freq_shift_q = 0
                else:
                    self.freq_shift = 0
                    self.freq_shift_q=0
            else:
                self.freq_shift=0
                self.freq_shift_q=0

            if pt%2 == 1:
                self.psb.append('q','half_pi', self.pulse_type)
                # self.psb.append('q','pi', self.pulse_type,phase=-64.85)
                # self.psb.append('q','half_pi', self.pulse_type,phase=self.offset_phase)
            else:
                self.psb.append('q','pi', self.pulse_type)


        elif self.exp == 14:

            modelist = array([0,1,3,4,5,6,7,9,10])

            self.id2 = modelist[pt/3]
            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(int(self.id2)),'pi_ge')
                if self.shift_freq:
                    self.freq_shift = -self.multimode_cfg[self.id2]['shift'] + self.add_freq
                    if self.shift_freq_q:
                        self.freq_shift_q = +self.multimode_cfg[self.id2]['shift']
                    else:
                        self.freq_shift_q = 0
                else:
                    self.freq_shift = 0
                    self.freq_shift_q=0
            else:
                self.freq_shift=0
                self.freq_shift_q=0

            if pt%3 == 0:
                self.psb.idle(10)
            elif pt%3 == 1:
                self.psb.append('q','half_pi', self.pulse_type)
            else:
                self.psb.append('q','pi', self.pulse_type)



        elif self.exp == 15:

            modelist = array([0,1,3,4,5,6,7,9,10])

            self.psb.append('q,mm'+str(10),'pi_ge')

            if pt<27:
                self.id2 = modelist[pt/3]
                if self.load_photon:
                    self.psb.append('q','pi', self.pulse_type)
                    self.psb.append('q,mm'+str(int(self.id2)),'pi_ge')
                    if self.shift_freq:
                        self.freq_shift = -self.multimode_cfg[self.id2]['shift'] + self.add_freq
                        if self.shift_freq_q:
                            self.freq_shift_q = +self.multimode_cfg[self.id2]['shift']
                        else:
                            self.freq_shift_q = 0
                    else:
                        self.freq_shift = 0
                        self.freq_shift_q=0
                else:
                    self.freq_shift=0
                    self.freq_shift_q=0

                if pt%3 == 0:
                    self.psb.idle(10)
                elif pt%3 == 1:
                    self.psb.append('q','half_pi', self.pulse_type)
                else:
                    self.psb.append('q','pi', self.pulse_type)

            elif pt == 27:
                self.psb.idle(10)
            else:
                self.psb.append('q','pi', self.pulse_type)

### Protocol 1: Minimal number of gates in proc tom sequence. Not appropriate for extracting gate fidelities
class MultimodeProcessTomographyPhaseSweepSequence_1(QubitPulseSequence):


    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.cfg = cfg
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.proc_tom = cfg['proc_tom_phases']
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)

        if 'sb_cool' in self.extra_args:
            sb_cool = self.extra_args['sb_cool']
        else:
            sb_cool = False

        if 'pair_index' in self.extra_args:
            self.pair_index = self.extra_args['pair_index']
        else:
            self.pair_index = 0
        print "Pair index= " + str(self.pair_index)


        self.tomography_num = self.extra_args['tomography_num']
        self.state_num = self.extra_args['state_num']
        self.ef_phase_0 = 0
        self.ef_phase_1 = 0

        if 'sweep_time' in self.extra_args:
            self.sweep_time = self.extra_args['sweep_time']
        else:
            self.sweep_time = 0

        if 'phase_correct_cz' in self.extra_args:
            print 'phase_correct_cz: ' + str(self.extra_args['phase_correct_cz'])
            if self.extra_args['phase_correct_cz']:
                self.ef_phase_0 = self.proc_tom[self.pair_index]['ef_phase_0'][self.state_num][self.tomography_num]
                print "ef phase in first ef sideband of CPhase: " + str(self.ef_phase_0)

        if 'phase_correct_cnot' in self.extra_args:
            print 'phase_correct_cnot: ' + str(self.extra_args['phase_correct_cnot'])
            self.phase_correct_cnot = self.extra_args['phase_correct_cnot']
        else:
            self.phase_correct_cnot = False

        if self.phase_correct_cnot:
            self.ef_phase_1 = self.proc_tom[self.pair_index]['ef_phase_1'][self.state_num][self.tomography_num]
            print "Added (subtracted) phase in first (second) ef sideband of CPhase (CNOT): " + str(self.ef_phase_1)



        if 'sweep_cnot' in self.extra_args:
            print "CNOT phase sweep: " +str(self.extra_args['sweep_cnot'])
            self.sweep_cnot = self.extra_args['sweep_cnot']
        else:
            self.sweep_cnot = False


        if 'cnot_ef_qubit_phase' in self.extra_args:
            self.pi_ef_qubit_phase = self.extra_args['cnot_ef_qubit_phase']
            print "Phase ef pi pulse in CNOT gate: " + str(self.pi_ef_qubit_phase) + " deg"
        else:
            self.pi_ef_qubit_phase = 0

        if 'sweep_final_sb' in self.extra_args:
            self.sweep_final_sb = self.extra_args['sweep_final_sb']
            print "Sweeping final sideband: " + str(self.sweep_final_sb)
        else:
            self.sweep_final_sb = False

        if 'ef_sb_offset' in self.extra_args:
            self.ef_sb_offset = self.extra_args['ef_sb_offset']
            print "Additional offset to CZ/CNOT ef sidebands: " + str(self.ef_sb_offset)
        else:
            self.ef_sb_offset = 0

        if 'sweep_ef_qubit_phase' in self.extra_args:
            self.sweep_ef_qubit_phase = self.extra_args['sweep_ef_qubit_phase']
            print "Sweeping ef qubit phase: " + str(self.sweep_ef_qubit_phase)
            self.ef_phase_1 += self.ef_sb_offset
        else:
            self.sweep_ef_qubit_phase = False






        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses,sb_cool = sb_cool)


    def define_points(self):
        ## we define
        ## automauted
        self.tomography_pulse_num = 15
        self.expt_pts = np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']

        if 'id1' in self.extra_args:
            self.id1 = self.extra_args['id1']
        else:
            self.id1 = self.expt_cfg['id1']

        if 'id2' in self.extra_args:
            self.id2 = self.extra_args['id2']
        else:
            self.id2 = self.expt_cfg['id2']

        if 'gate_num' in self.extra_args:
            self.gate_num = self.extra_args['gate_num']
        else:
            self.gate_num = 0


        self.id = self.expt_cfg['id']

        self.offset_phase = self.pulse_cfg[self.pulse_type]['offset_phase']
        self.halfpicounter1 = 0
        self.halfpicounter2 = 0

        self.mode_mode_cnot_dc_phase = self.cfg['mode_mode_offset']['cnot_dc_phase']
        self.mode_mode_cnot_phase = self.cfg['mode_mode_offset']['cnot_phase']
        self.mode_mode_cnot_phase2 = self.cfg['mode_mode_offset']['cnot_phase2']

        self.mode_mode_cz_dc_phase = self.cfg['mode_mode_offset']['cz_dc_phase']
        self.mode_mode_cz_phase = self.cfg['mode_mode_offset']['cz_phase']
        self.mode_mode_cz_phase2 = self.cfg['mode_mode_offset']['cz_phase2']

        self.cz_phase_cz = self.mode_mode_cz_phase[self.id1][self.id2]
        self.cz_phase2_cz = self.mode_mode_cz_phase2[self.id1][self.id2]

        self.cz_phase_zc = self.mode_mode_cz_phase[self.id2][self.id1]
        self.cz_phase2_zc = self.mode_mode_cz_phase2[self.id2][self.id1]

        self.cnot_phase_cx = self.mode_mode_cnot_phase[self.id1][self.id2]
        self.cnot_phase2_cx = self.mode_mode_cnot_phase2[self.id1][self.id2]

        self.cnot_phase_xc = self.mode_mode_cnot_phase[self.id2][self.id1]
        self.cnot_phase2_xc = self.mode_mode_cnot_phase2[self.id2][self.id1]
        self.tom_pulse_list = ['0','half_pi_y','half_pi','pi']
        self.pulse_num_id1 = self.state_num%4
        self.pulse_num_id2 = self.state_num/4
        # print "Phase added to first ef sideband of CPhase = " +str(self.ef_phase_0)
        # print "Additional Phase added (subtracted) to/from first (second) ef sideband of CPhase (CNOT) = " +str(self.ef_phase_1)

    def define_pulses(self,pt):

        ### Initiate states
        self.define_states(pt)
        ### Idle
        if 'sweep_time' in self.extra_args:
            self.psb.idle(self.sweep_time)
        else:
            pass
        ### Act Gate
        self.define_gate(pt)
        ### Tomography
        self.define_tomography_pulse(pt)

    def define_states(self,pt):
        ### Preparing all input states required for two-mode process tomography
        if self.pulse_num_id1  == 1 or self.pulse_num_id1 == 2:
            add_phase_1 = -self.offset_phase
        else:
            add_phase_1 = 0

        if self.pulse_num_id2  == 1 or self.pulse_num_id2 == 2:
            add_phase_2 = -self.offset_phase
        else:
            add_phase_2 = 0

        self.psb.append('q',self.tom_pulse_list[self.pulse_num_id2], self.pulse_type)
        self.psb.append('q,mm'+str(self.id2),'pi_ge',phase = self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + add_phase_2 - 90.0)
        self.psb.append('q',self.tom_pulse_list[self.pulse_num_id1], self.pulse_type)
        self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + add_phase_1 - 90.0)

    def define_gate(self,pt):
        ### Preparing all input states required for two-mode process tomography
        tomo_index = (self.tomography_num)%self.tomography_pulse_num
        if self.sweep_final_sb:
            self.add_cphase = 0
            if self.phase_correct_cnot:
                if tomo_index in array([4,5,8,9]):
                    self.add_cphase = self.ef_phase_1
        elif self.sweep_ef_qubit_phase:
            self.pi_ef_qubit_phase = pt
            self.add_cphase = self.ef_phase_1
        else:
            self.add_cphase = pt


        if self.gate_num ==0:
            # I
            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.ef_phase_0+self.cz_phase_cz + 180.0 + self.add_cphase,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

        elif self.gate_num == 1:
            # CZ
            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.ef_phase_0+self.cz_phase_cz + self.add_cphase,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
        elif self.gate_num == 2:
            # CX
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=0,efsbphase_0=self.ef_phase_0+self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0+ 90.0)
        elif self.gate_num == 3:
            # CY
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=90.0,efsbphase_0=self.cnot_phase_cx + self.ef_phase_0 + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0+ 90.0)
        else:
            pass

    def define_tomography_pulse(self,pt):

        ### Correlators for two mode tomography; while sweeping the phase of the ef sideband pulse within the gate (in define gate)

        # State convention : |id2, id1 >
        # Gate convention CNOT/CZ(control_id,target_id)

        #CNOT(id2, id1) = CX
        #CZ(id2, id1) = CZ
        if self.sweep_cnot:
            self.add_cnot_phase_2 = -pt
        else:
            self.add_cnot_phase_2 = 0
        if self.sweep_final_sb:
            self.final_sb_phase = pt
        else:
            self.final_sb_phase = 0
        if self.gate_num == -1:
            add_cnot_efphase_0 = pt
        else:
            add_cnot_efphase_0 = 0

        tomo_index = (self.tomography_num)%self.tomography_pulse_num

        if tomo_index == 0:

            # -<IX>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

            # self.halfpicounter2+=1

        elif tomo_index == 1:
            # <IY>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)

            # self.halfpicounter2+=1

        elif tomo_index == 2:
            # <IZ>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0+ self.final_sb_phase)

        elif tomo_index == 3:
            # -<XI>
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)
            # self.halfpicounter2+=1
        elif tomo_index == 4:
            # <XX> = XI + CX

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx + add_cnot_efphase_0,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0+ 90.0,efsbphase_2=self.add_cnot_phase_2-self.ef_phase_1)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 5:
            # -<XY> = XI + CY

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase= -90.0 + self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx + add_cnot_efphase_0,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2-self.ef_phase_1)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)


        elif tomo_index == 6:
            # <XZ>

            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.cz_phase_cz,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 7:
            # <YI>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)

        elif tomo_index == 8:
            # -<YX>

            #CNOT
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx+add_cnot_efphase_0,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2-self.ef_phase_1)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 9:
            # <YY>

            #CY
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=-90.0+self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx + add_cnot_efphase_0,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2-self.ef_phase_1)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 10:
            # -<YZ>

            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.cz_phase_cz,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0+ 90.0)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)

#### Was wrong untill 02/13

        elif tomo_index == 11:
            # <ZI>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)

        elif tomo_index == 12:
            # <ZX>

            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.cz_phase_cz,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 13:
            # <ZY>

            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.cz_phase_cz,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 14:
            # <ZZ>

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2+self.ef_phase_1)
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)

### Protocol 2: All the measurement segments have a gate. Appropriate for extracting gate fidelities, with the insetion of the gate of interest between prep and measurement
class MultimodeProcessTomographyPhaseSweepSequence_2(QubitPulseSequence):


    def __init__(self,name, cfg, expt_cfg,**kwargs):

        self.qubit_cfg = cfg['qubit']
        self.cfg = cfg
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.proc_tom_phase_matrix = load("S:\\_Data\\160912 - 2D Multimode Qubit (Chip MM3, 11 modes)\\process_tomography_correlations.npy")
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)

        if 'sb_cool' in self.extra_args:
            sb_cool = self.extra_args['sb_cool']
        else:
            sb_cool = False

        if 'id1' in self.extra_args:
            self.id1 = self.extra_args['id1']
        else:
            self.id1 = self.expt_cfg['id1']

        if 'id2' in self.extra_args:
            self.id2 = self.extra_args['id2']
        else:
            self.id2 = self.expt_cfg['id2']

        self.proc_tom =   self.proc_tom_phase_matrix[self.id2][self.id1]
        self.tomography_num = self.extra_args['tomography_num']
        self.state_num = self.extra_args['state_num']
        self.ef_phase_0 = 0
        self.ef_phase_1 = 0

        if 'sweep_time' in self.extra_args:
            self.sweep_time = self.extra_args['sweep_time']
        else:
            self.sweep_time = 0

        if 'phase_correct_cz' in self.extra_args:
            self.phase_correct_cz = self.extra_args['phase_correct_cz']
        else:
            self.phase_correct_cz = False
        print 'phase_correct_cz: ' + str(self.phase_correct_cz)

        if self.phase_correct_cz:
            self.ef_phase_0 = self.proc_tom[0][self.state_num][self.tomography_num]
            print "ef phase in first ef sideband of CPhase: " + str(self.ef_phase_0)

        if 'phase_correct_cnot' in self.extra_args:
            self.phase_correct_cnot = self.extra_args['phase_correct_cnot']
        else:
            self.phase_correct_cnot = False

        print 'phase_correct_cnot: ' + str(self.phase_correct_cnot)

        if self.phase_correct_cnot:
            self.ef_phase_1 =  self.proc_tom[1][self.state_num][self.tomography_num]
            print "Added (subtracted) phase in first (second) ef sideband of CPhase (CNOT): " + str(self.ef_phase_1)

        if 'cnot_ef_qubit_phase' in self.extra_args:
            self.pi_ef_qubit_phase = self.extra_args['cnot_ef_qubit_phase']
            print "Phase ef pi pulse in CNOT gate: " + str(self.pi_ef_qubit_phase) + " deg"
        else:
            self.pi_ef_qubit_phase = 0

        if 'use_saved_cnot_ef_qubit_phase' in self.extra_args:
            self.use_saved_cnot_ef_qubit_phase = self.extra_args['use_saved_cnot_ef_qubit_phase']
        else:
            self.use_saved_cnot_ef_qubit_phase = False

        if self.use_saved_cnot_ef_qubit_phase:
            self.pi_ef_qubit_phase = self.proc_tom[2][6][4]
            # self.pi_ef_qubit_phase = 0

        if 'sweep_cnot' in self.extra_args:
            print "CNOT phase sweep: " +str(self.extra_args['sweep_cnot'])
            self.sweep_cnot = self.extra_args['sweep_cnot']
        else:
            self.sweep_cnot = False


        if 'sweep_final_sb' in self.extra_args:
            self.sweep_final_sb = self.extra_args['sweep_final_sb']
            print "Sweeping final sideband: " + str(self.sweep_final_sb)
        else:
            self.sweep_final_sb = False

        if 'ef_sb_offset' in self.extra_args:
            self.ef_sb_offset = self.extra_args['ef_sb_offset']
            print "Additional offset to CZ/CNOT ef sidebands: " + str(self.ef_sb_offset)
        else:
            self.ef_sb_offset = 0

        if 'sweep_ef_qubit_phase' in self.extra_args:
            self.sweep_ef_qubit_phase = self.extra_args['sweep_ef_qubit_phase']
            print "Sweeping ef qubit phase: " + str(self.sweep_ef_qubit_phase)
            # self.ef_phase_1 += self.ef_sb_offset
        else:
            self.sweep_ef_qubit_phase = False

        if 'sweep_ef_sb_offset_phase' in self.extra_args:
            self.sweep_ef_sb_offset_phase = self.extra_args['sweep_ef_sb_offset_phase']
            print "Sweeping ef sb offset phase: " + str(self.sweep_ef_sb_offset_phase)
            # self.ef_phase_1 += self.ef_sb_offset
        else:
            self.sweep_ef_sb_offset_phase = False


        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses,sb_cool = sb_cool)

    def define_points(self):
        ## we define
        ## automauted
        self.tomography_pulse_num = 15

        self.expt_pts = np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']


        if 'gate_num' in self.extra_args:
            self.gate_num = self.extra_args['gate_num']
        else:
            self.gate_num = 0


        self.id = self.expt_cfg['id']

        self.offset_phase = self.pulse_cfg[self.pulse_type]['offset_phase']
        self.halfpicounter1 = 0
        self.halfpicounter2 = 0

        self.mode_mode_cnot_dc_phase = self.cfg['mode_mode_offset']['cnot_dc_phase']
        self.mode_mode_cnot_phase = self.cfg['mode_mode_offset']['cnot_phase']
        self.mode_mode_cnot_phase2 = self.cfg['mode_mode_offset']['cnot_phase2']

        self.mode_mode_cz_dc_phase = self.cfg['mode_mode_offset']['cz_dc_phase']
        self.mode_mode_cz_phase = self.cfg['mode_mode_offset']['cz_phase']
        self.mode_mode_cz_phase2 = self.cfg['mode_mode_offset']['cz_phase2']

        self.cz_phase_cz = self.mode_mode_cz_phase[self.id1][self.id2]
        self.cz_phase2_cz = self.mode_mode_cz_phase2[self.id1][self.id2]

        self.cz_phase_zc = self.mode_mode_cz_phase[self.id2][self.id1]
        self.cz_phase2_zc = self.mode_mode_cz_phase2[self.id2][self.id1]

        self.cnot_phase_cx = self.mode_mode_cnot_phase[self.id1][self.id2]
        self.cnot_phase2_cx = self.mode_mode_cnot_phase2[self.id1][self.id2]

        self.cnot_phase_xc = self.mode_mode_cnot_phase[self.id2][self.id1]
        self.cnot_phase2_xc = self.mode_mode_cnot_phase2[self.id2][self.id1]
        self.tom_pulse_list = ['0','half_pi_y','half_pi','pi']
        self.pulse_num_id1 = self.state_num%4
        self.pulse_num_id2 = self.state_num/4
        # print "Phase added to first ef sideband of CPhase = " +str(self.ef_phase_0)
        # print "Additional Phase added (subtracted) to/from first (second) ef sideband of CPhase (CNOT) = " +str(self.ef_phase_1)

    def define_pulses(self,pt):

        ### Initiate states
        self.define_states(pt)
        ### Idle
        if 'sweep_time' in self.extra_args:
            self.psb.idle(self.sweep_time)
        else:
            pass
        ### Act Gate
        self.define_gate(pt)
        ### Tomography
        self.define_tomography_pulse(pt)

    def define_states(self,pt):
        ### Preparing all input states required for two-mode process tomography
        if self.pulse_num_id1  == 1 or self.pulse_num_id1 == 2:
            add_phase_1 = -self.offset_phase
        else:
            add_phase_1 = 0

        if self.pulse_num_id2  == 1 or self.pulse_num_id2 == 2:
            add_phase_2 = -self.offset_phase
        else:
            add_phase_2 = 0

        self.psb.append('q',self.tom_pulse_list[self.pulse_num_id2], self.pulse_type)
        self.psb.append('q,mm'+str(self.id2),'pi_ge',phase = self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + add_phase_2 - 90.0)
        self.psb.append('q',self.tom_pulse_list[self.pulse_num_id1], self.pulse_type)
        self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + add_phase_1 - 90.0)

    def define_gate(self,pt):
        ### Preparing all input states required for two-mode process tomography
        tomo_index = (self.tomography_num)%self.tomography_pulse_num

        if self.gate_num ==0:
            # I
            pass

        elif self.gate_num == 1:
            # CZ
            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.cz_phase_cz,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
        elif self.gate_num == 2:
            # CX
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=0,efsbphase_0=self.cnot_phase_cx ,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0)
        elif self.gate_num == 3:
            # CY
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=-90.0,efsbphase_0=self.cnot_phase_cx + self.ef_phase_0 ,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0+ 90.0)
        else:
            pass

    def define_tomography_pulse(self,pt):

        ### Correlators for two mode tomography; while sweeping the phase of the ef sideband pulse within the gate (in define gate)

        # State convention : |id2, id1 >
        # Gate convention CNOT/CZ(control_id,target_id)

        tomo_index = (self.tomography_num)%self.tomography_pulse_num
        #CNOT(id2, id1) = CX
        #CZ(id2, id1) = CZ


        #### Adds ef phase 0 to first ef sideband of CNOT gate

        self.add_cphase = 0
        self.final_sb_phase = 0
        self.add_cnot_phase_2 = 0

        if self.phase_correct_cz:
            self.add_cphase += self.ef_phase_0
        if tomo_index in array([4,5,8,9]):
            self.add_cphase += self.ef_sb_offset
            self.add_cnot_phase_2 -= self.ef_sb_offset
            if self.phase_correct_cnot:
                self.add_cphase += self.ef_phase_1
                self.add_cnot_phase_2 -= self.ef_phase_1

        if self.sweep_final_sb:
            self.final_sb_phase += pt
        elif self.sweep_ef_qubit_phase:
            self.pi_ef_qubit_phase = pt
        else:
            ### pt added to Cphase for both CZ and CNOT sweeps
            self.add_cphase += pt

        if self.sweep_cnot:
            self.add_cnot_phase_2 -=pt
        elif self.sweep_ef_sb_offset_phase:
            self.add_cnot_phase_2 +=pt
        else:
            pass




        ### Adds a CPhase gate to all correlators that do not involve a CNOT gate

        if tomo_index not in array([4,5,8,9,14]):
            ### i.e not in XX, XY, YX, YY, ZZ
            if tomo_index in array([6,10,12,13]):
                ### i.e is XZ,YZ,ZX, ZY
                id_phase = 0.0
            else:
                id_phase = 180.0
            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.cz_phase_cz +  id_phase + self.add_cphase,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
        else:
            ### Gate added for CX correlators later
            pass


        if tomo_index == 0:

            # -<IX>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

            # self.halfpicounter2+=1

        elif tomo_index == 1:
            # <IY>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)

            # self.halfpicounter2+=1

        elif tomo_index == 2:
            # <IZ>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0+ self.final_sb_phase)

        elif tomo_index == 3:
            # -<XI>
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)
            # self.halfpicounter2+=1
        elif tomo_index == 4:
            # <XX> = XI + CX

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

            print "ef qubit phase = " +str(self.pi_ef_qubit_phase)
            print "add c phase = " +str(self.add_cphase)
            print "add cnot phase 2 = " +str(self.add_cnot_phase_2)
            print "cnot_phase_2_cx = " +str(self.cnot_phase2_cx)
            print "ef phase 1 = " +str(self.ef_phase_1)
            print "ef phase 0 = "+str(self.ef_phase_0)


        elif tomo_index == 5:
            # -<XY> = XI + CY

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase= -90.0 + self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)


        elif tomo_index == 6:
            # <XZ>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 7:
            # <YI>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)

        elif tomo_index == 8:
            # -<YX>

            #CNOT
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx+self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 9:
            # <YY>

            #CY
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=-90.0+self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 10:
            # -<YZ>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 11:
            # <ZI>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)

        elif tomo_index == 12:
            # <ZX>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 13:
            # <ZY>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 14:
            # <ZZ>

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)

### Protocol 3: valid for any gate
class MultimodeProcessTomographyPhaseSweepSequence_3(QubitPulseSequence):


    def __init__(self,name, cfg, expt_cfg,**kwargs):

        self.qubit_cfg = cfg['qubit']
        self.cfg = cfg
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.proc_tom_phase_matrix = load("S:\\_Data\\160912 - 2D Multimode Qubit (Chip MM3, 11 modes)\\process_tomography_correlations.npy")
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)

        if 'sb_cool' in self.extra_args:
            sb_cool = self.extra_args['sb_cool']
        else:
            sb_cool = False

        if 'id1' in self.extra_args:
            self.id1 = self.extra_args['id1']
        else:
            self.id1 = self.expt_cfg['id1']

        if 'id2' in self.extra_args:
            self.id2 = self.extra_args['id2']
        else:
            self.id2 = self.expt_cfg['id2']

        self.proc_tom =   self.proc_tom_phase_matrix[self.id2][self.id1]
        self.tomography_num = self.extra_args['tomography_num']
        self.state_num = self.extra_args['state_num']
        self.ef_phase_0 = 0
        self.ef_phase_1 = 0

        if 'sweep_time' in self.extra_args:
            self.sweep_time = self.extra_args['sweep_time']
        else:
            self.sweep_time = 0

        if 'phase_correct_cz' in self.extra_args:
            self.phase_correct_cz = self.extra_args['phase_correct_cz']
        else:
            self.phase_correct_cz = False
        print 'phase_correct_cz: ' + str(self.phase_correct_cz)

        if self.phase_correct_cz:
            self.ef_phase_0 = self.proc_tom[0][self.state_num][self.tomography_num]
            print "ef phase in first ef sideband of CPhase: " + str(self.ef_phase_0)

        if 'phase_correct_cnot' in self.extra_args:
            self.phase_correct_cnot = self.extra_args['phase_correct_cnot']
        else:
            self.phase_correct_cnot = False

        print 'phase_correct_cnot: ' + str(self.phase_correct_cnot)

        if self.phase_correct_cnot:
            self.ef_phase_1 =  self.proc_tom[1][self.state_num][self.tomography_num]
            print "Added (subtracted) phase in first (second) ef sideband of CPhase (CNOT): " + str(self.ef_phase_1)

        if 'cnot_ef_qubit_phase' in self.extra_args:
            self.pi_ef_qubit_phase = self.extra_args['cnot_ef_qubit_phase']
            print "Phase ef pi pulse in CNOT gate: " + str(self.pi_ef_qubit_phase) + " deg"
        else:
            self.pi_ef_qubit_phase = 0

        if 'use_saved_cnot_ef_qubit_phase' in self.extra_args:
            self.use_saved_cnot_ef_qubit_phase = self.extra_args['use_saved_cnot_ef_qubit_phase']
        else:
            self.use_saved_cnot_ef_qubit_phase = False

        if self.use_saved_cnot_ef_qubit_phase:
            self.pi_ef_qubit_phase = self.proc_tom[2][6][4]
            # self.pi_ef_qubit_phase = 0

        if 'sweep_cnot' in self.extra_args:
            print "CNOT phase sweep: " +str(self.extra_args['sweep_cnot'])
            self.sweep_cnot = self.extra_args['sweep_cnot']
        else:
            self.sweep_cnot = False


        if 'sweep_final_sb' in self.extra_args:
            self.sweep_final_sb = self.extra_args['sweep_final_sb']
            print "Sweeping final sideband: " + str(self.sweep_final_sb)
        else:
            self.sweep_final_sb = False

        if 'ef_sb_offset' in self.extra_args:
            self.ef_sb_offset = self.extra_args['ef_sb_offset']
            print "Additional offset to CZ/CNOT ef sidebands: " + str(self.ef_sb_offset)
        else:
            self.ef_sb_offset = 0

        if 'sweep_ef_qubit_phase' in self.extra_args:
            self.sweep_ef_qubit_phase = self.extra_args['sweep_ef_qubit_phase']
            print "Sweeping ef qubit phase: " + str(self.sweep_ef_qubit_phase)
            # self.ef_phase_1 += self.ef_sb_offset
        else:
            self.sweep_ef_qubit_phase = False

        if 'sweep_ef_sb_offset_phase' in self.extra_args:
            self.sweep_ef_sb_offset_phase = self.extra_args['sweep_ef_sb_offset_phase']
            print "Sweeping ef sb offset phase: " + str(self.sweep_ef_sb_offset_phase)
            # self.ef_phase_1 += self.ef_sb_offset
        else:
            self.sweep_ef_sb_offset_phase = False


        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses,sb_cool = sb_cool)

    def define_points(self):
        ## we define
        ## automauted
        self.tomography_pulse_num = 15

        self.expt_pts = np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']


        if 'gate_num' in self.extra_args:
            self.gate_num = self.extra_args['gate_num']
        else:
            self.gate_num = 0


        self.id = self.expt_cfg['id']
        self.gate_phase = 0

        self.offset_phase = self.pulse_cfg[self.pulse_type]['offset_phase']
        self.halfpicounter1 = 0
        self.halfpicounter2 = 0

        self.mode_mode_cnot_dc_phase = self.cfg['mode_mode_offset']['cnot_dc_phase']
        self.mode_mode_cnot_phase = self.cfg['mode_mode_offset']['cnot_phase']
        self.mode_mode_cnot_phase2 = self.cfg['mode_mode_offset']['cnot_phase2']

        self.mode_mode_cz_dc_phase = self.cfg['mode_mode_offset']['cz_dc_phase']
        self.mode_mode_cz_phase = self.cfg['mode_mode_offset']['cz_phase']
        self.mode_mode_cz_phase2 = self.cfg['mode_mode_offset']['cz_phase2']

        self.cz_phase_cz = self.mode_mode_cz_phase[self.id1][self.id2]
        self.cz_phase2_cz = self.mode_mode_cz_phase2[self.id1][self.id2]

        self.cz_phase_zc = self.mode_mode_cz_phase[self.id2][self.id1]
        self.cz_phase2_zc = self.mode_mode_cz_phase2[self.id2][self.id1]

        self.cnot_phase_cx = self.mode_mode_cnot_phase[self.id1][self.id2]
        self.cnot_phase2_cx = self.mode_mode_cnot_phase2[self.id1][self.id2]

        self.cnot_phase_xc = self.mode_mode_cnot_phase[self.id2][self.id1]
        self.cnot_phase2_xc = self.mode_mode_cnot_phase2[self.id2][self.id1]
        self.tom_pulse_list = ['0','half_pi_y','half_pi','pi']
        self.pulse_num_id1 = self.state_num%4
        self.pulse_num_id2 = self.state_num/4
        # print "Phase added to first ef sideband of CPhase = " +str(self.ef_phase_0)
        # print "Additional Phase added (subtracted) to/from first (second) ef sideband of CPhase (CNOT) = " +str(self.ef_phase_1)

    def define_pulses(self,pt):

        ### Initiate states
        self.define_states(pt)
        ### Idle
        if 'sweep_time' in self.extra_args:
            self.psb.idle(self.sweep_time)
        else:
            pass
        ### Act Gate
        self.define_gate(pt)
        ### Tomography
        self.define_tomography_pulse(pt)

    def define_states(self,pt):
        ### Preparing all input states required for two-mode process tomography
        if self.pulse_num_id1  == 1 or self.pulse_num_id1 == 2:
            add_phase_1 = -self.offset_phase
        else:
            add_phase_1 = 0

        if self.pulse_num_id2  == 1 or self.pulse_num_id2 == 2:
            add_phase_2 = -self.offset_phase
        else:
            add_phase_2 = 0

        self.psb.append('q',self.tom_pulse_list[self.pulse_num_id2], self.pulse_type)
        self.psb.append('q,mm'+str(self.id2),'pi_ge',phase = self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + add_phase_2 - 90.0)
        self.psb.append('q',self.tom_pulse_list[self.pulse_num_id1], self.pulse_type)
        self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + add_phase_1 - 90.0)

    def define_gate(self,pt):
        ### Preparing all input states required for two-mode process tomography
        tomo_index = (self.tomography_num)%self.tomography_pulse_num
        self.add_cphase_g = 0
        if self.phase_correct_cz:
            if self.gate_num is not 0:
                self.add_cphase_g += self.ef_phase_0
        if tomo_index in array([4,5,8,9]):
            if self.phase_correct_cnot:
                if self.gate_num is not 0:
                    self.add_cphase_g += self.ef_phase_1

        if self.gate_num ==0:
            # I
            pass

        elif self.gate_num == 1:
            # CZ
            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.cz_phase_cz + self.add_cphase_g,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
        elif self.gate_num == 2:
            # CX
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=0,efsbphase_0=self.cnot_phase_cx + self.add_cphase_g,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0)
        elif self.gate_num == 3:
            # CY
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=-90.0,efsbphase_0=self.cnot_phase_cx + self.add_cphase_g ,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0+ 90.0)
        elif self.gate_num == -1:
            self.gate_phase = 180.0

            pass

    def define_tomography_pulse(self,pt):

        ### Correlators for two mode tomography; while sweeping the phase of the ef sideband pulse within the gate (in define gate)

        # State convention : |id2, id1 >
        # Gate convention CNOT/CZ(control_id,target_id)

        tomo_index = (self.tomography_num)%self.tomography_pulse_num
        #CNOT(id2, id1) = CX
        #CZ(id2, id1) = CZ


        #### Adds ef phase 0 to first ef sideband of CNOT gate

        self.add_cphase = 0
        self.final_sb_phase = 0
        self.add_cnot_phase_2 = 0

        if self.phase_correct_cz:
            if self.gate_num == 0:
                self.add_cphase += self.ef_phase_0
        if tomo_index in array([4,5,8,9]):
            self.add_cphase += self.ef_sb_offset
            self.add_cnot_phase_2 -= self.ef_sb_offset
            if self.phase_correct_cnot:
                if self.gate_num==0:
                    self.add_cphase += self.ef_phase_1
                self.add_cnot_phase_2 -= self.ef_phase_1

        if self.sweep_final_sb:
            self.final_sb_phase += pt
        elif self.sweep_ef_qubit_phase:
            self.pi_ef_qubit_phase = pt
        else:
            ### pt added to Cphase for both CZ and CNOT sweeps
            self.add_cphase += pt

        if self.sweep_cnot:
            self.add_cnot_phase_2 -=pt
        elif self.sweep_ef_sb_offset_phase:
            self.add_cnot_phase_2 +=pt
        else:
            pass




        ### Adds a CPhase gate to all correlators that do not involve a CNOT gate

        if tomo_index not in array([4,5,8,9,14]):
            ### i.e not in XX, XY, YX, YY, ZZ
            if tomo_index in array([6,10,12,13]):
                ### i.e is XZ,YZ,ZX, ZY
                id_phase = 0.0
            else:
                id_phase = 180.0
            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.gate_phase+self.cz_phase_cz +  id_phase + self.add_cphase,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
        else:
            ### Gate added for CX correlators later
            pass


        if tomo_index == 0:

            # -<IX>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

            # self.halfpicounter2+=1

        elif tomo_index == 1:
            # <IY>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)

            # self.halfpicounter2+=1

        elif tomo_index == 2:
            # <IZ>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0+ self.final_sb_phase)

        elif tomo_index == 3:
            # -<XI>
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)
            # self.halfpicounter2+=1
        elif tomo_index == 4:
            # <XX> = XI + CX

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.gate_phase+self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)



        elif tomo_index == 5:
            # -<XY> = XI + CY

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase= -90.0 + self.pi_ef_qubit_phase,efsbphase_0=self.gate_phase+self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)


        elif tomo_index == 6:
            # <XZ>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 7:
            # <YI>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)

        elif tomo_index == 8:
            # -<YX>

            #CNOT
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.gate_phase+self.cnot_phase_cx+self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 9:
            # <YY>

            #CY
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=-90.0+self.pi_ef_qubit_phase,efsbphase_0=self.gate_phase+self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 10:
            # -<YZ>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 11:
            # <ZI>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)

        elif tomo_index == 12:
            # <ZX>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 13:
            # <ZY>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 14:
            # <ZZ>

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.gate_phase+self.cnot_phase_cx,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)

### Flips the order of sidebands during state preparation so that the final prep sideband == measurement sideband
class MultimodeProcessTomographySequence_2(QubitPulseSequence):


    def __init__(self,name, cfg, expt_cfg,**kwargs):

        self.qubit_cfg = cfg['qubit']
        self.cfg = cfg
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.proc_tom_phase_matrix = load("S:\\_Data\\160912 - 2D Multimode Qubit (Chip MM3, 11 modes)\\process_tomography_correlations.npy")
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)

        if 'sb_cool' in self.extra_args:
            sb_cool = self.extra_args['sb_cool']
        else:
            sb_cool = False

        if 'proc_tom_set' in self.extra_args:
            self.proc_tom_set = kwargs['proc_tom_set']
        else:
            self.proc_tom_set = 0

        if 'id1' in self.extra_args:
            self.id1 = self.extra_args['id1']
        else:
            self.id1 = self.expt_cfg['id1']

        if 'id2' in self.extra_args:
            self.id2 = self.extra_args['id2']
        else:
            self.id2 = self.expt_cfg['id2']

        self.proc_tom =   self.proc_tom_phase_matrix[self.id2][self.id1]


        self.pi_ef_qubit_phase = self.proc_tom[2][6][4]
        # self.pi_ef_qubit_phase = self.proc_tom[2][6][4]
            # self.pi_ef_qubit_phase = 0


        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses,sb_cool = sb_cool)

    def define_points(self):
        ## we define
        ## automauted
        self.tomography_pulse_num = 15

        if self.proc_tom_set == 0:
            self.expt_pts = np.arange(120)
        else:
            self.expt_pts = np.arange(120,240)

    def define_parameters(self):

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']


        if 'gate_num' in self.extra_args:
            self.gate_num = self.extra_args['gate_num']
        else:
            self.gate_num = 0


        self.offset_phase = self.pulse_cfg[self.pulse_type]['offset_phase']


        self.mode_mode_cnot_dc_phase = self.cfg['mode_mode_offset']['cnot_dc_phase']
        self.mode_mode_cnot_phase = self.cfg['mode_mode_offset']['cnot_phase']
        self.mode_mode_cnot_phase2 = self.cfg['mode_mode_offset']['cnot_phase2']

        self.mode_mode_cz_dc_phase = self.cfg['mode_mode_offset']['cz_dc_phase']
        self.mode_mode_cz_phase = self.cfg['mode_mode_offset']['cz_phase']
        self.mode_mode_cz_phase2 = self.cfg['mode_mode_offset']['cz_phase2']

        self.cz_phase_cz = self.mode_mode_cz_phase[self.id1][self.id2]
        self.cz_phase2_cz = self.mode_mode_cz_phase2[self.id1][self.id2]

        self.cz_phase_zc = self.mode_mode_cz_phase[self.id2][self.id1]
        self.cz_phase2_zc = self.mode_mode_cz_phase2[self.id2][self.id1]

        self.cnot_phase_cx = self.mode_mode_cnot_phase[self.id1][self.id2]
        self.cnot_phase2_cx = self.mode_mode_cnot_phase2[self.id1][self.id2]

        self.cnot_phase_xc = self.mode_mode_cnot_phase[self.id2][self.id1]
        self.cnot_phase2_xc = self.mode_mode_cnot_phase2[self.id2][self.id1]
        self.tom_pulse_list = ['0','half_pi_y','half_pi','pi']

        # print "Phase added to first ef sideband of CPhase = " +str(self.ef_phase_0)
        # print "Additional Phase added (subtracted) to/from first (second) ef sideband of CPhase (CNOT) = " +str(self.ef_phase_1)

    def define_pulses(self,pt):

        ### Initiate states
        self.define_states(pt)
        ### Act Gate
        self.define_gate(pt)
        ### Tomography
        self.define_tomography_pulse(pt)

    def define_states(self,pt):
        ### Preparing all input states required for two-mode process tomography

        self.state_num = pt%16
        self.pulse_num_id1 = self.state_num%4
        self.pulse_num_id2 = self.state_num/4


        if self.pulse_num_id1  == 1 or self.pulse_num_id1 == 2:
            add_phase_1 = -self.offset_phase
        else:
            add_phase_1 = 0

        if self.pulse_num_id2  == 1 or self.pulse_num_id2 == 2:
            add_phase_2 = -self.offset_phase
        else:
            add_phase_2 = 0

        self.psb.append('q',self.tom_pulse_list[self.pulse_num_id2], self.pulse_type)
        self.psb.append('q,mm'+str(self.id2),'pi_ge',phase = self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + add_phase_2 - 90.0)
        self.psb.append('q',self.tom_pulse_list[self.pulse_num_id1], self.pulse_type)
        self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + add_phase_1 - 90.0)

    def define_gate(self,pt):
        ### Preparing all input states required for two-mode process tomography
        self.tomography_num = pt/16
        self.state_num = pt%16
        tomo_index = (self.tomography_num)%self.tomography_pulse_num
        self.ef_phase_0 = self.proc_tom[0][self.state_num][self.tomography_num]
        self.ef_phase_1 =  self.proc_tom[1][self.state_num][self.tomography_num]


        if self.gate_num ==0:
            # I
            pass

        elif self.gate_num == 1:
            # CZ
            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.cz_phase_cz,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
        elif self.gate_num == 2:
            # CX
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=0,efsbphase_0=self.cnot_phase_cx ,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0)
        elif self.gate_num == 3:
            # CY
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=-90.0,efsbphase_0=self.cnot_phase_cx + self.ef_phase_0 ,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0+ 90.0)
        else:
            pass

    def define_tomography_pulse(self,pt):

        ### Correlators for two mode tomography; while sweeping the phase of the ef sideband pulse within the gate (in define gate)

        # State convention : |id2, id1 >
        # Gate convention CNOT/CZ(control_id,target_id)

        self.tomography_num = pt/16
        self.state_num = pt%16
        tomo_index = (self.tomography_num)%self.tomography_pulse_num
        self.ef_phase_0 = self.proc_tom[0][self.state_num][self.tomography_num]
        self.ef_phase_1 =  self.proc_tom[1][self.state_num][self.tomography_num]

        #CNOT(id2, id1) = CX
        #CZ(id2, id1) = CZ


        #### Adds ef phase 0 to first ef sideband of CNOT gate

        self.add_cphase = 0
        self.final_sb_phase = 0
        self.add_cnot_phase_2 = 0

        if tomo_index in array([4,5,8,9]):
            self.add_cphase += self.ef_phase_0 + self.ef_phase_1
            self.add_cnot_phase_2 -= self.ef_phase_1


        ### Adds a CPhase gate to all correlators that do not involve a CNOT gate

        if tomo_index not in array([4,5,8,9,14]):
            ### i.e not in XX, XY, YX, YY, ZZ
            if tomo_index in array([6,10,12,13]):
                ### i.e is XZ,YZ,ZX, ZY
                id_phase = 0.0
            else:
                id_phase = 180.0
            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.ef_phase_0+self.cz_phase_cz +  id_phase + self.add_cphase,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
        else:
            ### Gate added for CX correlators later
            pass


        if tomo_index == 0:

            # -<IX>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

            # self.halfpicounter2+=1

        elif tomo_index == 1:
            # <IY>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)

            # self.halfpicounter2+=1

        elif tomo_index == 2:
            # <IZ>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0+ self.final_sb_phase)

        elif tomo_index == 3:
            # -<XI>
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)
            # self.halfpicounter2+=1
        elif tomo_index == 4:
            # <XX> = XI + CX

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

            print "ef qubit phase = " +str(self.pi_ef_qubit_phase)
            print "add c phase = " +str(self.add_cphase)
            print "add cnot phase 2 = " +str(self.add_cnot_phase_2)
            print "cnot_phase_2_cx = " +str(self.cnot_phase2_cx)
            print "ef phase 1 = " +str(self.ef_phase_1)
            print "ef phase 0 = "+str(self.ef_phase_0)


        elif tomo_index == 5:
            # -<XY> = XI + CY

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase= -90.0 + self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)


        elif tomo_index == 6:
            # <XZ>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 7:
            # <YI>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)

        elif tomo_index == 8:
            # -<YX>

            #CNOT
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx+self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 9:
            # <YY>

            #CY
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=-90.0+self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 10:
            # -<YZ>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 11:
            # <ZI>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)

        elif tomo_index == 12:
            # <ZX>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 13:
            # <ZY>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 14:
            # <ZZ>

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)

class MultimodeProcessTomographySequence_3(QubitPulseSequence):


    def __init__(self,name, cfg, expt_cfg,**kwargs):

        self.qubit_cfg = cfg['qubit']
        self.cfg = cfg
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.proc_tom_phase_matrix = load("S:\\_Data\\160912 - 2D Multimode Qubit (Chip MM3, 11 modes)\\process_tomography_correlations.npy")
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)

        if 'sb_cool' in self.extra_args:
            sb_cool = self.extra_args['sb_cool']
        else:
            sb_cool = False

        if 'proc_tom_set' in self.extra_args:
            self.proc_tom_set = kwargs['proc_tom_set']
        else:
            self.proc_tom_set = 0

        if 'id1' in self.extra_args:
            self.id1 = self.extra_args['id1']
        else:
            self.id1 = self.expt_cfg['id1']

        if 'id2' in self.extra_args:
            self.id2 = self.extra_args['id2']
        else:
            self.id2 = self.expt_cfg['id2']

        self.proc_tom =   self.proc_tom_phase_matrix[self.id2][self.id1]


        self.pi_ef_qubit_phase = self.proc_tom[2][6][4]
        # self.pi_ef_qubit_phase = self.proc_tom[2][6][4]
            # self.pi_ef_qubit_phase = 0


        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses,sb_cool = sb_cool)

    def define_points(self):
        ## we define
        ## automauted
        self.tomography_pulse_num = 15

        if self.proc_tom_set == 0:
            self.expt_pts = np.arange(120)
        else:
            self.expt_pts = np.arange(120,240)

    def define_parameters(self):

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']


        if 'gate_num' in self.extra_args:
            self.gate_num = self.extra_args['gate_num']
        else:
            self.gate_num = 0


        self.offset_phase = self.pulse_cfg[self.pulse_type]['offset_phase']


        self.gate_phase = 0
        self.mode_mode_cnot_dc_phase = self.cfg['mode_mode_offset']['cnot_dc_phase']
        self.mode_mode_cnot_phase = self.cfg['mode_mode_offset']['cnot_phase']
        self.mode_mode_cnot_phase2 = self.cfg['mode_mode_offset']['cnot_phase2']

        self.mode_mode_cz_dc_phase = self.cfg['mode_mode_offset']['cz_dc_phase']
        self.mode_mode_cz_phase = self.cfg['mode_mode_offset']['cz_phase']
        self.mode_mode_cz_phase2 = self.cfg['mode_mode_offset']['cz_phase2']

        self.cz_phase_cz = self.mode_mode_cz_phase[self.id1][self.id2]
        self.cz_phase2_cz = self.mode_mode_cz_phase2[self.id1][self.id2]

        self.cz_phase_zc = self.mode_mode_cz_phase[self.id2][self.id1]
        self.cz_phase2_zc = self.mode_mode_cz_phase2[self.id2][self.id1]

        self.cnot_phase_cx = self.mode_mode_cnot_phase[self.id1][self.id2]
        self.cnot_phase2_cx = self.mode_mode_cnot_phase2[self.id1][self.id2]

        self.cnot_phase_xc = self.mode_mode_cnot_phase[self.id2][self.id1]
        self.cnot_phase2_xc = self.mode_mode_cnot_phase2[self.id2][self.id1]
        self.tom_pulse_list = ['0','half_pi_y','half_pi','pi']

        # print "Phase added to first ef sideband of CPhase = " +str(self.ef_phase_0)
        # print "Additional Phase added (subtracted) to/from first (second) ef sideband of CPhase (CNOT) = " +str(self.ef_phase_1)

    def define_pulses(self,pt):

        ### Initiate states
        self.define_states(pt)
        ### Act Gate
        self.define_gate(pt)
        ### Tomography
        self.define_tomography_pulse(pt)

    def define_states(self,pt):
        ### Preparing all input states required for two-mode process tomography

        self.state_num = pt%16
        self.pulse_num_id1 = self.state_num%4
        self.pulse_num_id2 = self.state_num/4


        if self.pulse_num_id1  == 1 or self.pulse_num_id1 == 2:
            add_phase_1 = -self.offset_phase
        else:
            add_phase_1 = 0

        if self.pulse_num_id2  == 1 or self.pulse_num_id2 == 2:
            add_phase_2 = -self.offset_phase
        else:
            add_phase_2 = 0

        self.psb.append('q',self.tom_pulse_list[self.pulse_num_id2], self.pulse_type)
        self.psb.append('q,mm'+str(self.id2),'pi_ge',phase = self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + add_phase_2 - 90.0)
        self.psb.append('q',self.tom_pulse_list[self.pulse_num_id1], self.pulse_type)
        self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + add_phase_1 - 90.0)

    def define_gate(self,pt):
        ### Preparing all input states required for two-mode process tomography
        self.tomography_num = pt/16
        self.state_num = pt%16
        tomo_index = (self.tomography_num)%self.tomography_pulse_num
        self.ef_phase_0 = self.proc_tom[0][self.state_num][self.tomography_num]
        self.ef_phase_1 =  self.proc_tom[1][self.state_num][self.tomography_num]
        self.add_cphase_g = 0

        if self.gate_num is not 0:
            self.add_cphase_g += self.ef_phase_0
            if tomo_index in array([4,5,8,9]):
                self.add_cphase_g += self.ef_phase_1

        if self.gate_num ==0:
            # I
            pass

        elif self.gate_num == 1:
            # CZ
            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.cz_phase_cz+self.add_cphase_g,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
        elif self.gate_num == 2:
            # CX
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=0,efsbphase_0=self.cnot_phase_cx + self.add_cphase_g ,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0)
        elif self.gate_num == 3:
            # CY
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=-90.0,efsbphase_0=self.cnot_phase_cx + self.add_cphase_g ,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0+ 90.0)
        elif self.gate_num == -1:
            self.gate_phase = 180.0
        else:
            pass

    def define_tomography_pulse(self,pt):

        ### Correlators for two mode tomography; while sweeping the phase of the ef sideband pulse within the gate (in define gate)

        # State convention : |id2, id1 >
        # Gate convention CNOT/CZ(control_id,target_id)

        self.tomography_num = pt/16
        self.state_num = pt%16
        tomo_index = (self.tomography_num)%self.tomography_pulse_num
        self.ef_phase_0 = self.proc_tom[0][self.state_num][self.tomography_num]
        self.ef_phase_1 =  self.proc_tom[1][self.state_num][self.tomography_num]

        #CNOT(id2, id1) = CX
        #CZ(id2, id1) = CZ


        #### Adds ef phase 0 to first ef sideband of CNOT gate

        self.add_cphase = 0
        self.final_sb_phase = 0
        self.add_cnot_phase_2 = 0
        if self.gate_num in array([0,-1]):
            self.add_cphase  = self.ef_phase_0
        if tomo_index in array([4,5,8,9]):
            if self.gate_num in array([0,-1]) :
                self.add_cphase += self.ef_phase_1
            self.add_cnot_phase_2 -= self.ef_phase_1


        ### Adds a CPhase gate to all correlators that do not involve a CNOT gate

        if tomo_index not in array([4,5,8,9,14]):
            ### i.e not in XX, XY, YX, YY, ZZ
            if tomo_index in array([6,10,12,13]):
                ### i.e is XZ,YZ,ZX, ZY
                id_phase = 0.0
            else:
                id_phase = 180.0
            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.gate_phase+self.cz_phase_cz +  id_phase + self.add_cphase,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
        else:
            ### Gate added for CX correlators later
            pass


        if tomo_index == 0:

            # -<IX>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

            # self.halfpicounter2+=1

        elif tomo_index == 1:
            # <IY>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)

            # self.halfpicounter2+=1

        elif tomo_index == 2:
            # <IZ>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0+ self.final_sb_phase)

        elif tomo_index == 3:
            # -<XI>
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)
            # self.halfpicounter2+=1
        elif tomo_index == 4:
            # <XX> = XI + CX

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.gate_phase+self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

            print "ef qubit phase = " +str(self.pi_ef_qubit_phase)
            print "add c phase = " +str(self.add_cphase)
            print "add cnot phase 2 = " +str(self.add_cnot_phase_2)
            print "cnot_phase_2_cx = " +str(self.cnot_phase2_cx)
            print "ef phase 1 = " +str(self.ef_phase_1)
            print "ef phase 0 = "+str(self.ef_phase_0)


        elif tomo_index == 5:
            # -<XY> = XI + CY

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase= -90.0 + self.pi_ef_qubit_phase,efsbphase_0=self.gate_phase+self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)


        elif tomo_index == 6:
            # <XZ>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 7:
            # <YI>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)

        elif tomo_index == 8:
            # -<YX>

            #CNOT
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.gate_phase+self.cnot_phase_cx+self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 9:
            # <YY>

            #CY
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=-90.0+self.pi_ef_qubit_phase,efsbphase_0=self.gate_phase+self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 10:
            # -<YZ>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 11:
            # <ZI>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)

        elif tomo_index == 12:
            # <ZX>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 13:
            # <ZY>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 14:
            # <ZZ>

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.gate_phase+self.cnot_phase_cx,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2)
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
### Test and old process tomography sequences

class MultimodeProcessTomographyPhaseSweepSequence_test(QubitPulseSequence):

    def __init__(self,name, cfg, expt_cfg,**kwargs):

        self.qubit_cfg = cfg['qubit']
        self.cfg = cfg
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.proc_tom_phase_matrix = load("S:\\_Data\\160912 - 2D Multimode Qubit (Chip MM3, 11 modes)\\process_tomography_correlations_test.npy")
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)

        if 'sb_cool' in self.extra_args:
            sb_cool = self.extra_args['sb_cool']
        else:
            sb_cool = False

        if 'pair_index' in self.extra_args:
            self.pair_index = self.extra_args['pair_index']
        else:
            self.pair_index = 0
        print "Pair index= " + str(self.pair_index)

        if 'id1' in self.extra_args:
            self.id1 = self.extra_args['id1']
        else:
            self.id1 = self.expt_cfg['id1']

        if 'id2' in self.extra_args:
            self.id2 = self.extra_args['id2']
        else:
            self.id2 = self.expt_cfg['id2']

        self.proc_tom =   self.proc_tom_phase_matrix[self.id2][self.id1]
        self.tomography_num = self.extra_args['tomography_num']
        self.state_num = self.extra_args['state_num']
        self.ef_phase_0 = 0
        self.ef_phase_1 = 0

        if 'sweep_time' in self.extra_args:
            self.sweep_time = self.extra_args['sweep_time']
        else:
            self.sweep_time = 0

        if 'phase_correct_cz' in self.extra_args:
            print 'phase_correct_cz: ' + str(self.extra_args['phase_correct_cz'])
            if self.extra_args['phase_correct_cz']:
                self.ef_phase_0 = self.proc_tom[0][self.state_num][self.tomography_num]
                print "ef phase in first ef sideband of CPhase: " + str(self.ef_phase_0)

        if 'phase_correct_cnot' in self.extra_args:
            print 'phase_correct_cnot: ' + str(self.extra_args['phase_correct_cnot'])
            self.phase_correct_cnot = self.extra_args['phase_correct_cnot']
        else:
            self.phase_correct_cnot = False

        if self.phase_correct_cnot:
            self.ef_phase_1 =  self.proc_tom[1][self.state_num][self.tomography_num]
            print "Added (subtracted) phase in first (second) ef sideband of CPhase (CNOT): " + str(self.ef_phase_1)

        if 'cnot_ef_qubit_phase' in self.extra_args:
            self.pi_ef_qubit_phase = self.extra_args['cnot_ef_qubit_phase']
            print "Phase ef pi pulse in CNOT gate: " + str(self.pi_ef_qubit_phase) + " deg"
        else:
            self.pi_ef_qubit_phase = 0

        if 'use_saved_cnot_ef_qubit_phase' in self.extra_args:
            self.use_saved_cnot_ef_qubit_phase = self.extra_args['use_saved_cnot_ef_qubit_phase']
        else:
            self.use_saved_cnot_ef_qubit_phase = False

        if self.use_saved_cnot_ef_qubit_phase:
            self.pi_ef_qubit_phase = self.proc_tom[2][6][4]

        if 'sweep_cnot' in self.extra_args:
            print "CNOT phase sweep: " +str(self.extra_args['sweep_cnot'])
            self.sweep_cnot = self.extra_args['sweep_cnot']
        else:
            self.sweep_cnot = False


        if 'sweep_final_sb' in self.extra_args:
            self.sweep_final_sb = self.extra_args['sweep_final_sb']
            print "Sweeping final sideband: " + str(self.sweep_final_sb)
        else:
            self.sweep_final_sb = False

        if 'ef_sb_offset' in self.extra_args:
            self.ef_sb_offset = self.extra_args['ef_sb_offset']
            print "Additional offset to CZ/CNOT ef sidebands: " + str(self.ef_sb_offset)
        else:
            self.ef_sb_offset = 0

        if 'sweep_ef_qubit_phase' in self.extra_args:
            self.sweep_ef_qubit_phase = self.extra_args['sweep_ef_qubit_phase']
            print "Sweeping ef qubit phase: " + str(self.sweep_ef_qubit_phase)
            self.ef_phase_1 += self.ef_sb_offset
        else:
            self.sweep_ef_qubit_phase = False

        if 'sweep_ef_sb_offset_phase' in self.extra_args:
            self.sweep_ef_sb_offset_phase = self.extra_args['sweep_ef_sb_offset_phase']
            print "Sweeping ef sb offset phase: " + str(self.sweep_ef_sb_offset_phase)
            self.ef_phase_1 += self.ef_sb_offset
        else:
            self.sweep_ef_sb_offset_phase = False


        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses,sb_cool = sb_cool)

    def define_points(self):
        ## we define
        ## automauted
        self.tomography_pulse_num = 15

        self.expt_pts = np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']


        if 'gate_num' in self.extra_args:
            self.gate_num = self.extra_args['gate_num']
        else:
            self.gate_num = 0


        self.id = self.expt_cfg['id']

        self.offset_phase = self.pulse_cfg[self.pulse_type]['offset_phase']
        self.halfpicounter1 = 0
        self.halfpicounter2 = 0

        self.mode_mode_cnot_dc_phase = self.cfg['mode_mode_offset']['cnot_dc_phase']
        self.mode_mode_cnot_phase = self.cfg['mode_mode_offset']['cnot_phase']
        self.mode_mode_cnot_phase2 = self.cfg['mode_mode_offset']['cnot_phase2']

        self.mode_mode_cz_dc_phase = self.cfg['mode_mode_offset']['cz_dc_phase']
        self.mode_mode_cz_phase = self.cfg['mode_mode_offset']['cz_phase']
        self.mode_mode_cz_phase2 = self.cfg['mode_mode_offset']['cz_phase2']

        self.cz_phase_cz = self.mode_mode_cz_phase[self.id1][self.id2]
        self.cz_phase2_cz = self.mode_mode_cz_phase2[self.id1][self.id2]

        self.cz_phase_zc = self.mode_mode_cz_phase[self.id2][self.id1]
        self.cz_phase2_zc = self.mode_mode_cz_phase2[self.id2][self.id1]

        self.cnot_phase_cx = self.mode_mode_cnot_phase[self.id1][self.id2]
        self.cnot_phase2_cx = self.mode_mode_cnot_phase2[self.id1][self.id2]

        self.cnot_phase_xc = self.mode_mode_cnot_phase[self.id2][self.id1]
        self.cnot_phase2_xc = self.mode_mode_cnot_phase2[self.id2][self.id1]
        self.tom_pulse_list = ['0','half_pi_y','half_pi','pi']
        self.pulse_num_id1 = self.state_num%4
        self.pulse_num_id2 = self.state_num/4
        # print "Phase added to first ef sideband of CPhase = " +str(self.ef_phase_0)
        # print "Additional Phase added (subtracted) to/from first (second) ef sideband of CPhase (CNOT) = " +str(self.ef_phase_1)

    def define_pulses(self,pt):

        ### Initiate states
        self.define_states(pt)
        ### Idle
        if 'sweep_time' in self.extra_args:
            self.psb.idle(self.sweep_time)
        else:
            pass
        ### Act Gate
        self.define_gate(pt)
        ### Tomography
        self.define_tomography_pulse(pt)

    def define_states(self,pt):
        ### Preparing all input states required for two-mode process tomography
        if self.pulse_num_id1  == 1 or self.pulse_num_id1 == 2:
            add_phase_1 = -self.offset_phase
        else:
            add_phase_1 = 0

        if self.pulse_num_id2  == 1 or self.pulse_num_id2 == 2:
            add_phase_2 = -self.offset_phase
        else:
            add_phase_2 = 0

        if self.tomography_num in array([6,10]):

            self.psb.append('q',self.tom_pulse_list[self.pulse_num_id1], self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + add_phase_1 - 90.0)
            self.psb.append('q',self.tom_pulse_list[self.pulse_num_id2], self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase = self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + add_phase_2 - 90.0)


        else:

            self.psb.append('q',self.tom_pulse_list[self.pulse_num_id2], self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase = self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + add_phase_2 - 90.0)
            self.psb.append('q',self.tom_pulse_list[self.pulse_num_id1], self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + add_phase_1 - 90.0)


    def define_gate(self,pt):
        ### Preparing all input states required for two-mode process tomography
        tomo_index = (self.tomography_num)%self.tomography_pulse_num

        if self.gate_num ==0:
            # I
            pass

        elif self.gate_num == 1:
            # CZ
            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.cz_phase_cz,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
        elif self.gate_num == 2:
            # CX
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=0,efsbphase_0=self.cnot_phase_cx ,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0)
        elif self.gate_num == 3:
            # CY
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=-90.0,efsbphase_0=self.cnot_phase_cx + self.ef_phase_0 ,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0+ 90.0)
        else:
            pass

    def define_tomography_pulse(self,pt):

        ### Correlators for two mode tomography; while sweeping the phase of the ef sideband pulse within the gate (in define gate)

        # State convention : |id2, id1 >
        # Gate convention CNOT/CZ(control_id,target_id)

        tomo_index = (self.tomography_num)%self.tomography_pulse_num
        #CNOT(id2, id1) = CX
        #CZ(id2, id1) = CZ
        if self.sweep_final_sb:
            self.add_cphase = 0
            if self.phase_correct_cnot:
                if tomo_index in array([4,5,8,9]):
                    self.add_cphase = self.ef_phase_1
        elif self.sweep_ef_qubit_phase:
            self.pi_ef_qubit_phase = pt
            self.add_cphase = self.ef_phase_1

        elif self.sweep_ef_sb_offset_phase:
            self.pi_ef_qubit_phase = 0
            self.ef_phase_1 = self.ef_phase_1 + pt
            self.add_cphase = self.ef_phase_1
        else:
            self.add_cphase = pt

        if self.sweep_cnot:
            self.add_cnot_phase_2 = -pt
        else:
            self.add_cnot_phase_2 = 0
        if self.sweep_final_sb:
            self.final_sb_phase = pt
        else:
            self.final_sb_phase = 0

        ### Adds a CPhase gate to all correlators that do not involve a CNOT gate

        if tomo_index not in array([4,5,8,9,14]):
            ### i.e not in XX, XY, YX, YY, ZZ
            if tomo_index in array([6,10,12,13]):
                ### i.e is XZ,YZ,ZX, ZY
                id_phase = 0.0
                if tomo_index in array([12,13]):
                     cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.ef_phase_0+self.cz_phase_cz +  id_phase + self.add_cphase,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
                else:
                     cphase_v3(self.psb,self.id1,self.id2,efsbphase_0=self.ef_phase_0+self.cz_phase_zc +  id_phase + self.add_cphase,efsbphase_1=self.cz_phase2_zc,gesbphase1=self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0)

            else:
                id_phase = 180.0
                cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.ef_phase_0+self.cz_phase_cz +  id_phase + self.add_cphase,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)



        if tomo_index == 0:

            # -<IX>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

            # self.halfpicounter2+=1

        elif tomo_index == 1:
            # <IY>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)

            # self.halfpicounter2+=1

        elif tomo_index == 2:
            # <IZ>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0+ self.final_sb_phase)

        elif tomo_index == 3:
            # -<XI>
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)
            # self.halfpicounter2+=1
        elif tomo_index == 4:
            # <XX> = XI + CX

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2-self.ef_phase_1)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 5:
            # -<XY> = XI + CY

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase= -90.0 + self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2-self.ef_phase_1)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)


        elif tomo_index == 6:
            # <XZ>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 7:
            # <YI>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)

        elif tomo_index == 8:
            # -<YX>

            #CNOT
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx+self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2-self.ef_phase_1)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 9:
            # <YY>

            #CY
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=-90.0+self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2-self.ef_phase_1)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 10:
            # -<YZ>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)

### Corrected on 02_13

        elif tomo_index == 11:
            # <ZI>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)

        elif tomo_index == 12:
            # <ZX>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 13:
            # <ZY>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 14:
            # <ZZ>

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2+self.ef_phase_1)
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)

class MultimodeProcessTomographyPhaseSweepSequence_2_old(QubitPulseSequence):


    def __init__(self,name, cfg, expt_cfg,**kwargs):

        self.qubit_cfg = cfg['qubit']
        self.cfg = cfg
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.proc_tom_phase_matrix = load("S:\\_Data\\160912 - 2D Multimode Qubit (Chip MM3, 11 modes)\\process_tomography_correlations.npy")
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)

        if 'sb_cool' in self.extra_args:
            sb_cool = self.extra_args['sb_cool']
        else:
            sb_cool = False

        if 'pair_index' in self.extra_args:
            self.pair_index = self.extra_args['pair_index']
        else:
            self.pair_index = 0
        print "Pair index= " + str(self.pair_index)

        if 'id1' in self.extra_args:
            self.id1 = self.extra_args['id1']
        else:
            self.id1 = self.expt_cfg['id1']

        if 'id2' in self.extra_args:
            self.id2 = self.extra_args['id2']
        else:
            self.id2 = self.expt_cfg['id2']

        self.proc_tom =   self.proc_tom_phase_matrix[self.id2][self.id1]
        self.tomography_num = self.extra_args['tomography_num']
        self.state_num = self.extra_args['state_num']
        self.ef_phase_0 = 0
        self.ef_phase_1 = 0

        if 'sweep_time' in self.extra_args:
            self.sweep_time = self.extra_args['sweep_time']
        else:
            self.sweep_time = 0

        if 'phase_correct_cz' in self.extra_args:
            print 'phase_correct_cz: ' + str(self.extra_args['phase_correct_cz'])
            if self.extra_args['phase_correct_cz']:
                self.ef_phase_0 = self.proc_tom[0][self.state_num][self.tomography_num]
                print "ef phase in first ef sideband of CPhase: " + str(self.ef_phase_0)

        if 'phase_correct_cnot' in self.extra_args:
            print 'phase_correct_cnot: ' + str(self.extra_args['phase_correct_cnot'])
            self.phase_correct_cnot = self.extra_args['phase_correct_cnot']
        else:
            self.phase_correct_cnot = False

        if self.phase_correct_cnot:
            self.ef_phase_1 =  self.proc_tom[1][self.state_num][self.tomography_num]
            print "Added (subtracted) phase in first (second) ef sideband of CPhase (CNOT): " + str(self.ef_phase_1)

        if 'cnot_ef_qubit_phase' in self.extra_args:
            self.pi_ef_qubit_phase = self.extra_args['cnot_ef_qubit_phase']
            print "Phase ef pi pulse in CNOT gate: " + str(self.pi_ef_qubit_phase) + " deg"
        else:
            self.pi_ef_qubit_phase = 0

        if 'use_saved_cnot_ef_qubit_phase' in self.extra_args:
            self.use_saved_cnot_ef_qubit_phase = self.extra_args['use_saved_cnot_ef_qubit_phase']
        else:
            self.use_saved_cnot_ef_qubit_phase = False

        if self.use_saved_cnot_ef_qubit_phase:
            self.pi_ef_qubit_phase = self.proc_tom[2][6][4]

        if 'sweep_cnot' in self.extra_args:
            print "CNOT phase sweep: " +str(self.extra_args['sweep_cnot'])
            self.sweep_cnot = self.extra_args['sweep_cnot']
        else:
            self.sweep_cnot = False


        if 'sweep_final_sb' in self.extra_args:
            self.sweep_final_sb = self.extra_args['sweep_final_sb']
            print "Sweeping final sideband: " + str(self.sweep_final_sb)
        else:
            self.sweep_final_sb = False

        if 'ef_sb_offset' in self.extra_args:
            self.ef_sb_offset = self.extra_args['ef_sb_offset']
            print "Additional offset to CZ/CNOT ef sidebands: " + str(self.ef_sb_offset)
        else:
            self.ef_sb_offset = 0

        if 'sweep_ef_qubit_phase' in self.extra_args:
            self.sweep_ef_qubit_phase = self.extra_args['sweep_ef_qubit_phase']
            print "Sweeping ef qubit phase: " + str(self.sweep_ef_qubit_phase)
            self.ef_phase_1 += self.ef_sb_offset
        else:
            self.sweep_ef_qubit_phase = False

        if 'sweep_ef_sb_offset_phase' in self.extra_args:
            self.sweep_ef_sb_offset_phase = self.extra_args['sweep_ef_sb_offset_phase']
            print "Sweeping ef sb offset phase: " + str(self.sweep_ef_sb_offset_phase)
            self.ef_phase_1 += self.ef_sb_offset
        else:
            self.sweep_ef_sb_offset_phase = False


        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses,sb_cool = sb_cool)

    def define_points(self):
        ## we define
        ## automauted
        self.tomography_pulse_num = 15

        self.expt_pts = np.arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']


        if 'gate_num' in self.extra_args:
            self.gate_num = self.extra_args['gate_num']
        else:
            self.gate_num = 0


        self.id = self.expt_cfg['id']

        self.offset_phase = self.pulse_cfg[self.pulse_type]['offset_phase']
        self.halfpicounter1 = 0
        self.halfpicounter2 = 0

        self.mode_mode_cnot_dc_phase = self.cfg['mode_mode_offset']['cnot_dc_phase']
        self.mode_mode_cnot_phase = self.cfg['mode_mode_offset']['cnot_phase']
        self.mode_mode_cnot_phase2 = self.cfg['mode_mode_offset']['cnot_phase2']

        self.mode_mode_cz_dc_phase = self.cfg['mode_mode_offset']['cz_dc_phase']
        self.mode_mode_cz_phase = self.cfg['mode_mode_offset']['cz_phase']
        self.mode_mode_cz_phase2 = self.cfg['mode_mode_offset']['cz_phase2']

        self.cz_phase_cz = self.mode_mode_cz_phase[self.id1][self.id2]
        self.cz_phase2_cz = self.mode_mode_cz_phase2[self.id1][self.id2]

        self.cz_phase_zc = self.mode_mode_cz_phase[self.id2][self.id1]
        self.cz_phase2_zc = self.mode_mode_cz_phase2[self.id2][self.id1]

        self.cnot_phase_cx = self.mode_mode_cnot_phase[self.id1][self.id2]
        self.cnot_phase2_cx = self.mode_mode_cnot_phase2[self.id1][self.id2]

        self.cnot_phase_xc = self.mode_mode_cnot_phase[self.id2][self.id1]
        self.cnot_phase2_xc = self.mode_mode_cnot_phase2[self.id2][self.id1]
        self.tom_pulse_list = ['0','half_pi_y','half_pi','pi']
        self.pulse_num_id1 = self.state_num%4
        self.pulse_num_id2 = self.state_num/4
        # print "Phase added to first ef sideband of CPhase = " +str(self.ef_phase_0)
        # print "Additional Phase added (subtracted) to/from first (second) ef sideband of CPhase (CNOT) = " +str(self.ef_phase_1)

    def define_pulses(self,pt):

        ### Initiate states
        self.define_states(pt)
        ### Idle
        if 'sweep_time' in self.extra_args:
            self.psb.idle(self.sweep_time)
        else:
            pass
        ### Act Gate
        self.define_gate(pt)
        ### Tomography
        self.define_tomography_pulse(pt)

    def define_states(self,pt):
        ### Preparing all input states required for two-mode process tomography
        if self.pulse_num_id1  == 1 or self.pulse_num_id1 == 2:
            add_phase_1 = -self.offset_phase
        else:
            add_phase_1 = 0

        if self.pulse_num_id2  == 1 or self.pulse_num_id2 == 2:
            add_phase_2 = -self.offset_phase
        else:
            add_phase_2 = 0

        self.psb.append('q',self.tom_pulse_list[self.pulse_num_id2], self.pulse_type)
        self.psb.append('q,mm'+str(self.id2),'pi_ge',phase = self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + add_phase_2 - 90.0)
        self.psb.append('q',self.tom_pulse_list[self.pulse_num_id1], self.pulse_type)
        self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + add_phase_1 - 90.0)

    def define_gate(self,pt):
        ### Preparing all input states required for two-mode process tomography
        tomo_index = (self.tomography_num)%self.tomography_pulse_num

        if self.gate_num ==0:
            # I
            pass

        elif self.gate_num == 1:
            # CZ
            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.cz_phase_cz,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
        elif self.gate_num == 2:
            # CX
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=0,efsbphase_0=self.cnot_phase_cx ,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0)
        elif self.gate_num == 3:
            # CY
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=-90.0,efsbphase_0=self.cnot_phase_cx + self.ef_phase_0 ,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0+ 90.0)
        else:
            pass

    def define_tomography_pulse(self,pt):

        ### Correlators for two mode tomography; while sweeping the phase of the ef sideband pulse within the gate (in define gate)

        # State convention : |id2, id1 >
        # Gate convention CNOT/CZ(control_id,target_id)

        tomo_index = (self.tomography_num)%self.tomography_pulse_num
        #CNOT(id2, id1) = CX
        #CZ(id2, id1) = CZ
        if self.sweep_final_sb:
            self.add_cphase = 0
            if self.phase_correct_cnot:
                if tomo_index in array([4,5,8,9]):
                    self.add_cphase = self.ef_phase_1
        elif self.sweep_ef_qubit_phase:
            self.pi_ef_qubit_phase = pt
            self.add_cphase = self.ef_phase_1

        elif self.sweep_ef_sb_offset_phase:
            self.pi_ef_qubit_phase = 0
            self.ef_phase_1 = self.ef_phase_1 + pt
            self.add_cphase = self.ef_phase_1
        else:
            self.add_cphase = pt

        if self.sweep_cnot:
            self.add_cnot_phase_2 = -pt
        else:
            self.add_cnot_phase_2 = 0
        if self.sweep_final_sb:
            self.final_sb_phase = pt
        else:
            self.final_sb_phase = 0

        ### Adds a CPhase gate to all correlators that do not involve a CNOT gate

        if tomo_index not in array([4,5,8,9,14]):
            ### i.e not in XX, XY, YX, YY, ZZ
            if tomo_index in array([6,10,12,13]):
                ### i.e is XZ,YZ,ZX, ZY
                id_phase = 0.0
            else:
                id_phase = 180.0
            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.ef_phase_0+self.cz_phase_cz +  id_phase + self.add_cphase,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)



        if tomo_index == 0:

            # -<IX>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

            # self.halfpicounter2+=1

        elif tomo_index == 1:
            # <IY>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)

            # self.halfpicounter2+=1

        elif tomo_index == 2:
            # <IZ>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0+ self.final_sb_phase)

        elif tomo_index == 3:
            # -<XI>
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)
            # self.halfpicounter2+=1
        elif tomo_index == 4:
            # <XX> = XI + CX

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2-self.ef_phase_1)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 5:
            # -<XY> = XI + CY

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase= -90.0 + self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2-self.ef_phase_1)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)


        elif tomo_index == 6:
            # <XZ>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 7:
            # <YI>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)

        elif tomo_index == 8:
            # -<YX>

            #CNOT
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx+self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2-self.ef_phase_1)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 9:
            # <YY>

            #CY
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=-90.0+self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2-self.ef_phase_1)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 10:
            # -<YZ>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 11:
            # <ZI>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)

        elif tomo_index == 12:
            # <ZX>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 13:
            # <ZY>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 14:
            # <ZZ>

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2+self.ef_phase_1)
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)

class MultimodeProcessTomographySequence_2_old(QubitPulseSequence):

    def __init__(self,name, cfg, expt_cfg,**kwargs):

        self.qubit_cfg = cfg['qubit']
        self.cfg = cfg
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        self.proc_tom_phase_matrix = load("S:\\_Data\\160912 - 2D Multimode Qubit (Chip MM3, 11 modes)\\process_tomography_correlations.npy")
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)

        if 'sb_cool' in self.extra_args:
            sb_cool = self.extra_args['sb_cool']
        else:
            sb_cool = False

        if 'proc_tom_set' in self.extra_args:
            self.proc_tom_set = kwargs['proc_tom_set']
        else:
            self.proc_tom_set = 0

        if 'id1' in self.extra_args:
            self.id1 = self.extra_args['id1']
        else:
            self.id1 = self.expt_cfg['id1']

        if 'id2' in self.extra_args:
            self.id2 = self.extra_args['id2']
        else:
            self.id2 = self.expt_cfg['id2']

        self.proc_tom =   self.proc_tom_phase_matrix[self.id2][self.id1]

        self.ef_phase_0 = 0
        self.ef_phase_1 = 0


        self.pi_ef_qubit_phase = self.proc_tom[2][6][4]


        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses,sb_cool = sb_cool)

    def define_points(self):
        ## we define
        ## automauted
        self.tomography_pulse_num = 15

        if self.proc_tom_set == 0:
            self.expt_pts = np.arange(120)
        else:
            self.expt_pts = np.arange(120,240)

    def define_parameters(self):

        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']


        if 'gate_num' in self.extra_args:
            self.gate_num = self.extra_args['gate_num']
        else:
            self.gate_num = 0


        self.offset_phase = self.pulse_cfg[self.pulse_type]['offset_phase']
        self.mode_mode_cnot_dc_phase = self.cfg['mode_mode_offset']['cnot_dc_phase']
        self.mode_mode_cnot_phase = self.cfg['mode_mode_offset']['cnot_phase']
        self.mode_mode_cnot_phase2 = self.cfg['mode_mode_offset']['cnot_phase2']

        self.mode_mode_cz_dc_phase = self.cfg['mode_mode_offset']['cz_dc_phase']
        self.mode_mode_cz_phase = self.cfg['mode_mode_offset']['cz_phase']
        self.mode_mode_cz_phase2 = self.cfg['mode_mode_offset']['cz_phase2']

        self.cz_phase_cz = self.mode_mode_cz_phase[self.id1][self.id2]
        self.cz_phase2_cz = self.mode_mode_cz_phase2[self.id1][self.id2]

        self.cz_phase_zc = self.mode_mode_cz_phase[self.id2][self.id1]
        self.cz_phase2_zc = self.mode_mode_cz_phase2[self.id2][self.id1]

        self.cnot_phase_cx = self.mode_mode_cnot_phase[self.id1][self.id2]
        self.cnot_phase2_cx = self.mode_mode_cnot_phase2[self.id1][self.id2]

        self.cnot_phase_xc = self.mode_mode_cnot_phase[self.id2][self.id1]
        self.cnot_phase2_xc = self.mode_mode_cnot_phase2[self.id2][self.id1]
        self.tom_pulse_list = ['0','half_pi_y','half_pi','pi']

        # print "Phase added to first ef sideband of CPhase = " +str(self.ef_phase_0)
        # print "Additional Phase added (subtracted) to/from first (second) ef sideband of CPhase (CNOT) = " +str(self.ef_phase_1)

    def define_pulses(self,pt):

        ### Initiate states
        self.define_states(pt)
        ### Idle
        ### Act Gate
        self.define_gate(pt)
        ### Tomography
        self.define_tomography_pulse(pt)

    def define_states(self,pt):
        ### Preparing all input states required for two-mode process tomography
        self.state_num = pt%16
        self.pulse_num_id1 = self.state_num%4
        self.pulse_num_id2 = self.state_num/4

        if self.pulse_num_id1  == 1 or self.pulse_num_id1 == 2:
            add_phase_1 = -self.offset_phase
        else:
            add_phase_1 = 0

        if self.pulse_num_id2  == 1 or self.pulse_num_id2 == 2:
            add_phase_2 = -self.offset_phase
        else:
            add_phase_2 = 0



        self.psb.append('q',self.tom_pulse_list[self.pulse_num_id2], self.pulse_type)
        self.psb.append('q,mm'+str(self.id2),'pi_ge',phase = self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + add_phase_2 - 90.0)
        self.psb.append('q',self.tom_pulse_list[self.pulse_num_id1], self.pulse_type)
        self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + add_phase_1 - 90.0)

    def define_gate(self,pt):
        ### Preparing all input states required for two-mode process tomography
        self.tomography_num = pt/16
        self.state_num = pt%16
        tomo_index = (self.tomography_num)%self.tomography_pulse_num
        self.ef_phase_0 = self.proc_tom[0][self.state_num][self.tomography_num]
        self.ef_phase_1 =  self.proc_tom[1][self.state_num][self.tomography_num]
        if self.gate_num ==0:
            # I
            pass

        elif self.gate_num == 1:
            # CZ
            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.cz_phase_cz,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)
        elif self.gate_num == 2:
            # CX
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=0,efsbphase_0=self.cnot_phase_cx ,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0)
        elif self.gate_num == 3:
            # CY
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=-90.0,efsbphase_0=self.cnot_phase_cx + self.ef_phase_0 ,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0+ 90.0)
        else:
            pass

    def define_tomography_pulse(self,pt):

        ### Correlators for two mode tomography; while sweeping the phase of the ef sideband pulse within the gate (in define gate)

        # State convention : |id2, id1 >
        # Gate convention CNOT/CZ(control_id,target_id)
        self.tomography_num = pt/16
        tomo_index = (self.tomography_num)%self.tomography_pulse_num
        self.state_num = pt%16
        tomo_index = (self.tomography_num)%self.tomography_pulse_num
        self.ef_phase_0 = self.proc_tom[0][self.state_num][self.tomography_num]
        self.ef_phase_1 =  self.proc_tom[1][self.state_num][self.tomography_num]
        #CNOT(id2, id1) = CX
        #CZ(id2, id1) = CZ

        self.add_cphase = 0
        if self.tomography_num in array([4,5,8,9]):
            self.add_cphase = self.ef_phase_1


        self.add_cnot_phase_2 =0

        self.final_sb_phase = 0

        ### Adds a CPhase gate to all correlators that do not involve a CNOT gate

        if tomo_index not in array([4,5,8,9,14]):
            ### i.e not in XX, XY, YX, YY, ZZ
            if tomo_index in array([6,10,12,13]):
                ### i.e is XZ,YZ,ZX, ZY
                id_phase = 0.0
            else:
                id_phase = 180.0

            cphase_v3(self.psb,self.id2,self.id1,efsbphase_0=self.ef_phase_0+self.cz_phase_cz +  id_phase + self.add_cphase,efsbphase_1=self.cz_phase2_cz,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0)


        if tomo_index == 0:

            # -<IX>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

            # self.halfpicounter2+=1

        elif tomo_index == 1:
            # <IY>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)

            # self.halfpicounter2+=1

        elif tomo_index == 2:
            # <IZ>
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0+ self.final_sb_phase)

        elif tomo_index == 3:
            # -<XI>
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)
            # self.halfpicounter2+=1
        elif tomo_index == 4:
            # <XX> = XI + CX

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2-self.ef_phase_1)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 5:
            # -<XY> = XI + CY

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase= -90.0 + self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2-self.ef_phase_1)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)


        elif tomo_index == 6:
            # <XZ>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 7:
            # <YI>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)

        elif tomo_index == 8:
            # -<YX>

            #CNOT
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx+self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2-self.ef_phase_1)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 9:
            # <YY>

            #CY
            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=-90.0+self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx + self.add_cphase,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2-self.ef_phase_1)

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 +  90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 10:
            # -<YZ>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 11:
            # <ZI>

            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=-self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)

        elif tomo_index == 12:
            # <ZX>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi_y', self.pulse_type)

        elif tomo_index == 13:
            # <ZY>

            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)
            self.psb.append('q','half_pi', self.pulse_type)


        elif tomo_index == 14:
            # <ZZ>

            cnot_v2(self.psb,self.id2,self.id1,cnot_phase=self.pi_ef_qubit_phase,efsbphase_0=self.cnot_phase_cx,efsbphase_1=self.cnot_phase2_cx,gesbphase1=self.multimode_cfg[self.id2]['pi_pi_offset_phase']/2.0,efsbphase_2=self.add_cnot_phase_2+self.ef_phase_1)
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=-self.multimode_cfg[self.id1]['pi_pi_offset_phase']/2.0 + 90.0 + self.final_sb_phase)


