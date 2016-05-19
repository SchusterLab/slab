__author__ = 'Nelson'

from slab.instruments.awg.PulseSequence import *
from numpy import arange, linspace, sin, pi, sign, append
from slab.experiments.ExpLib.PulseSequenceBuilder import *
from slab.experiments.ExpLib.QubitPulseSequence import *
from slab.experiments.ExpLib.PulseSequenceGroup import *
import random
from numpy import around, mean
from liveplot import LivePlotClient


class MultimodeRabiSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.multimode_cfg = cfg['multimodes']
        # self.pulse_cfg = cfg['pulse_info']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.id = self.expt_cfg['id']
        self.flux_pulse_type = self.multimode_cfg[int(self.id)]['flux_pulse_type']

    def define_pulses(self,pt):
        # self.psb.append('q','pi', self.pulse_type)
        # self.psb.append('q,mm'+str(4),'pi_ge')
        # self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length= self.multimode_cfg[int(self.id)]['flux_pi_length'])
        self.psb.append('q','pi', self.pulse_type)
        self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length=pt)

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
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg[self.exp]['start'], self.expt_cfg[self.exp]['stop'], self.expt_cfg[self.exp]['step'])


    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.id = self.mode
        self.flux_pulse_type = self.multimode_cfg[int(self.id)]['flux_pulse_type']
        if self.exp == 'multimode_rabi':
            pass
        else:
            if self.flux_pulse_cfg['fix_phase']:
                self.phase_freq = self.expt_cfg[self.exp]['ramsey_freq']
            else:
                self.phase_freq = self.multimode_cfg[int(self.id)]['dc_offset_freq'] + self.expt_cfg[self.exp]['ramsey_freq']


    def define_pulses(self,pt):
        if self.exp == 'multimode_rabi':
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length=pt)
        else:
            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.idle(pt)
            self.psb.append('q,mm'+str(self.id),'pi_ge', phase = 360.0*self.phase_freq*pt/(1.0e9))
            self.psb.append('q','half_pi', self.pulse_type)


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
        self.dc_offset_guess_ef =  self.extra_args['dc_offset_guess_ef']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg[self.exp]['start'], self.expt_cfg[self.exp]['stop'], self.expt_cfg[self.exp]['step'])


    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.id = self.mode
        self.flux_pulse_type = self.multimode_cfg[int(self.id)]['flux_pulse_type']
        if 'mode' in self.extra_args:
            pass
        else:
            self.exp = self.expt_cfg['exp']
            self.mode = self.expt_cfg['mode']

        if self.exp == 'multimode_ef_rabi':
            pass
        else:
            if self.flux_pulse_cfg['fix_phase']:
                self.phase_freq = self.expt_cfg[self.exp]['ramsey_freq']
            else:
                self.phase_freq = self.dc_offset_guess_ef + self.expt_cfg[self.exp]['ramsey_freq']

    def define_pulses(self,pt):

        if self.exp == 'multimode_ef_rabi':
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q','pi_q_ef')
            self.psb.append('q:mm','general', self.flux_pulse_type, amp=self.multimode_cfg[self.mode]['a_ef'], length=pt,freq=self.multimode_cfg[self.mode]['flux_pulse_freq_ef'])
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
        self.psb.idle(pt)
        self.psb.append('q,mm'+str(self.id),'pi_ef', phase = 360.0*self.phase_freq*pt/(1.0e9))
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
        if 'amp' in kwargs:
            self.amp = kwargs['amp']
        else:
            self.amp = self.expt_cfg['a']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses, **kwargs)


    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']

    def define_pulses(self,pt):
        self.psb.append('q','pi', self.pulse_type)
        #self.psb.append('q,mm0','general', self.flux_pulse_type, amp=self.expt_cfg['a'], length=pt)
        self.psb.append('q:mm','general', self.flux_pulse_type, amp=self.amp, length=pt,freq=self.flux_freq)

class MultimodeDCOffsetSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        self.amp = self.extra_args['amp']
        self.freq = self.extra_args['freq']
        self.multimode_cfg = cfg['multimodes']
        # self.pulse_cfg = cfg['pulse_info']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.flux_pulse_type = self.expt_cfg['flux_pulse_type']

    def define_pulses(self,pt):
        self.psb.append('q','half_pi', self.pulse_type)
        self.psb.append('q:mm','general', self.flux_pulse_type, amp=self.amp, length=pt,freq=self.freq)
        self.psb.append('q','half_pi', self.pulse_type, phase = 360.0*self.expt_cfg['ramsey_freq']*pt/(1.0e9))
        # self.psb.append('q,mm'+str(14),'pi_ge')


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
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses, **kwargs)


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
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.id = self.expt_cfg['id']
        self.flux_pulse_type = self.multimode_cfg[int(self.id)]['flux_pulse_type']

    def define_pulses(self,pt):
        self.psb.append('q','pi', self.pulse_type)
        #self.psb.append('q,mm0','pi', self.pulse_type)
        self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(+self.id)]['a'], length= self.multimode_cfg[int(self.id)]['flux_pi_length'])
        self.psb.idle(pt)
        #self.psb.append('q,mm0','general', self.flux_pulse_type, amp=self.expt_cfg['a'], length= self.expt_cfg['pi_sb_ge'])
        self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(+self.id)]['a'], length= self.multimode_cfg[int(self.id)]['flux_pi_length'],phase= 360.0*self.multimode_cfg[int(self.id)]['dc_offset_freq']*pt/(1.0e9))
        #self.psb.append('q,mm0','pi', self.pulse_type)


class MultimodeGeneralEntanglementSequence(QubitPulseSequence):
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
        self.psb.append('q','pi', self.pulse_type)
        self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= pt)
        self.psb.append('q','pi', self.pulse_type)
        for ii in np.arange(2,self.number):
            self.psb.append('q','pi_q_ef')
            self.psb.append('q,mm'+str(getattr(self, "id"+str(ii))),'pi_ef')
        self.psb.append('q,mm'+str(getattr(self, "id"+str(self.number))),'pi_ge')
        self.psb.append('q,mm'+str(self.idm),'pi_ge')

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
        self.id1 = self.expt_cfg['id1']
        self.id2 = self.expt_cfg['id2']
        self.id3 = self.expt_cfg['id3']
        self.idm = self.expt_cfg['idm']


    def define_pulses(self,pt):
        if self.expt_cfg['2_mode']:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= pt, phase=180)
            if self.expt_cfg['GHZ']:
               self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id2)]['a'], length=  self.multimode_cfg[int(self.id2)]['flux_pi_length'], phase=180)
            self.psb.append('q,mm'+str(self.idm),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.idm)]['a'], length=  self.multimode_cfg[int(self.idm)]['flux_pi_length'])


        elif self.expt_cfg['3_mode']:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= pt)
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q','pi_q_ef')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q,mm'+str(self.id3),'pi_ge')
            self.psb.append('q,mm'+str(self.idm),'pi_ge')


class MultimodeCPhaseTestsSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        if self.expt_cfg["tomography"]:
            self.expt_pts = np.array([0,1,2])
        else:
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
        self.idef = self.expt_cfg['idef']
        self.half_pi_offset = self.pulse_cfg[self.pulse_type]['offset_phase']


    def define_pulses(self,pt):

        if self.expt_cfg["tomography"]:
            par = self.expt_cfg["time_slice"]
        else:
            if self.expt_cfg["slice"]:
                par = self.expt_cfg["time_slice"]
            else:
                par = pt

        # Test of phase with 2pi sideband rotation with variation of the length of guassian qubit drive pulse

        if self.expt_cfg["test_2pisideband"]:
            self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'])
            # self.psb.idle(100)
            if self.expt_cfg["include_2pisideband"]:
                # self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length=self.multimode_cfg[int(self.id)]['flux_2pi_length'])
                self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length=self.multimode_cfg[int(self.id)]['flux_pi_length'])
                self.psb.idle(100)
                self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length=self.multimode_cfg[int(self.id)]['flux_pi_length'], phase = self.multimode_cfg[int(self.id)]['pi_pi_offset_phase'])
                self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length=self.multimode_cfg[int(self.id)]['flux_pi_length'])
                self.psb.idle(100)
                self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length=self.multimode_cfg[int(self.id)]['flux_pi_length'], phase = self.multimode_cfg[int(self.id)]['pi_pi_offset_phase'])

            if self.expt_cfg["include_theta_offset_phase"]:
               self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'], phase = self.multimode_cfg[int(self.id)]['2pi_offset_phase'] )
            else:
                self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'], phase = 0)

        # Finding offset phase for second theta pulse due to the 2pi sideband

        if self.expt_cfg["find_2pi_sideband_offset"]:
            self.psb.append('q','general', self.pulse_type, amp=1, length=self.expt_cfg['theta_length'], freq=self.expt_cfg['iq_freq'])
            self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length=self.multimode_cfg[int(self.id)]['flux_2pi_length'])
            self.psb.append('q','general', self.pulse_type, amp=1, length=self.expt_cfg['theta_length'], freq=self.expt_cfg['iq_freq'],phase=pt)


        #  Finding offset phase for second theta pulse due 2 pi sidebands

        if self.expt_cfg["find_pi_pi_sideband_offset"]:
            self.psb.append('q','half_pi',self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge')
            self.psb.idle(self.expt_cfg["pi_pi_idle"])
            self.psb.append('q,mm'+str(self.id),'pi_ge', phase=pt)
            self.psb.append('q','half_pi',self.pulse_type, self.half_pi_offset)


        # Testing 2pi ef rotation
        if self.expt_cfg["test_2pi_ef_rotation"]:

            self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'])
            self.psb.append('q','general', self.ef_pulse_type, amp=1, length=self.expt_cfg['pi_ef_length'],freq=self.ef_sideband_freq)
            self.psb.idle(100)
            # self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length=self.multimode_cfg[int(self.id)]['flux_pi_length'])
            # self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length=self.multimode_cfg[int(self.id)]['flux_pi_length'])
            self.psb.append('q','general', self.ef_pulse_type, amp=1, length=self.expt_cfg['pi_ef_length'],freq=self.ef_sideband_freq, phase =self.expt_cfg['pi_ef_offset'] )
            self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'])

        if self.expt_cfg["test_ef_with_resonator_loaded"]:

            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length=self.multimode_cfg[int(self.id)]['flux_pi_length'])

            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q','general', self.ef_pulse_type, amp=1, length=par ,freq=self.ef_sideband_freq)
            self.psb.append('q','pi', self.pulse_type)


        if self.expt_cfg["find_2pi_ef_offset"]:
            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q','general', self.ef_pulse_type, amp=1, length=self.expt_cfg['pi_ef_length'],freq=self.ef_sideband_freq)
            self.psb.idle(138)
            self.psb.append('q','general', self.ef_pulse_type, amp=1, length=self.expt_cfg['pi_ef_length'],freq=self.ef_sideband_freq, phase = pt)
            self.psb.append('q','half_pi', self.pulse_type, self.half_pi_offset)

        if self.expt_cfg["test_2pi_ef_sideband_rotation"]:

            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.id)]['a'], length=self.multimode_cfg[int(self.id)]['flux_pi_length'])
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.idef),'general', self.flux_pulse_type, amp=self.multimode_cfg[int(self.idef)]['a'], length=par)
            self.psb.append('q','pi', self.pulse_type)


        if self.expt_cfg["test_cphase"]:

        #State preparation
            if  self.expt_cfg["prepare_state"] == 0:

                self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'])
                self.psb.append('q,mm'+str(self.id1),'pi_ge')

            if  self.expt_cfg["prepare_state"] == 1:

                self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'])
                self.psb.append('q,mm'+str(self.id2),'pi_ge')

            if  self.expt_cfg["prepare_state"] == 2:

                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id1),'pi_ge')
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id2),'pi_ge')

            if  self.expt_cfg["prepare_state"] == 3:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id1),'pi_ge')
                self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'])
                self.psb.append('q,mm'+str(self.id2),'pi_ge')

            # <XZ>

            if  self.expt_cfg["prepare_state"] == 4:
                self.psb.append('q','half_pi_y', self.pulse_type)
                self.psb.append('q,mm'+str(self.id1),'pi_ge')
                self.psb.append('q,mm'+str(self.id2),'pi_ge')


        # Cphase Gate



            if self.expt_cfg["cphase_on"]:

                if self.expt_cfg["cphase_type"]==0:


                    self.psb.append('q,mm'+str(self.id2),'pi_ge')
                    self.psb.append('q','general', self.ef_pulse_type, amp=1, length=self.expt_cfg['pi_ef_length'],freq=self.ef_sideband_freq)
                    self.psb.append('q,mm'+str(self.id1),'pi_ge')
                    self.psb.append('q,mm'+str(self.id1),'pi_ge')
                    self.psb.append('q','general', self.ef_pulse_type, amp=1, length=self.expt_cfg['pi_ef_length'],freq=self.ef_sideband_freq, phase = self.expt_cfg['pi_ef_offset'] )
                    self.psb.append('q,mm'+str(self.id2),'pi_ge')

                if self.expt_cfg["cphase_type"]==1:

                    cphase(self.psb,self.id1,self.id2)


            else:
                self.psb.idle(self.expt_cfg['no_cphase_idle'])

        #Reversing State preparation

            if  self.expt_cfg["prepare_state"] == 0:
                self.psb.append('q,mm'+str(self.id1),'pi_ge', phase = pt )
                self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'])

            if  self.expt_cfg["prepare_state"] == 1:
                self.psb.append('q,mm'+str(self.id2),'pi_ge', phase =   pt )
                self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'])

            if  self.expt_cfg["prepare_state"] == 2:

                pass

            if  self.expt_cfg["prepare_state"] == 3:
                self.psb.append('q,mm'+str(self.id2),'pi_ge', phase = pt)
                self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'])

            if  self.expt_cfg["prepare_state"] == 4:
                self.psb.append('q,mm'+str(self.id1),'pi_ge', phase = pt)
                self.psb.append('q','half_pi_y', self.pulse_type)

        # Tomography at a given time slice

            if self.expt_cfg["tomography"]:
                 ### gates before measurement for tomography
                if pt == 0:
                    # <X>
                    self.psb.append('q','half_pi', self.pulse_type)
                elif pt == 1:
                    # <Y>
                    self.psb.append('q','half_pi_y', self.pulse_type)
                elif pt == 2:
                    # <Z>
                    pass

class MultimodeCPhaseSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        if self.expt_cfg["tomography"]:
            self.expt_pts = np.array([0,1,2])
        else:
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
        self.idef = self.expt_cfg['idef']
        self.half_pi_offset = self.pulse_cfg[self.pulse_type]['offset_phase']
        self.halfpicounter1 = 0
        self.halfpicounter2 = 0


    def define_pulses(self,pt):

        if self.expt_cfg["tomography"]:
            par = self.expt_cfg["time_slice"]
        else:
            if self.expt_cfg["slice"]:
                par = self.expt_cfg["time_slice"]
            else:
                par = pt


        if self.expt_cfg["test_cphase"]:

        #State preparation
            if  self.expt_cfg["prepare_state"] == 0:

                self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['pulse_type']['a'], length=par, freq=self.expt_cfg['iq_freq'])
                self.psb.append('q,mm'+str(self.id1),'pi_ge')

            if  self.expt_cfg["prepare_state"] == 1:

                self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'])
                self.psb.append('q,mm'+str(self.id2),'pi_ge')

            if  self.expt_cfg["prepare_state"] == 2:

                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id1),'pi_ge')
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id2),'pi_ge')

            if  self.expt_cfg["prepare_state"] == 3:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id1),'pi_ge')
                self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'])
                self.psb.append('q,mm'+str(self.id2),'pi_ge')

            # <XZ>

            if  self.expt_cfg["prepare_state"] == 4:
                self.halfpicounter1 = 0
                self.halfpicounter2 = 0
                self.psb.append('q','half_pi_y', self.pulse_type)
                self.psb.append('q,mm'+str(self.id1),'pi_ge')
                self.halfpicounter1+=1
                # self.psb.idle(168.5)
                # self.psb.append('q','pi',self.pulse_type)
                # self.psb.append('q,mm'+str(self.id2),'pi_ge')

            # <YZ>

            if  self.expt_cfg["prepare_state"] == 5:
                self.halfpicounter1 = 0
                self.halfpicounter2 = 0

                self.psb.append('q','half_pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id1),'pi_ge')
                self.halfpicounter1+=1
                self.psb.append('q','pi',self.pulse_type)
                self.psb.append('q,mm'+str(self.id2),'pi_ge')


            #<ZX>

            if  self.expt_cfg["prepare_state"] == 6:
                self.halfpicounter1 = 0
                self.halfpicounter2 = 0
                self.psb.append('q','half_pi_y', self.pulse_type)
                self.psb.append('q,mm'+str(self.id2),'pi_ge')
                self.halfpicounter2+=1
                self.psb.append('q','pi',self.pulse_type)
                self.psb.append('q,mm'+str(self.id1),'pi_ge')

            #<ZY>

            if  self.expt_cfg["prepare_state"] == 7:
                self.halfpicounter1 = 0
                self.halfpicounter2 = 0
                self.psb.append('q','half_pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id2),'pi_ge')
                self.halfpicounter2+=1
                self.psb.append('q','pi',self.pulse_type)
                self.psb.append('q,mm'+str(self.id1),'pi_ge')

            #<IX>
            if  self.expt_cfg["prepare_state"] == 8:
                self.halfpicounter1 = 0
                self.halfpicounter2 = 0
                self.psb.append('q','half_pi_y', self.pulse_type)
                self.psb.append('q,mm'+str(self.id2),'pi_ge')
                self.halfpicounter2+=1
                self.psb.append('q','pi',self.pulse_type)
                self.psb.append('q,mm'+str(self.id1),'pi_ge')

            #phi bell
            if  self.expt_cfg["prepare_state"] == 9:
                self.halfpicounter1 = 0
                self.halfpicounter2 = 0
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= self.expt_cfg['time_slice'], phase=180)
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id2),'pi_ge', phase=180)



        # Cphase Gate


            if self.expt_cfg["cphase_on"]:

                cphase(self.psb,self.id1,self.id2)

            else:
                self.psb.idle(self.expt_cfg['no_cphase_idle'])

        #Reversing State preparation

            if  self.expt_cfg["measure_state"] == 0:
                self.psb.append('q,mm'+str(self.id1),'pi_ge', phase = pt )
                self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['pulse_type']['a'], length=par, freq=self.expt_cfg['iq_freq'])

            if  self.expt_cfg["measure_state"] == 1:
                self.psb.append('q,mm'+str(self.id2),'pi_ge', phase =   pt )
                self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'])

            if  self.expt_cfg["measure_state"] == 2:
                pass

            if  self.expt_cfg["measure_state"] == 3:
                self.psb.append('q,mm'+str(self.id2),'pi_ge', phase = pt)
                self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'])

            #<XZ>
            if  self.expt_cfg["measure_state"] == 4:
                self.psb.append('q,mm'+str(self.id1),'pi_ge', phase = self.expt_cfg['final_offset'] + pt)
                self.psb.append('q','half_pi_y', self.pulse_type, phase = 90 + self.halfpicounter1*self.half_pi_offset)
            #<YZ>
            if  self.expt_cfg["measure_state"] == 5:
                self.psb.append('q,mm'+str(self.id1),'pi_ge', phase = self.expt_cfg['final_offset'] + pt)
                self.psb.append('q','half_pi', self.pulse_type, phase = self.halfpicounter1*self.half_pi_offset)
            #<ZX>
            if  self.expt_cfg["measure_state"] == 6:
                self.psb.append('q,mm'+str(self.id2),'pi_ge', phase = self.expt_cfg['final_offset2'] + pt)
                self.psb.append('q','half_pi_y', self.pulse_type, phase = 90 + self.halfpicounter2*self.half_pi_offset)
            #<ZY>
            if  self.expt_cfg["measure_state"] == 7:
                self.psb.append('q,mm'+str(self.id2),'pi_ge', phase = self.expt_cfg['final_offset2'] + pt)
                self.psb.append('q','half_pi', self.pulse_type, phase = self.halfpicounter2*self.half_pi_offset)

            #<IX>
            if  self.expt_cfg["measure_state"] == 8:
                self.psb.append('q,mm'+str(self.id2),'pi_ge', phase = self.expt_cfg['final_offset_t'] + pt)
                self.psb.append('q','half_pi_y', self.pulse_type, phase = self.halfpicounter2*self.half_pi_offset)

            #<IY>
            if  self.expt_cfg["measure_state"] == 9:
                self.psb.append('q,mm'+str(self.id2),'pi_ge', phase = self.expt_cfg['final_offset_t'] + pt)
                self.psb.append('q','half_pi', self.pulse_type, phase = self.halfpicounter2*self.half_pi_offset)

            #<IZ>
            if  self.expt_cfg["measure_state"] == 10:
                self.psb.append('q,mm'+str(self.id2),'pi_ge', phase = self.expt_cfg['final_offset_t'] + pt)

            #<XI>
            if  self.expt_cfg["measure_state"] == 11:
                self.psb.append('q,mm'+str(self.id1),'pi_ge', phase = self.expt_cfg['final_offset_t'] + pt)
                self.psb.append('q','half_pi_y', self.pulse_type, phase = self.halfpicounter1*self.half_pi_offset)
            #<YI>
            if  self.expt_cfg["measure_state"] == 12:
                self.psb.append('q,mm'+str(self.id1),'pi_ge', phase = self.expt_cfg['final_offset_t'] + pt)
                self.psb.append('q','half_pi', self.pulse_type, phase = self.halfpicounter1*self.half_pi_offset)

             #<ZI>
            if  self.expt_cfg["measure_state"] == 13:
                self.psb.append('q,mm'+str(self.id1),'pi_ge', phase = self.expt_cfg['final_offset_t'] + pt)
                self.psb.append('q','half_pi', self.pulse_type, phase = self.halfpicounter1*self.half_pi_offset)



        # Tomography at a given time slice

            if self.expt_cfg["tomography"]:
                 ### gates before measurement for tomography
                if pt == 0:
                    # <X>
                    self.psb.append('q','half_pi', self.pulse_type)
                elif pt == 1:
                    # <Y>
                    self.psb.append('q','half_pi_y', self.pulse_type)
                elif pt == 2:
                    # <Z>
                    pass

class MultimodeCNOTSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.multimode_cfg = cfg['multimodes']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        if self.expt_cfg["tomography"]:
            self.expt_pts = np.array([0,1,2])
        else:
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
        self.idef = self.expt_cfg['idef']
        self.half_pi_offset = self.pulse_cfg[self.pulse_type]['offset_phase']
        self.halfpicounter1 = 0
        self.halfpicounter2 = 0



    def define_pulses(self,pt):

        if self.expt_cfg["tomography"]:
            par = self.expt_cfg["time_slice"]
        else:
            if self.expt_cfg["slice"]:
                par = self.expt_cfg["time_slice"]
            else:
                par = pt


        #State preparation
        if  self.expt_cfg["prepare_state"] == 0:

            self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'])
            self.psb.append('q,mm'+str(self.id1),'pi_ge')

        if  self.expt_cfg["prepare_state"] == 1:

            self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'])
            self.psb.append('q,mm'+str(self.id2),'pi_ge')

        if  self.expt_cfg["prepare_state"] == 2:

            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge')

        if  self.expt_cfg["prepare_state"] == 3:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'])
            self.psb.append('q,mm'+str(self.id2),'pi_ge')

        # <XX>

        if  self.expt_cfg["prepare_state"] == 4:
            self.halfpicounter1 = 0
            self.halfpicounter2 = 0

            self.psb.append('q','half_pi_y', self.pulse_type)
            self.halfpicounter1 +=1
            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q','half_pi_y', self.pulse_type)
            self.halfpicounter2 +=1
            self.psb.append('q,mm'+str(self.id2),'pi_ge')


        # <XY>
        if  self.expt_cfg["prepare_state"] == 5:
            self.halfpicounter1 = 0
            self.halfpicounter2 = 0

            self.psb.append('q','half_pi_y', self.pulse_type)
            self.halfpicounter1 +=1
            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q','half_pi', self.pulse_type, phase=0)
            self.halfpicounter2 +=1
            self.psb.append('q,mm'+str(self.id2),'pi_ge')

        # <YX>
        if  self.expt_cfg["prepare_state"] == 6:
            self.halfpicounter1 = 0
            self.halfpicounter2 = 0

            self.psb.append('q','half_pi', self.pulse_type)
            self.halfpicounter1 +=1
            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q','half_pi_y', self.pulse_type)
            self.halfpicounter2 +=1
            self.psb.append('q,mm'+str(self.id2),'pi_ge')

        # <YY>
        if  self.expt_cfg["prepare_state"] == 7:
            self.halfpicounter1 = 0
            self.halfpicounter2 = 0

            self.psb.append('q','half_pi', self.pulse_type)
            self.halfpicounter1 +=1
            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q','half_pi', self.pulse_type)
            self.halfpicounter2 +=1
            self.psb.append('q,mm'+str(self.id2),'pi_ge')

         # <ZZ>
        if  self.expt_cfg["prepare_state"] == 8:
            self.halfpicounter1 = 0
            self.halfpicounter2 = 0

            self.psb.append('q','pi')
            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q','pi')
            self.psb.append('q,mm'+str(self.id2),'pi_ge')

        #phi bell
        if  self.expt_cfg["prepare_state"] == 9:
            self.halfpicounter1 = 0
            self.halfpicounter2 = 0
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= self.expt_cfg['time_slice'], phase=180)
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id2),'pi_ge', phase=180)

        #psi bell
        if  self.expt_cfg["prepare_state"] == 10:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id1),'general', self.flux_pulse_type, amp= self.multimode_cfg[int(self.id1)]['a'], length= self.expt_cfg['time_slice'])
            self.psb.append('q,mm'+str(self.id2),'pi_ge')

        # CNOT Gate

        if self.expt_cfg["cnot_on"]:

            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            if self.expt_cfg['sweep_ef_phase']:
                phase_temp = self.expt_cfg['pi_ef_offset'] + pt
            else:
                phase_temp = self.expt_cfg['pi_ef_offset']
            self.psb.append('q','pi_q_ef', phase=phase_temp )
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=180)

        elif self.expt_cfg["cy_on"]:

            self.psb.append('q,mm'+str(self.id1),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            if self.expt_cfg['sweep_ef_phase']:
                phase_temp = self.expt_cfg['pi_ef_offset'] + pt
            else:
                phase_temp = self.expt_cfg['pi_ef_offset']
            self.psb.append('q','pi_q_ef', phase=phase_temp + 90)
            self.psb.append('q,mm'+str(self.id2),'pi_ef')
            self.psb.append('q,mm'+str(self.id1),'pi_ge',phase=180)

        else:
            self.psb.idle(self.expt_cfg['no_cnot_idle'])

        #Reversing State preparation


        if self.expt_cfg['sweep_final_phase']:
            phase_temp_2 = self.expt_cfg['final_offset'] + pt

        else:
            phase_temp_2 = self.expt_cfg['final_offset']

        if  self.expt_cfg["measure_state"] == 0:
            self.psb.append('q,mm'+str(self.id2),'pi_ge', phase = phase_temp_2 )
            # self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'])

        if  self.expt_cfg["measure_state"] == 1:
            self.psb.append('q,mm'+str(self.id2),'pi_ge', phase =  phase_temp_2 )
            self.psb.append('q','general', self.pulse_type, amp=1, length=par, freq=self.expt_cfg['iq_freq'])

        if  self.expt_cfg["measure_state"] == 2:
            pass

        if  self.expt_cfg["measure_state"] == 3:
            self.psb.append('q,mm'+str(self.id2),'pi_ge', phase = 0)
 # <XX>
        if  self.expt_cfg["measure_state"] == 4:
            self.psb.append('q,mm'+str(self.id1),'pi_ge', phase=phase_temp_2  )
            self.psb.append('q','half_pi_y', self.pulse_type, phase= self.halfpicounter1*self.half_pi_offset  + 90)

 # <XY>
        if  self.expt_cfg["measure_state"] == 5:
            self.psb.append('q,mm'+str(self.id1),'pi_ge', phase=phase_temp_2 )
            self.psb.append('q','half_pi_y', self.pulse_type, phase= self.halfpicounter1*self.half_pi_offset  + 90)
 # <YX>
        if  self.expt_cfg["measure_state"] == 6:
            self.psb.append('q,mm'+str(self.id1),'pi_ge', phase=phase_temp_2 )
            self.psb.append('q','half_pi', self.pulse_type, phase= self.halfpicounter1*self.half_pi_offset)
 # <YY>
        if  self.expt_cfg["measure_state"] == 7:
            self.psb.append('q,mm'+str(self.id1),'pi_ge', phase=phase_temp_2 )
            self.psb.append('q','half_pi', self.pulse_type, phase= self.halfpicounter1*self.half_pi_offset)
 # <ZZ>
        if  self.expt_cfg["measure_state"] == 8:
            self.psb.append('q,mm'+str(self.id2),'pi_ge', phase=phase_temp_2 )


        # Tomography at a given time slice

        if self.expt_cfg["tomography"]:
             ### gates before measurement for tomography
            if pt == 0:
                # <X>
                self.psb.append('q','half_pi', self.pulse_type)
            elif pt == 1:
                # <Y>
                self.psb.append('q','half_pi_y', self.pulse_type)
            elif pt == 2:
                # <Z>
                pass

class MultimodePi_PiSequence(QubitPulseSequence):
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
        # self.psb.append('q','pi', self.pulse_type)
        # self.psb.append('q,mm'+str(4),'pi_ge')
        self.psb.append('q','half_pi', self.pulse_type)
        self.psb.append('q,mm'+str(self.id),'pi_ge')
        self.psb.append('q,mm'+str(self.id),'pi_ge',phase=pt)
        self.psb.append('q','half_pi', self.pulse_type, phase=self.offset_phase)


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
            self.psb.append('q','half_pi', self.pulse_type)

            cphase_v2(self.psb,'q',self.id,cz_dc_phase=pt, cz_phase = 0)

            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase)

        #Calibration of phase for second conditional sideband
        elif self.offset_exp == 1:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','half_pi', self.pulse_type)
            cphase_v2(self.psb,'q',self.id,cz_dc_phase=self.cz_dc_phase,cz_phase = pt)
            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase)

        # Verify that the phases works
        elif self.offset_exp == 2:
            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','half_pi', self.pulse_type)
            cphase_v2(self.psb,'q',self.id,cz_dc_phase= self.cz_dc_phase, cz_phase = self.multimode_cfg[self.id]['cz_phase'])
            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase + pt)

        # replace half pi by varying length
        elif self.offset_exp == 3:
            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a']
                            , length=pt,freq=self.pulse_cfg['gauss']['iq_freq'])
            cphase_v2(self.psb,'q',self.id,cz_dc_phase= self.cz_dc_phase, cz_phase = self.multimode_cfg[self.id]['cz_phase'])
            self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a'],
                            length=pt,freq=self.pulse_cfg['gauss']['iq_freq'], phase= self.offset_phase)

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

        self.cz_dc_phase = self.multimode_cfg[self.id]['cz_dc_phase']
        self.cz_phase = self.multimode_cfg[self.id]['cz_phase']

        self.mode_mode_cz_dc_phase = self.cfg['mode_mode_offset']['cz_dc_phase']
        self.mode_mode_cz_phase = self.cfg['mode_mode_offset']['cz_phase']



    def define_pulses(self,pt):
        #Calibration of DC offset due to the 2pi ge sideband
        if self.offset_exp == 0:
            self.psb.append('q','half_pi', self.pulse_type)

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            cphase_v2(self.psb,'q',self.id,cz_dc_phase= self.cz_dc_phase + pt, cz_phase = self.cz_phase)

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])


            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase)

                #Calibration of phase for second conditional sideband
        elif self.offset_exp == 1:
            self.psb.append('q','pi', self.pulse_type)
            self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','half_pi', self.pulse_type)

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            cphase_v2(self.psb,'q',self.id
                      ,cz_dc_phase=self.cz_dc_phase + self.mode_mode_cz_dc_phase[self.id][self.id2]
                      ,cz_phase = self.cz_phase+pt)

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase)

        # Verify that the phases works
        elif self.offset_exp == 2:
            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','half_pi', self.pulse_type)

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            cphase_v2(self.psb,'q',self.id
                      ,cz_dc_phase=self.cz_dc_phase + self.mode_mode_cz_dc_phase[self.id][self.id2]
                      ,cz_phase = self.cz_phase+self.mode_mode_cz_phase[self.id][self.id2])

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            self.psb.append('q','half_pi', self.pulse_type, phase= self.offset_phase + pt)

        # replace half pi by varying length
        elif self.offset_exp == 3:
            if self.load_photon:
                self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q,mm'+str(self.id),'pi_ge')

            self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a']
                            , length=pt,freq=self.pulse_cfg['gauss']['iq_freq'])

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            cphase_v2(self.psb,'q',self.id
                      ,cz_dc_phase=self.cz_dc_phase + self.mode_mode_cz_dc_phase[self.id][self.id2]
                      ,cz_phase = self.cz_phase+self.mode_mode_cz_phase[self.id][self.id2])

            self.psb.append('q,mm'+str(self.id2),'pi_ge')
            self.psb.append('q,mm'+str(self.id2),'pi_ge',phase=self.multimode_cfg[self.id2]['pi_pi_offset_phase'])

            self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg['gauss']['a'],
                            length=pt,freq=self.pulse_cfg['gauss']['iq_freq'], phase= self.offset_phase)


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
        self.halfpicounter1 = 0
        self.halfpicounter2 = 0

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

        self.psb.append('q','half_pi', self.pulse_type)
        self.psb.append('q,mm'+str(self.id1),'pi_ge')

        if self.number > 0:
            for i in arange(self.number):
                self.psb.append('q,mm'+str(self.id1),'pi_ge', phase = self.multimode_cfg[self.id1]['pi_pi_offset_phase'])
                self.psb.append('q,mm'+str(self.id2),'pi_ef')
                self.psb.append('q','pi_q_ef')
                self.psb.append('q,mm'+str(self.id2),'pi_ef')
                self.psb.append('q,mm'+str(self.id1),'pi_ge',phase= self.multimode_cfg[self.id2]['ef_pi_pi_offset_phase'])
        if self.number%2==0:
            self.psb.append('q,mm'+str(self.id1),'pi_ge', phase = pt+ self.multimode_cfg[self.id1]['pi_pi_offset_phase'])
            self.psb.append('q','half_pi', self.pulse_type, phase = self.offset_phase)
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

        # 'exp' == 1 and 'exp' == 2 (drive at Chi shifted frequency) corresponding to qubit rabi with photon loaded

        if self.extra_args['exp']==1 or self.extra_args['exp']== 2 or self.extra_args['exp'] == 6:
            self.expt_pts =arange(self.expt_cfg['start_rabi'], self.expt_cfg['stop_rabi'], self.expt_cfg['step_rabi'])

        # 'exp' == 3 and 'exp' == 4 (drive at Chi shifted frequency): shift in offset phase of qubit due to photon loaded

        elif self.extra_args['exp']==3 or self.extra_args['exp']== 4 or self.extra_args['exp']== 5:

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


        self.exp = self.extra_args['exp']

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