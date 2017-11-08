__author__ = 'Nelson'

from slab.instruments.awg.PulseSequence import *
from numpy import arange, linspace
from slab.experiments.ExpLib.PulseSequenceBuilder import *
from slab.experiments.ExpLib.QubitPulseSequence import *
import random
from liveplot import LivePlotClient


class RabiSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.pulse_cfg = cfg['pulse_info']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']

    def define_pulses(self,pt):
        if self.expt_cfg['sweep_amp']:
            self.psb.append('q','general', self.pulse_type, amp=pt, length=self.expt_cfg['length'],freq=self.expt_cfg['iq_freq'],phase=self.expt_cfg['phase'])
        else:
            self.psb.append('q', 'general', self.pulse_type, amp=self.pulse_cfg[self.pulse_type]['a'], length=pt, freq=self.expt_cfg['iq_freq'], phase=self.expt_cfg['phase'])

class PulseProbeIQSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.pulse_cfg = cfg['pulse_info']
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])
        self.length = self.expt_cfg['pulse_probe_len']

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']

        if 'amp' in self.extra_args:
            self.a = self.extra_args['amp']
        else:
            self.a = self.expt_cfg['a']


    def define_pulses(self,pt):
        self.psb.append('q', 'general', self.pulse_type, amp=self.a, length=self.length, freq=self.pulse_cfg[self.expt_cfg['pulse_type']]['iq_freq'] + pt)

class BlueSidebandRabisequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.pulse_cfg = cfg['pulse_info']
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])
        self.iq_center = self.expt_cfg['iq_freq_center']
        self.length = self.expt_cfg['pulse_probe_len']

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']

        if 'amp' in self.extra_args:
            self.a = self.extra_args['amp']
        else:
            self.a = self.expt_cfg['a']


    def define_pulses(self,pt):
        self.psb.append('q', 'general', self.pulse_type, amp=self.a, length=pt, freq=self.pulse_cfg[self.expt_cfg['pulse_type']]['iq_freq'] + self.expt_cfg['det_blue'])


class PulseProbeEFIQSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.pulse_cfg = cfg['pulse_info']
        self.alpha = cfg['qubit']['alpha']
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])
        self.iq_center = self.expt_cfg['iq_freq_center']
        self.length = self.expt_cfg['pulse_probe_len']

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type_ge']
        self.pulse_type_ef = self.expt_cfg['pulse_type_ef']
        self.a = self.expt_cfg['a']


    def define_pulses(self,pt):
        self.psb.append('q', 'pi', self.pulse_type)
        self.psb.append('q', 'general', self.pulse_type_ef, amp=self.a, length=self.length, freq= self.pulse_cfg[self.pulse_type]['iq_freq'] + pt)

class RamseySequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.pulse_cfg = cfg['pulse_info']
        self.expt_cfg = expt_cfg
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg,self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']

    def define_pulses(self,pt):
        self.psb.append('q','half_pi', self.pulse_type)
        self.psb.idle(pt)
        self.psb.append('q','half_pi', self.pulse_type, phase = 360.0*self.expt_cfg['ramsey_freq']*pt/(1.0e9)+ self.pulse_cfg[self.pulse_type]['offset_phase'])

class SpinEchoSequence(QubitPulseSequence):
    def __init__(self,name,cfg, expt_cfg,**kwargs):
        self.pulse_cfg = cfg['pulse_info']
        self.expt_cfg = cfg['spin_echo']

        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)
        if 'number' in self.extra_args:
            self.number = self.extra_args['number']
        else:
            self.number = self.expt_cfg['number']

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.half_pi_offset = self.pulse_cfg[self.pulse_type]['offset_phase']

    def define_pulses(self,pt):

        self.psb.append('q','half_pi', self.pulse_type)
        for i in arange(self.number):
            self.psb.idle(pt/(float(2*self.number)))
            if self.expt_cfg['CP']:
                self.psb.append('q','pi', self.pulse_type)
            elif self.expt_cfg['CPMG']:
                self.psb.append('q','pi', self.pulse_type, phase=90)
            self.psb.idle(pt/(float(2*self.number)))
        self.psb.append('q','half_pi', self.pulse_type, phase= self.half_pi_offset + 360.0*self.expt_cfg['ramsey_freq']*pt/(1.0e9))



class EFRabiSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.extra_args={}

        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
        if 'ge_pi' in self.extra_args:
            self.ge_pi = self.extra_args['ge_pi']
        else:
            self.ge_pi = cfg['ef_rabi']['ge_pi']

            #print str(key) + ": " + str(value)
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)



    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        #self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq'] + self.qubit_cfg['alpha']

    def define_pulses(self,pt):
        if self.ge_pi:
           self.psb.append('q','pi', self.pulse_type)
        self.psb.append('q','general', self.ef_pulse_type, amp=self.pulse_cfg[self.expt_cfg['ef_pulse_type']]['pi_ef_a'], length=pt,freq=self.ef_sideband_freq)
        self.psb.append('q','pi', self.pulse_type)
        if self.expt_cfg['ef_cal']:
           self.psb.append('q', 'pi_q_ef', self.pulse_type)


class EFRamseySequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        QubitPulseSequence.__init__(self, name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)

    def define_pulses(self,pt):
        if self.expt_cfg['ge_pi']:
            self.psb.append('q','pi', self.pulse_type)
        self.psb.append('q','general', self.ef_pulse_type,amp=self.pulse_cfg[self.expt_cfg['ef_pulse_type']]['half_pi_ef_a'],length = self.pulse_cfg[self.expt_cfg['ef_pulse_type']]['half_pi_ef_length'], freq=self.ef_sideband_freq )
        self.psb.idle(pt)
        self.psb.append('q','general', self.ef_pulse_type,amp=self.pulse_cfg[self.expt_cfg['ef_pulse_type']]['half_pi_ef_a'],length = self.pulse_cfg[self.expt_cfg['ef_pulse_type']]['half_pi_ef_length'],freq=self.ef_sideband_freq,phase =  + 360.0*self.expt_cfg['ramsey_freq']*pt/(1.0e9))
        self.psb.append('q','pi', self.pulse_type)
        self.psb.append('q','pi_ef', self.ef_pulse_type)

class EFT1Sequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters,self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.ef_pulse_type = self.expt_cfg['ef_pulse_type']
        ef_freq = self.qubit_cfg['frequency']+self.qubit_cfg['alpha']
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)

    def define_pulses(self,pt):
        if self.expt_cfg['ge_pi']:
            self.psb.append('q','pi', self.pulse_type)
        self.psb.append('q','general', self.ef_pulse_type,amp=self.expt_cfg['a'],length = self.expt_cfg['pi_ef'], freq=self.ef_sideband_freq )
        self.psb.idle(pt)
        self.psb.append('q','pi', self.pulse_type)



class T1Sequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg,self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']

    def define_pulses(self,pt):
        # self.psb.append('q','general', 'square', amp=1, length=16,freq=150e6)
        self.psb.append('q','pi', self.pulse_type)
        self.psb.idle(pt)

class QPPumpingSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg,self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']

    def define_pulses(self,pt):

        #pi pulse qubit and idle for a quasiparticle lifetime, repeat n times
        for ii in range(self.expt_cfg['number_pump_pulses']):
            self.psb.append('q', 'pi', self.pulse_type)
            self.psb.idle(self.expt_cfg['pump_delay']) #change idle time to QP lifetime
        self.psb.append('q','pi', self.pulse_type)
        self.psb.idle(pt)

class T1rhoSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)
        if 'amp' in self.extra_args:
            self.amp = self.extra_args['amp']
        else:
            self.amp = 1.0#self.pulse_cfg['gauss']['a']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg,self.define_points, self.define_parameters, self.define_pulses)


    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']

    def define_pulses(self,pt):

        if self.expt_cfg['echo']:

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q','pi', self.pulse_type,phase=90+self.pulse_cfg[self.pulse_type]['offset_phase'])
            self.psb.append('q','general', 'square', amp=self.amp, length=pt,freq=self.pulse_cfg[self.pulse_type]['iq_freq'],phase=90.0 + 2*self.pulse_cfg[self.pulse_type]['offset_phase'])
            self.psb.append('q','pi', self.pulse_type,phase=90+3*self.pulse_cfg[self.pulse_type]['offset_phase'])
            self.psb.append('q','half_pi', self.pulse_type, phase = +4*self.pulse_cfg[self.pulse_type]['offset_phase'] )

        else:

            self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q','general', 'square', amp=self.amp, length=pt,freq=self.pulse_cfg[self.pulse_type]['iq_freq'],phase=90.0 + self.pulse_cfg[self.pulse_type]['offset_phase'])
            self.psb.append('q','half_pi', self.pulse_type, phase = +2*self.pulse_cfg[self.pulse_type]['offset_phase'] )


class HalfPiXOptimizationSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.pulse_cfg = cfg['pulse_info']
        self.expt_cfg = expt_cfg
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg,self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

        # Number of pi/2 pulse

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']

    def define_pulses(self,pt):

        for i in arange(pt):
            self.psb.append('q','half_pi', self.pulse_type)
        if pt%4 ==1:
            self.psb.append('q', 'half_pi', self.pulse_type)
        elif pt%4 == 2:
            pass
        elif pt%4 == 3:
            self.psb.append('q', 'half_pi', self.pulse_type,phase=180.0)
        else:
            self.psb.append('q', 'pi', self.pulse_type)

        # for i in arange(22):
        #     self.psb.append('q', 'general', self.pulse_type, amp=pt, length=self.pulse_cfg['gauss']['half_pi_length'],freq=self.pulse_cfg['gauss']['iq_freq'])



class HalfPiXOptimizationSweepSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.pulse_cfg = cfg['pulse_info']
        self.expt_cfg = expt_cfg
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)
        self.pulse_length = self.extra_args['pulse_length']
        self.pulse_amp = self.extra_args['pulse_amp']

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg,self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']

    def define_pulses(self,pt):
        n = 2*pt+1
        i = 0
        while i< n:
            self.psb.append('q','general', self.pulse_type, amp=self.pulse_amp, length=self.pulse_length,freq=self.pulse_cfg[self.pulse_type]['iq_freq'], phase= i*self.expt_cfg['phase'])
            i += 1


class PiXOptimizationSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.pulse_cfg = cfg['pulse_info']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg,self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']

    def define_pulses(self,pt):
        for i in arange(pt):
            # self.psb.append('q', 'half_pi', self.pulse_type, phase=0)
            # self.psb.append('q', 'half_pi', self.pulse_type, phase=self.pulse_cfg['gauss']['phase'])
            self.psb.append('q', 'pi', self.pulse_type)


        if pt%2 ==1:
            pass
        else:
            # self.psb.append('q', 'half_pi', self.pulse_type, phase=0)
            # self.psb.append('q', 'half_pi', self.pulse_type, phase=self.pulse_cfg['gauss']['phase'])
            self.psb.append('q', 'pi', self.pulse_type)

        # for i in arange(21):
        #     self.psb.append('q', 'general', self.pulse_type, amp=pt, length=self.pulse_cfg['gauss']['pi_length'],freq=self.pulse_cfg['gauss']['iq_freq'])

class PiXoptimizationChangeAmpSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.pulse_cfg = cfg['pulse_info']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg,self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']

    def define_pulses(self,pt):

        for i in arange(21):
            self.psb.append('q', 'general', self.pulse_type, amp=pt, length=self.pulse_cfg['gauss']['pi_length'],freq=self.pulse_cfg['gauss']['iq_freq'])



class HalfPiYPhaseOptimizationSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.pulse_cfg = cfg['pulse_info']
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)
        if 'qubit_dc_offset' in self.extra_args:
            self.qubit_dc_offset = self.extra_args['qubit_dc_offset']
            self.pulse_cfg['qubit_dc_offset_pi'] =  self.qubit_dc_offset
            self.pulse_cfg['fix_phase'] =  True
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg,self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']


    def define_pulses(self,pt):
        # for i in arange(21):
        #     self.psb.append('q','half_pi', self.pulse_type, phase=0)
        # self.psb.append('q','half_pi', self.pulse_type,phase=pt)
        self.psb.append('q','half_pi', self.pulse_type, phase=0)
        # self.psb.append('q','pi', self.pulse_type,phase=pt)
        self.psb.append('q','half_pi', self.pulse_type,phase=self.pulse_cfg['gauss']['phase'] + pt)

class TomographySequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.pulse_cfg = cfg['pulse_info']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg,self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = np.array([0,1,2])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']

    def define_pulses(self,pt):
        ### Initiate states
        self.psb.append('q','pi', self.pulse_type)
        self.psb.idle(600)
        # self.psb.append('q','general', "square", amp=1.0, length=1200,freq=125e6)

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

class PiXOptimizationSweepSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.pulse_cfg = cfg['pulse_info']
        self.expt_cfg = expt_cfg
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)
        self.pulse_length = self.extra_args['pulse_length']
        self.pulse_amp = self.extra_args['pulse_amp']

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg,self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']

    def define_pulses(self,pt):
        # n = pt
        # i = 0
        # self.psb.append('q','half_pi', self.pulse_type)
        # while i< n:
        #     self.psb.append('q','general', self.pulse_type, amp=self.pulse_amp, length=self.pulse_length,freq=self.pulse_cfg[self.pulse_type]['iq_freq'])
        #     i += 1
        for i in arange(15):
            self.psb.append('q', 'pi', self.pulse_type)
        if pt%2 ==1:
            pass
        else:
            self.psb.append('q', 'pi', self.pulse_type)

class RabiSweepSequence(QubitPulseSequence):

    def __init__(self,name, cfg, expt_cfg, **kwargs):
        self.pulse_cfg = cfg['pulse_info']

        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses, **kwargs)


    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']

    def define_pulses(self,pt):
        self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=pt,freq=self.expt_cfg['iq_freq'])


class RabiRamseyT1FluxSweepSequence(QubitPulseSequence):

    def __init__(self,name, cfg, expt_cfg, **kwargs):
        self.pulse_cfg = cfg['pulse_info']
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)
        self.exp = self.extra_args['exp']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses, **kwargs)


    def define_points(self):
        self.expt_pts = arange(self.expt_cfg[self.exp]['start'], self.expt_cfg[self.exp]['stop'], self.expt_cfg[self.exp]['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']

    def define_pulses(self,pt):
        if self.exp == 'rabi' or self.exp == 'rabi_long':
            self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=pt,freq=self.expt_cfg['iq_freq'])
        elif self.exp =='ef_rabi':
            self.ef_sideband_freq = self.expt_cfg['iq_freq']+self.extra_args['alpha']
            self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=self.extra_args['pi_length'],freq=self.expt_cfg['iq_freq'])
            self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=pt,freq=self.ef_sideband_freq)
            self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=self.extra_args['pi_length'],freq=self.expt_cfg['iq_freq'])
        elif self.exp == 'ramsey' or self.exp == 'ramsey_long':
            self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=self.extra_args['half_pi_length'],freq=self.expt_cfg['iq_freq'])
            self.psb.idle(pt)
            self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=self.extra_args['half_pi_length'],freq=self.expt_cfg['iq_freq'], phase = 360.0*self.expt_cfg[self.exp]['ramsey_freq']*pt/(1.0e9))
        elif self.exp == 'ef_ramsey' or self.exp == 'ef_ramsey_long':
            self.ef_sideband_freq = self.expt_cfg['iq_freq']+self.extra_args['alpha']
            self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=self.extra_args['pi_length'],freq=self.expt_cfg['iq_freq'])
            self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=self.extra_args['ef_half_pi_length'],freq=self.ef_sideband_freq)
            self.psb.idle(pt)
            self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=self.extra_args['ef_half_pi_length'],freq=self.ef_sideband_freq, phase = 360.0*self.expt_cfg[self.exp]['ramsey_freq']*pt/(1.0e9))
            self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=self.extra_args['pi_length'],freq=self.expt_cfg['iq_freq'])
        elif self.exp == 'ef_t1':
            self.ef_sideband_freq = self.expt_cfg['iq_freq']+self.extra_args['alpha']
            self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=self.extra_args['pi_length'],freq=self.expt_cfg['iq_freq'])
            self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=self.extra_args['ef_pi_length'],freq=self.ef_sideband_freq)
            self.psb.idle(pt)
            self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=self.extra_args['pi_length'],freq=self.expt_cfg['iq_freq'])
        elif self.exp == 't1':
            self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=self.extra_args['pi_length'],freq=self.expt_cfg['iq_freq'])
            self.psb.idle(pt)
        elif self.exp == 'half_pi_phase_sweep':
            self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=self.extra_args['half_pi_length'],freq=self.expt_cfg['iq_freq'])
            self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=self.extra_args['half_pi_length'],freq=self.expt_cfg['iq_freq'], phase = pt)

class RandomizedBenchmarkingPhaseOffsetSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.pulse_cfg = cfg['pulse_info']
        self.expt_cfg = expt_cfg
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg,self.define_points, self.define_parameters, self.define_pulses)

    def expmat(self, mat, theta):
        return np.cos(theta/2)*self.I - 1j*np.sin(theta/2)*mat

    def R_q(self,theta,phi):
        return np.cos(theta/2.0)*self.I -1j*np.sin(theta/2.0)*(np.cos(phi)*self.X + np.sin(phi)*self.Y)

    def define_points(self):
        if self.expt_cfg['knill_length_list']:
            self.expt_pts = np.array([2,3,4,5,6,8,10,12,16,20,24,32,40,48,64,80,96, 128, 160, 192])
        else:
            self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
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
        #
        # self.random_cliffords_1 =  np.concatenate((np.array([0]),0*np.ones(max(self.expt_pts)-1)),axis=0).astype(int)
        # self.random_cliffords_2 =  np.concatenate((np.array([1]),1*np.ones(max(self.expt_pts)-1)),axis=0).astype(int)

        print [self.clifford_pulse_1_list[jj] for jj in self.random_cliffords_1]
        print [self.clifford_pulse_2_list[jj] for jj in self.random_cliffords_2]


    def define_pulses(self,pt):
        self.n = pt

        R = self.I
        self.znumber=0
        self.xnumber=0
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
            R = np.dot(C,R)
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
                        print "---" + str(self.n)
                        print "Number of z pulses in creation sequence %s" %(self.znumber)
                        print self.clifford_inv_pulse_1_list[ii]
                        print self.clifford_inv_pulse_2_list[jj]
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

        print "Total number of half pi pulses = %s"%(self.xnumber)


class RandomizedBenchmarkingSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.pulse_cfg = cfg['pulse_info']
        self.expt_cfg = expt_cfg
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg,self.define_points, self.define_parameters, self.define_pulses)

    def expmat(self, mat, theta):
        return np.cos(theta/2)*self.I - 1j*np.sin(theta/2)*mat

    def R_q(self,theta,phi):
        return np.cos(theta/2.0)*self.I -1j*np.sin(theta/2.0)*(np.cos(phi)*self.X + np.sin(phi)*self.Y)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        self.clifford_pulse_1_list = ['0','half_pi_y','pi_y','neg_half_pi_y']
        self.clifford_pulse_2_list = ['0','half_pi','pi','neg_half_pi','half_pi_z','neg_half_pi_z']
        # self.clifford_pulse_1_list = ['half_pi','neg_half_pi','half_pi_y','neg_half_pi_y']
        # self.clifford_pulse_2_list = ['pi','pi_y']


        self.clifford_inv_pulse_1_list = ['0','neg_half_pi_y','pi_y','half_pi_y']
        self.clifford_inv_pulse_2_list = ['0','neg_half_pi','pi','half_pi','neg_half_pi_z','half_pi_z']

        ## Clifford and Pauli operators
        self.I = np.array([[1,0],[0,1]])
        self.X = np.array([[0,1],[1,0]])
        self.Y = np.array([[0,-1j],[1j,0]])
        self.Z = np.array([[1,0],[0,-1]])

        self.Pauli = [self.I,self.X,self.Y,self.Z]

        self.P_gen = np.empty([len(self.clifford_inv_pulse_1_list),2,2],dtype=np.complex64)
        self.C_gen = np.empty([len(self.clifford_inv_pulse_2_list),2,2],dtype=np.complex64)

        self.P_gen[0] = self.I
        self.P_gen[1] = self.expmat(self.Y,np.pi/2)
        self.P_gen[2] = self.expmat(self.Y,np.pi)
        self.P_gen[3] = self.expmat(self.Y,-np.pi/2)

        clist1 = [0,1,1,1,3,3] # index of IXYZ
        clist2 = [0, np.pi/2,np.pi,-np.pi/2,np.pi/2,-np.pi/2]

        for i in arange(len(self.clifford_inv_pulse_2_list)):
            self.C_gen[i] = self.expmat(self.Pauli[clist1[i]],clist2[i])

        self.random_cliffords_1 = [random.randint(0,len(self.clifford_pulse_1_list)-1) for r in range(len(self.expt_pts))]
        # self.random_cliffords_1 = 2*np.ones(len(self.expt_pts)).astype(int)
        self.random_cliffords_2 = [random.randint(0,len(self.clifford_pulse_2_list)-1) for r in range(len(self.expt_pts))]
        # self.random_cliffords_2 = 2*np.ones(len(self.expt_pts)).astype(int)### no z pulse
        # self.random_cliffords_2 = [random.randint(0,4) for r in range(len(self.expt_pts))] ### no z pulse

        print [self.clifford_pulse_1_list[jj] for jj in self.random_cliffords_1]
        print [self.clifford_pulse_2_list[jj] for jj in self.random_cliffords_2]


    def final_pulse_dictionary(self,R_input):
        g_e_random = random.randint(0,1)


        found = 0


        for zz in range(4):
            R = ((1j)**zz)*R_input
            for ii in range(len(self.clifford_inv_pulse_1_list)):
                for jj in range(len(self.clifford_inv_pulse_2_list)):
                    C1 = self.P_gen[ii]
                    C2 = self.C_gen[jj]
                    C = np.dot(C2,C1)


                    if np.allclose(np.real(R),np.real(C)) and np.allclose(np.imag(R),np.imag(C)):
                        found +=1

                        print "---" + str(self.n)
                        print self.clifford_inv_pulse_2_list[jj]
                        print self.clifford_inv_pulse_1_list[ii]

                        if jj == 4:
                            self.psb.append('q','neg_half_pi', self.pulse_type)
                            self.psb.append('q','neg_half_pi_y', self.pulse_type)
                            self.psb.append('q','half_pi', self.pulse_type)
                        elif jj == 5:
                            self.psb.append('q','neg_half_pi', self.pulse_type)
                            self.psb.append('q','half_pi_y', self.pulse_type)
                            self.psb.append('q','half_pi', self.pulse_type)
                        else:
                            self.psb.append('q',self.clifford_inv_pulse_2_list[jj], self.pulse_type)


                        self.psb.append('q',self.clifford_inv_pulse_1_list[ii], self.pulse_type)

        if found == 0 :
            print "Error! Some pulse's inverse was not found."
        elif found > 1:
            print "Error! Non unique inverse."


    def define_pulses(self,pt):
        self.n = pt
        # random_cliffords_1 = [random.randint(0,len(self.clifford_pulse_1_list)-1) for r in range(n)]
        # random_cliffords_2 = [random.randint(0,len(self.clifford_pulse_2_list)-1) for r in range(n)]
        R = self.I
        for jj in range(self.n):
            C1 = self.P_gen[self.random_cliffords_1[jj]]
            self.psb.append('q',self.clifford_pulse_1_list[self.random_cliffords_1[jj]], self.pulse_type)
            C2 = self.C_gen[self.random_cliffords_2[jj]]
            if self.random_cliffords_2[jj] == 4:
                self.psb.append('q','neg_half_pi', self.pulse_type)
                self.psb.append('q','half_pi_y', self.pulse_type)
                self.psb.append('q','half_pi', self.pulse_type)
            elif self.random_cliffords_2[jj] == 5:
                self.psb.append('q','neg_half_pi', self.pulse_type)
                self.psb.append('q','neg_half_pi_y', self.pulse_type)
                self.psb.append('q','half_pi', self.pulse_type)
            else:
                self.psb.append('q',self.clifford_pulse_2_list[self.random_cliffords_2[jj]], self.pulse_type)

            C = np.dot(C2,C1)
            R = np.dot(C,R)
        self.final_pulse_dictionary(R)