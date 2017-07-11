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
        self.cfg = cfg
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']

    def define_pulses(self,pt):
        if self.expt_cfg['sweep_amp']:
            self.psb.append('q','general', self.pulse_type, amp=pt, length=self.expt_cfg['length'],freq=self.expt_cfg['iq_freq'])
        else:
            self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=pt,freq=self.expt_cfg['iq_freq'])
            # self.psb.append('q2', 'general', 'square', amp=0.5, length=self.cfg['readout']['width']+1000,
            #                 freq=self.cfg['readout']['heterodyne_freq'],delay=(self.cfg['readout']['width']+1000)/2)


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
        # self.psb.append('q','general', "square", amp=1.0, length=pt,freq=-200e6)
        self.psb.append('q','half_pi', self.pulse_type, phase = 360.0*self.expt_cfg['ramsey_freq']*pt/(1.0e9))

        # # Does not work because of introducing phase in definition of half_pi
        #
        # self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg[self.pulse_type]['half_pi_a'], length=self.pulse_cfg[self.pulse_type]['half_pi_length'],freq=self.pulse_cfg[self.pulse_type]['iq_freq'], phase= 0)
        # self.psb.idle(pt)
        # self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg[self.pulse_type]['half_pi_a'], length=self.pulse_cfg[self.pulse_type]['half_pi_length'],freq=self.pulse_cfg[self.pulse_type]['iq_freq'], phase= self.pulse_cfg[self.pulse_type]['phase']+ 360.0*self.expt_cfg['ramsey_freq']*pt/(1.0e9))


class SpinEchoSequence(QubitPulseSequence):
    def __init__(self,name,cfg, expt_cfg,**kwargs):
        self.pulse_cfg = cfg['pulse_info']
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']
        # self.half_pi_offset = self.pulse_cfg[self.pulse_type]['offset_phase']
        self.half_pi_offset = 0.0
    def define_pulses(self,pt):

        self.psb.append('q','half_pi', self.pulse_type)
        for i in arange(self.expt_cfg["number"]):
            self.psb.idle(pt/(float(2*self.expt_cfg["number"])))
            if self.expt_cfg['CP']:
                # self.psb.append('q','pi', self.pulse_type)
                self.psb.append('q', 'half_pi', self.pulse_type, phase=0)
                self.psb.append('q', 'half_pi', self.pulse_type, phase=self.pulse_cfg['gauss']['phase'])
            elif self.expt_cfg['CPMG']:
                self.psb.append('q','pi', self.pulse_type, phase=90)
            self.psb.idle(pt/(float(2*self.expt_cfg["number"])))
        self.psb.append('q','half_pi', self.pulse_type, phase= self.half_pi_offset + 360.0*self.expt_cfg['ramsey_freq']*pt/(1.0e9))


class EFRabiSequence(QubitPulseSequence):
    def __init__(self,name, cfg, expt_cfg,**kwargs):
        self.qubit_cfg = cfg['qubit']
        self.pulse_cfg = cfg['pulse_info']
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
        if self.expt_cfg['ge_pi']:
           self.psb.append('q','pi', self.pulse_type)
        self.psb.append('q','general', self.ef_pulse_type, amp=self.pulse_cfg[self.expt_cfg['ef_pulse_type']]['pi_ef_a'], length=pt,freq=self.ef_sideband_freq)
        #self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=pt,freq=self.ef_sideband_freq)
        #self.psb.idle(4*pt)

        if self.expt_cfg['end_ge_pi_swap']:
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
        # self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq + self.expt_cfg['ramsey_freq'])
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq'] + self.qubit_cfg['alpha']

    def define_pulses(self,pt):
        if self.expt_cfg['ge_pi']:
            self.psb.append('q','pi', self.pulse_type)
        #self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=pt,freq=self.ef_sideband_freq)
        self.psb.append('q','general', self.ef_pulse_type, amp=self.pulse_cfg[self.expt_cfg['ef_pulse_type']]['half_pi_ef_a'],\
                        length = self.pulse_cfg[self.expt_cfg['ef_pulse_type']]['half_pi_ef_length'], freq=self.ef_sideband_freq )
        self.psb.idle(pt)
        self.psb.append('q','general', self.ef_pulse_type, amp=self.pulse_cfg[self.expt_cfg['ef_pulse_type']]['half_pi_ef_a'],\
                        length = self.pulse_cfg[self.expt_cfg['ef_pulse_type']]['half_pi_ef_length'], freq=self.ef_sideband_freq, \
                        phase=360.0 * self.expt_cfg['ramsey_freq'] * pt / (1.0e9))
        self.psb.append('q','pi', self.pulse_type)

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


class RabiThermalizerSequence(QubitPulseSequence):
    def __init__(self, name, cfg, expt_cfg, **kwargs):
        self.pulse_cfg = cfg['pulse_info']
        self.cfg = cfg
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg, self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):

        if self.expt_cfg['vary_by_list']:
            self.expt_pts_value = array(self.expt_cfg['vary_list'])
        else:
            self.expt_pts_value = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

        # pts are idx here
        self.expt_pts = arange(0,len(self.expt_pts_value),1)

    def define_parameters(self):

        #used for cal pulses
        self.pulse_type =  self.expt_cfg['pulse_type']

    def define_pulses(self, pt, isFlux = False):

        # pt: idx, not actual value
        if pt < len(self.expt_pts):
            self.add_thermalizer_pulses(pt, isFlux)
        else:
            cal_pt = pt - len(self.expt_pts)
            self.add_cal_pulses(cal_pt, isFlux)

    def add_thermalizer_pulses(self, pt, isFlux):

        # all pulses need to have target 'flux_x' (so cannot use idle)
        # because psb.build is modified to have the flux channels independent

        # update expt_cfg for this pt:
        self.expt_cfg[self.expt_cfg['vary_param']] = self.expt_pts_value[pt]

        expt_cfg = self.expt_cfg
        pulse_cfg = self.cfg['pulse_info']
        flux_cfg = self.cfg['flux_pulse_info']

        #todo: figure out start_end_buffer?
        hw_delay = flux_cfg['pxdac_hw_delay']

        thermalizer_1_length = expt_cfg["drive1_length"]
        thermalizer_2_length = expt_cfg["drive2_length"]
        thermalizer_1_dur = ap.get_pulse_span_length(pulse_cfg, expt_cfg['thermalizer_pulse_type'], thermalizer_1_length)
        thermalizer_2_dur = ap.get_pulse_span_length(pulse_cfg, expt_cfg['thermalizer_pulse_type'], thermalizer_2_length)

        flux_settle_dur = flux_cfg['flux_settle_dur']
        flux_settle_long_dur = flux_cfg['flux_settle_long_dur']
        flux_ramp_dur = flux_cfg['flux_ramp_dur']
        comp_length_temp = flux_cfg['dc_comp_pulse_length']

        excite_idx = expt_cfg['excite_idx']
        cal_idx = expt_cfg['cal_idx']

        for ii in range(4):

            target = 'flux_' + str(ii + 1)

            flux_a_drive1 = expt_cfg['drive1_flux_a'][ii]
            flux_a_drive2 = expt_cfg['drive2_flux_a'][ii]
            flux_a_read = expt_cfg['read_flux_a'][ii]

            if expt_cfg['is_excite_ge']:
                flux_a_excite = expt_cfg['excite_ge_flux_a'][excite_idx][ii]
                excite_length = expt_cfg['excite_ge_length'][excite_idx]
            elif expt_cfg['is_excite_ef']:
                flux_a_excite = expt_cfg['excite_ef_flux_a'][excite_idx][ii]
                excite_length = expt_cfg['excite_ef_length'][excite_idx]
            else:
                flux_a_excite = 0.
                excite_length = 0.
            excite_type = self.pulse_type
            excite_dur = ap.get_pulse_span_length(pulse_cfg, excite_type, excite_length)

            read_dur = self.cfg['readout']['delay'] + self.cfg['readout']['width']

            # ramp to a_drive1
            flux_drive1_start_time = 0
            self.psb.append(target, 'general', 'logistic_ramp', start_amp=0, stop_amp=flux_a_drive1, length=flux_ramp_dur)
            self.psb.append(target, 'general', 'linear_ramp', start_amp=flux_a_drive1, stop_amp=flux_a_drive1, length=flux_settle_long_dur)
            # 1st thermalizer pulse
            thermalizer_1_start_time = flux_drive1_start_time + flux_ramp_dur + flux_settle_long_dur - hw_delay
            thermalizer_1_end_time = thermalizer_1_start_time + thermalizer_1_dur
            self.psb.append(target, 'general', 'linear_ramp', start_amp=flux_a_drive1, stop_amp=flux_a_drive1, length=thermalizer_1_dur)

            # ramp to a_excite for excitation pulses
            self.psb.append(target, 'general', 'logistic_ramp', start_amp=flux_a_drive1, stop_amp=flux_a_excite, length=flux_ramp_dur)
            self.psb.append(target, 'general', 'linear_ramp', start_amp=flux_a_excite, stop_amp=flux_a_excite, length=flux_settle_dur)
            # excite pulse
            excite_start_time = thermalizer_1_end_time + flux_ramp_dur + flux_settle_dur
            excite_end_time = excite_start_time + excite_dur
            self.psb.append(target, 'general', 'linear_ramp', start_amp=flux_a_excite, stop_amp=flux_a_excite, length=excite_dur)

            # ramp to a_drive2
            self.psb.append(target, 'general', 'logistic_ramp', start_amp=flux_a_excite, stop_amp=flux_a_drive2, length=flux_ramp_dur)
            self.psb.append(target, 'general', 'linear_ramp', start_amp=flux_a_drive2, stop_amp=flux_a_drive2, length=flux_settle_dur)
            # 2nd thermalizer pulse
            thermalizer_2_start_time = excite_end_time + flux_ramp_dur + flux_settle_dur
            thermalizer_2_end_time = thermalizer_2_start_time + thermalizer_2_dur
            self.psb.append(target, 'general', 'linear_ramp', start_amp=flux_a_drive2, stop_amp=flux_a_drive2, length=thermalizer_2_dur)

            # ramp to a_read
            self.psb.append(target, 'general', 'logistic_ramp', start_amp=flux_a_drive2, stop_amp=flux_a_read, length=flux_ramp_dur)
            self.psb.append(target, 'general', 'linear_ramp', start_amp=flux_a_read, stop_amp=flux_a_read, length=flux_settle_dur)
            # readout
            read_start_time = thermalizer_2_end_time + flux_ramp_dur + flux_settle_dur
            read_end_time = read_start_time + read_dur
            self.psb.append(target, 'general', 'linear_ramp', start_amp=flux_a_read, stop_amp=flux_a_read, length=read_dur)

            # ramp down to 0
            self.psb.append(target, 'general', 'logistic_ramp', start_amp=flux_a_read, stop_amp=0, length=flux_ramp_dur)
            self.psb.append(target, 'general', 'linear_ramp', start_amp=0, stop_amp=0, length=flux_settle_dur)

            comp_start_time = read_end_time + flux_ramp_dur + flux_settle_dur

            # get area of flux pulses
            flux_area = self.psb.get_total_flux_pulse_area(target = target)
            a_comp_temp = - flux_area/(comp_length_temp + flux_ramp_dur)
            a_comp = np.sign(a_comp_temp)* min(0.8, abs(a_comp_temp))
            if a_comp == 0:
                comp_length = 0.0
            else:
                comp_length =  - flux_area/a_comp - flux_ramp_dur

            # compensation pulse
            self.psb.append(target, 'general', 'logistic_ramp', start_amp=0, stop_amp=a_comp, length=flux_ramp_dur)
            self.psb.append(target, 'general', 'linear_ramp', start_amp=a_comp, stop_amp=a_comp, length=comp_length)
            self.psb.append(target, 'general', 'logistic_ramp', start_amp=a_comp, stop_amp=0, length=flux_ramp_dur)


        # if not flux, clear psb and add drive pulses
        if not isFlux:

            # clear psb
            dummy = self.psb.get_pulse_sequence()
            dummy = self.psb.get_total_pulse_span_length()
            dummy = self.psb.get_total_flux_pulse_span_length()

            # todo: check how are gauss/sq pulses aligned

            drive1_iq_freq = expt_cfg['drive1_freq'] - expt_cfg['drive_carrier_freq']
            drive2_iq_freq = expt_cfg['drive2_freq'] - expt_cfg['drive_carrier_freq']

            if expt_cfg['is_excite_ge']:
                excite_iq_freq = expt_cfg['excite_ge_freq'][excite_idx] - expt_cfg['drive_carrier_freq']
                excite_a = expt_cfg['excite_ge_a'][excite_idx]
            elif expt_cfg['is_excite_ef']:
                excite_iq_freq = expt_cfg['excite_ef_freq'][excite_idx] - expt_cfg['drive_carrier_freq']
                excite_a = expt_cfg['excite_ef_a'][excite_idx]
            else:
                excite_iq_freq = 0.
                excite_a = 0.

            if expt_cfg['separate_drive']:
                drive_target = 'q2'
                excite_target = 'q'
            else:
                drive_target = 'q'
                excite_target = 'q'

            #
            self.psb.idle(thermalizer_1_start_time)
            self.psb.append(drive_target, 'general', expt_cfg['thermalizer_pulse_type'],
                            amp=expt_cfg['drive1_a'], length=thermalizer_1_length, freq=drive1_iq_freq)

            self.psb.idle(excite_start_time - thermalizer_1_end_time)
            self.psb.append(excite_target, 'general', self.pulse_type,
                            amp=excite_a, length=excite_length, freq=excite_iq_freq)

            self.psb.idle(thermalizer_2_start_time - excite_end_time)
            self.psb.append(drive_target, 'general', expt_cfg['thermalizer_pulse_type'],
                            amp=expt_cfg['drive2_a'], length=thermalizer_2_length, freq=drive2_iq_freq)

    def add_cal_pulses(self, cal_pt, isFlux):

        expt_cfg = self.expt_cfg
        pulse_cfg = self.cfg['pulse_info']
        flux_cfg = self.cfg['flux_pulse_info']

        # todo: figure out start_end_buffer?
        hw_delay = flux_cfg['pxdac_hw_delay']

        flux_settle_dur = flux_cfg['flux_settle_dur']
        flux_settle_long_dur = flux_cfg['flux_settle_long_dur']
        flux_ramp_dur = flux_cfg['flux_ramp_dur']
        comp_length_temp = flux_cfg['dc_comp_pulse_length']

        excite_idx = expt_cfg['excite_idx']
        cal_idx = expt_cfg['cal_idx']

        for ii in range(4):

            target = 'flux_' + str(ii + 1)

            flux_a_read = expt_cfg['read_flux_a'][ii]

            # excite pulse now is the cal pulse
            ge_flux_a_excite = expt_cfg['excite_ge_flux_a'][cal_idx][ii]
            ge_excite_length = expt_cfg['excite_ge_length'][cal_idx]
            ef_flux_a_excite = expt_cfg['excite_ef_flux_a'][cal_idx][ii]
            ef_excite_length = expt_cfg['excite_ef_length'][cal_idx]

            excite_type = self.pulse_type
            ge_excite_dur = ap.get_pulse_span_length(pulse_cfg, excite_type, ge_excite_length)
            ef_excite_dur = ap.get_pulse_span_length(pulse_cfg, excite_type, ef_excite_length)

            read_dur = self.cfg['readout']['delay'] + self.cfg['readout']['width']

            # ramp up to ge_flux_a
            flux_start_time = 0
            self.psb.append(target, 'general', 'logistic_ramp', start_amp=0, stop_amp=ge_flux_a_excite, length=flux_ramp_dur)
            self.psb.append(target, 'general', 'linear_ramp', start_amp=ge_flux_a_excite, stop_amp=ge_flux_a_excite,
                            length=flux_settle_long_dur)
            # ge pi pulse
            ge_start_time = flux_start_time + flux_ramp_dur + flux_settle_long_dur - hw_delay
            ge_end_time = ge_start_time + ge_excite_dur
            self.psb.append(target, 'general', 'linear_ramp', start_amp=ge_flux_a_excite, stop_amp=ge_flux_a_excite,
                            length=ge_excite_dur)

            # ramp to ef_flux_a
            self.psb.append(target, 'general', 'logistic_ramp', start_amp=ge_flux_a_excite, stop_amp=ef_flux_a_excite,
                            length=flux_ramp_dur)
            self.psb.append(target, 'general', 'linear_ramp', start_amp=ef_flux_a_excite, stop_amp=ef_flux_a_excite,
                            length=flux_settle_dur)
            # excite pulse
            ef_start_time = ge_end_time + flux_ramp_dur + flux_settle_dur
            ef_end_time = ef_start_time + ef_excite_dur
            self.psb.append(target, 'general', 'linear_ramp', start_amp=ef_flux_a_excite, stop_amp=ef_flux_a_excite,
                            length=ef_excite_dur)

            # ramp to a_read
            self.psb.append(target, 'general', 'logistic_ramp', start_amp=ef_flux_a_excite, stop_amp=flux_a_read,
                            length=flux_ramp_dur)
            self.psb.append(target, 'general', 'linear_ramp', start_amp=flux_a_read, stop_amp=flux_a_read,
                            length=flux_settle_dur)
            # readout
            read_start_time = ef_end_time + flux_ramp_dur + flux_settle_dur
            read_end_time = read_start_time + read_dur
            self.psb.append(target, 'general', 'linear_ramp', start_amp=flux_a_read, stop_amp=flux_a_read, length=read_dur)

            # ramp down to 0
            self.psb.append(target, 'general', 'logistic_ramp', start_amp=flux_a_read, stop_amp=0, length=flux_ramp_dur)
            self.psb.append(target, 'general', 'linear_ramp', start_amp=0, stop_amp=0, length=flux_settle_dur)

            comp_start_time = read_end_time + flux_ramp_dur + flux_settle_dur

            # get area of flux pulses
            flux_area = self.psb.get_total_flux_pulse_area(target=target)
            a_comp_temp = - flux_area / (comp_length_temp + flux_ramp_dur)
            a_comp = np.sign(a_comp_temp) * min(0.8, abs(a_comp_temp))
            if a_comp == 0:
                comp_length = 0.0
            else:
                comp_length =  - flux_area/a_comp - flux_ramp_dur

            # compensation pulse
            self.psb.append(target, 'general', 'logistic_ramp', start_amp=0, stop_amp=a_comp, length=flux_ramp_dur)
            self.psb.append(target, 'general', 'linear_ramp', start_amp=a_comp, stop_amp=a_comp, length=comp_length)
            self.psb.append(target, 'general', 'logistic_ramp', start_amp=a_comp, stop_amp=0, length=flux_ramp_dur)

        # if not flux, clear psb and add drive pulses
        if not isFlux:

            # clear psb
            dummy = self.psb.get_pulse_sequence()
            dummy = self.psb.get_total_pulse_span_length()
            dummy = self.psb.get_total_flux_pulse_span_length()

            ge_excite_iq_freq = expt_cfg['excite_ge_freq'][cal_idx] - expt_cfg['drive_carrier_freq']
            ge_excite_a = expt_cfg['excite_ge_a'][cal_idx]

            ef_excite_iq_freq = expt_cfg['excite_ef_freq'][cal_idx] - expt_cfg['drive_carrier_freq']
            ef_excite_a = expt_cfg['excite_ef_a'][cal_idx]

            excite_target = 'q'

            if cal_pt == 0:
                self.psb.idle(ef_end_time)
            if cal_pt == 1:
                self.psb.idle(ge_start_time)
                self.psb.append(excite_target, 'general', self.pulse_type,
                                amp=ge_excite_a, length=ge_excite_length, freq=ge_excite_iq_freq)
                self.psb.idle(ef_end_time - ge_end_time)
            if cal_pt == 2:
                self.psb.idle(ge_start_time)
                self.psb.append(excite_target, 'general', self.pulse_type,
                                amp=ge_excite_a, length=ge_excite_length, freq=ge_excite_iq_freq)
                self.psb.idle(ef_start_time - ge_end_time)
                self.psb.append(excite_target, 'general', self.pulse_type,
                                amp=ef_excite_a, length=ef_excite_length, freq=ef_excite_iq_freq)