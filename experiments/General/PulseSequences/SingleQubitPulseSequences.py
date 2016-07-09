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
            self.psb.append('q','general', self.pulse_type, amp=pt, length=self.expt_cfg['length'],freq=self.expt_cfg['iq_freq'])
        else:
            self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=pt,freq=self.expt_cfg['iq_freq'])




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
                self.psb.append('q','pi', self.pulse_type)
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
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq)

    def define_pulses(self,pt):
        if self.expt_cfg['ge_pi']:
           self.psb.append('q','pi', self.pulse_type)
        self.psb.append('q','general', self.ef_pulse_type, amp=self.pulse_cfg[self.expt_cfg['ef_pulse_type']]['pi_ef_a'], length=pt,freq=self.ef_sideband_freq)
        #self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=pt,freq=self.ef_sideband_freq)
        #self.psb.idle(4*pt)
        self.psb.append('q','pi', self.pulse_type)


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
        self.ef_sideband_freq = self.pulse_cfg[self.pulse_type]['iq_freq']-(self.qubit_cfg['frequency']-ef_freq + self.expt_cfg['ramsey_freq'])

    def define_pulses(self,pt):
        if self.expt_cfg['ge_pi']:
            self.psb.append('q','pi', self.pulse_type)
        #self.psb.append('q','general', self.pulse_type, amp=self.expt_cfg['a'], length=pt,freq=self.ef_sideband_freq)
        self.psb.append('q','general', self.ef_pulse_type,amp=self.expt_cfg['a'],length = self.expt_cfg['half_pi_ef'], freq=self.ef_sideband_freq )
        self.psb.idle(pt)
        self.psb.append('q','general', self.ef_pulse_type,amp=self.expt_cfg['a'],length = self.expt_cfg['half_pi_ef'],freq=self.ef_sideband_freq )
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

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']

    def define_pulses(self,pt):
        n = 2*pt+1
        i = 0
        while i< n:
            #self.psb.append('q','half_pi', self.pulse_type)
            self.psb.append('q','general', self.pulse_type, amp=self.pulse_cfg[self.pulse_type]['half_pi_a'], length=self.pulse_cfg[self.pulse_type]['half_pi_length'],freq=self.pulse_cfg[self.pulse_type]['iq_freq'], phase= i*self.pulse_cfg[self.pulse_type]['phase'])
            i += 1


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
        QubitPulseSequence.__init__(self,name, cfg, expt_cfg,self.define_points, self.define_parameters, self.define_pulses)

    def define_points(self):
        self.expt_pts = arange(self.expt_cfg['start'], self.expt_cfg['stop'], self.expt_cfg['step'])

    def define_parameters(self):
        self.pulse_type =  self.expt_cfg['pulse_type']

    def define_pulses(self,pt):
        n = pt
        i = 0
        self.psb.append('q','half_pi', self.pulse_type)
        while i< n:
            self.psb.append('q','pi', self.pulse_type)
            i += 1

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
        self.psb.append('q','pi', self.pulse_type,phase=pt)
        self.psb.append('q','half_pi', self.pulse_type,phase=0)

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
        n = pt
        i = 0
        self.psb.append('q','half_pi', self.pulse_type)
        while i< n:
            self.psb.append('q','general', self.pulse_type, amp=self.pulse_amp, length=self.pulse_length,freq=self.pulse_cfg[self.pulse_type]['iq_freq'])
            i += 1


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