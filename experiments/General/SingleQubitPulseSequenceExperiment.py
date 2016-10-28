__author__ = 'Nelson'

from slab import *
from slab.instruments.Alazar import Alazar
from slab.experiments.ExpLib.QubitPulseSequenceExperiment import *
from numpy import mean, arange


class RabiExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Rabi', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=RabiSequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        self.drive.set_frequency(self.cfg['qubit']['frequency'] - self.cfg[self.expt_cfg_name]['iq_freq'])

    def post_run(self, expt_pts, expt_avg_data):


        if self.cfg[self.expt_cfg_name]['calibrate_pulse']:
            print "Analyzing Rabi Data"
            fitdata = fitdecaysin(expt_pts, expt_avg_data)
            pulse_type = self.cfg[self.expt_cfg_name]['pulse_type']
            # determine the start location of the rabi oscillation. +1 if it starts top; -1 if it starts bottom.
            start_signal = np.sign(180 - fitdata[2] % 360)

            self.excited_signal = start_signal
            # time takes to have the rabi oscillation phase to +/- pi/2
            pi_length = (self.excited_signal * np.sign(fitdata[0]) * 0.5 * np.pi - fitdata[
                2] * np.pi / 180.) / (2 * np.pi * fitdata[1]) % (1 / fitdata[1])
            # time takes to have the rabi oscillation phase to 0/ pi
            half_pi_length = (self.excited_signal * np.sign(fitdata[0]) * np.pi - fitdata[
                2] * np.pi / 180.) / (2 * np.pi * fitdata[1]) % (0.5 / fitdata[1])

            self.pi_length = pi_length
            self.half_pi_length = half_pi_length
            self.a = self.cfg[self.expt_cfg_name]['a']
            self.iq_freq = self.cfg[self.expt_cfg_name]['iq_freq']

        else:
            print "Analyzing Rabi Data"
            self.pi_length = self.cfg['pulse_info']['gauss']['pi_length']
            self.half_pi_length = self.cfg['pulse_info']['gauss']['half_pi_length']
            fitdata = fitdecaysin(expt_pts[10:], expt_avg_data[10:])

            # if (-fitdata[2]%180 - 90)/(360*fitdata[1]) < 0:
            #     print "Rabi pi/2 length =" + str((-fitdata[2]%180 + 180)/(360*fitdata[1]))
            #     print "Rabi pi length =" + str((-fitdata[2]%180 + 270)/(360*fitdata[1]))
            #     self.half_pi_length = ((-fitdata[2]%180 + 180)/(360*fitdata[1]))
            #     self.pi_length = ((-fitdata[2]%180 + 270)/(360*fitdata[1]))
            #
            #
            # else:
            #
            #     print "Rabi pi/2 length =" + str((-fitdata[2]%180 )/(360*fitdata[1]))
            #     print "Rabi pi length =" + str((-fitdata[2]%180 + 90)/(360*fitdata[1]))
            #     self.half_pi_length = ((-fitdata[2]%180)/(360*fitdata[1]))
            #     self.pi_length =((-fitdata[2]%180 + 90)/(360*fitdata[1]))

            self.pi_length = 0.5/fitdata[1]
            self.half_pi_length = 0.25/(fitdata[1])
            print 'Rabi pi: %s ns' % (0.5 / fitdata[1])
            print 'Rabi pi/2: %s ns' % (0.25 / fitdata[1])
            print 'T1*: %s ns' % (fitdata[3])
            if (self.cfg['pulse_info']['save_to_file']):

                if (self.cfg['pulse_info']['calibrate_amp']):
                     if(self.pi_length < self.cfg['pulse_info']['gauss']['pi_length']):
                        self.cfg['pulse_info']['gauss']['a'] =  (self.cfg['pulse_info']['gauss']['a'])*(self.pi_length/(self.cfg['pulse_info']['gauss']['pi_length']))
                     else:
                         temp = (self.cfg['pulse_info']['gauss']['a'])*((self.cfg['pulse_info']['gauss']['pi_length'])/self.pi_length)
                         if(temp > 1):
                             print "Increase pi length"
                             pass
                         else:
                            self.cfg['pulse_info']['gauss']['a'] =  temp
                else:
                    self.cfg['pulse_info']['gauss']['pi_length'] = self.pi_length
                    self.cfg['pulse_info']['gauss']['half_pi_length'] =  self.half_pi_length

class T1Experiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='T1', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=T1Sequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        # self.drive.set_frequency(self.cfg['qubit']['frequency'] - self.cfg['pulse_info'][self.pulse_type]['iq_freq'])

        pass

    def post_run(self, expt_pts, expt_avg_data):
        print "Analyzing T1 Data"
        fitdata = fitexp(expt_pts, expt_avg_data)
        print "T1: " + str(fitdata[3]) + " ns"


class RamseyExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Ramsey', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=RamseySequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        self.drive.set_frequency(
            self.cfg['qubit']['frequency'] - self.cfg['pulse_info'][self.pulse_type]['iq_freq'])
        print self.cfg['pulse_info'][self.pulse_type]['iq_freq']

    def post_run(self, expt_pts, expt_avg_data):
        print "Analyzing Ramsey Data"
        fitdata = fitdecaysin(expt_pts, expt_avg_data)

        self.offset_freq = self.cfg['ramsey']['ramsey_freq'] - fitdata[1] * 1e9

        self.flux = self.cfg['freq_flux']['flux']
        self.freq_flux_slope = self.cfg['freq_flux']['freq_flux_slope']

        self.suggested_qubit_freq = self.cfg['qubit']['frequency'] - (fitdata[1] * 1e9 - self.cfg['ramsey']['ramsey_freq'])
        # This equation had the wrong sign..?
        print "Oscillation frequency: " + str(fitdata[1] * 1e3) + " MHz"
        print "T2*: " + str(fitdata[3]) + " ns"

        print "Suggested Qubit Frequency: " + str(self.suggested_qubit_freq)
        print "Or Suggested Flux: " + str(self.flux - self.offset_freq / self.freq_flux_slope)
        self.cfg['freq_flux']['flux'] = self.flux - self.offset_freq / self.freq_flux_slope

class SpinEchoExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Spin_Echo', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=SpinEchoSequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        self.drive.set_frequency(
            self.cfg['qubit']['frequency'] - self.cfg['pulse_info'][self.pulse_type]['iq_freq'])

    def post_run(self, expt_pts, expt_avg_data):
        print "Analyzing Spin Echo Data"
        fitdata = fitdecaysin(expt_pts, expt_avg_data)
        suggested_qubit_freq = self.cfg['qubit']['frequency'] + (fitdata[1] * 1e9 - self.cfg['ramsey']['ramsey_freq'])
        print "Oscillation frequency: " + str(fitdata[1] * 1e3) + " MHz"
        print "T2*: " + str(fitdata[3]) + " ns"
        print "Suggested Qubit Frequency: " + str(suggested_qubit_freq)


class EFRabiExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='EF_Rabi', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file, liveplot_enabled=False,
                                              PulseSequence=EFRabiSequence, pre_run=self.pre_run,
                                              post_run=self.post_run)

    def pre_run(self):
        self.drive.set_frequency(
            self.cfg['qubit']['frequency'] - self.cfg['pulse_info'][self.pulse_type]['iq_freq'])


    def post_run(self, expt_pts, expt_avg_data):
        print "Analyzing ef Rabi Data"
        self.pi_ef_length = self.cfg['pulse_info']['gauss']['pi_ef_length']
        self.half_pi_ef_length = self.cfg['pulse_info']['gauss']['half_pi_ef_length']
        fitdata = fitdecaysin(expt_pts[3:], expt_avg_data[3:])
        self.pi_ef_length = 0.5 / fitdata[1]
        self.half_pi_ef_length = 0.25 / (fitdata[1])
        print 'ef Rabi pi: %s ns' % (0.5 / fitdata[1])
        print 'ef Rabi pi/2: %s ns' % (0.25 / fitdata[1])
        print 'ef T1*: %s ns' % (fitdata[3])
        if (self.cfg['pulse_info']['save_to_file']):
                self.cfg['pulse_info']['gauss']['pi_ef_length'] = self.pi_ef_length
                self.cfg['pulse_info']['gauss']['half_pi_ef_length'] =  self.half_pi_ef_length

class EFRamseyExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='EF_Ramsey', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=EFRamseySequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        self.drive.set_frequency(
            self.cfg['qubit']['frequency'] - self.cfg['pulse_info'][self.pulse_type]['iq_freq'])

    def post_run(self, expt_pts, expt_avg_data):
        pass
        print "Analyzing EF Ramsey Data"
        fitdata = fitdecaysin(expt_pts, expt_avg_data)

        # self.offset_freq =self.cfg['ramsey']['ramsey_freq'] - fitdata[1] * 1e9
        #self.flux_volt = self.cfg['freq_flux']['flux_volt']
        #self.freq_flux_slope = self.cfg['freq_flux']['slope']

        suggested_anharm = self.cfg['qubit']['alpha'] + (+fitdata[1] * 1e9 - self.cfg['ef_ramsey']['ramsey_freq'])

        print "Oscillation frequency: " + str(fitdata[1] * 1e3) + " MHz"
        print "T2*ef: " + str(fitdata[3]) + " ns"
        #if round(self.offset_freq/ self.freq_flux_slope,4)==0.0000:
        #   print "Qubit frequency is well calibrated."
        #else:
        print "Suggested Anharmonicity: " + str(suggested_anharm)
        #  print "Or Suggested Flux Voltage: " +str(round(self.flux_volt -self.offset_freq/ self.freq_flux_slope,4))
        if (self.cfg['pulse_info']['save_to_file']):
                self.cfg['qubit']['alpha'] = suggested_anharm

class EFT1Experiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='EF_T1', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=EFT1Sequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        self.drive.set_frequency(self.cfg['qubit']['frequency'] - self.cfg['pulse_info'][self.pulse_type]['iq_freq'])

    def post_run(self, expt_pts, expt_avg_data):
        print "Analyzing EF T1 Data"
        fitdata = fitexp(expt_pts, expt_avg_data)
        print "EF T1: " + str(fitdata[3]) + " ns"


class EFT1Experiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='EF_T1', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=EFT1Sequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        self.drive.set_frequency(self.cfg['qubit']['frequency'] - self.cfg['pulse_info'][self.pulse_type]['iq_freq'])

    def post_run(self, expt_pts, expt_avg_data):
        print "Analyzing EF T1 Data"
        fitdata = fitexp(expt_pts, expt_avg_data)
        print "EF T1: " + str(fitdata[3]) + " ns"


class EFT1Experiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='EF_T1', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=EFT1Sequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        self.drive.set_frequency(self.cfg['qubit']['frequency'] - self.cfg['pulse_info'][self.pulse_type]['iq_freq'])

    def post_run(self, expt_pts, expt_avg_data):
        print "Analyzing EF T1 Data"
        fitdata = fitexp(expt_pts, expt_avg_data)
        print "EF T1: " + str(fitdata[3]) + " ns"


class HalfPiXOptimizationExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='HalfPiXOptimization', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=HalfPiXOptimizationSequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        pass

    def post_run(self, expt_pts, expt_avg_data):
        pass


class PiXOptimizationExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='PiXOptimization', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=PiXOptimizationSequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        pass

    def post_run(self, expt_pts, expt_avg_data):
        pass

class PiXoptimizationChangeAmpExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='PiXOptimization_change_amp', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=PiXoptimizationChangeAmpSequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        pass

    def post_run(self, expt_pts, expt_avg_data):
        pass


class HalfPiYPhaseOptimizationExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='HalfPiYPhaseOptimization', config_file='..\\config.json', **kwargs):
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
        if 'qubit_dc_offset' in self.extra_args:
            self.qubit_dc_offset = self.extra_args['qubit_dc_offset']

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=HalfPiYPhaseOptimizationSequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        pass


    def post_run(self, expt_pts, expt_avg_data):
        if 'qubit_dc_offset' in self.extra_args:
            self.cfg['pulse_info']['fix_phase'] =  False
        else:
            pass


class TomographyExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Tomography', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=TomographySequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        pass

    def post_run(self, expt_pts, expt_avg_data):
        pass


class HalfPiXOptimizationSweepExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='HalfPiXOptimization_Sweep', config_file='..\\config.json', **kwargs):
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
        self.pulse_length = self.extra_args['pulse_length']
        self.pulse_amp = self.extra_args['pulse_amp']

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=HalfPiXOptimizationSweepSequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        pass

    def post_run(self, expt_pts, expt_avg_data):
        slab_file = SlabFile(self.data_file)
        with slab_file as f:
            f.append_pt('pulse_length', self.pulse_length)
            f.append_pt('pulse_amp', self.pulse_amp)
            f.append_line('sweep_expt_avg_data', expt_avg_data)
            f.append_line('sweep_expt_pts', expt_pts)
            f.close()


class PiXOptimizationSweepExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='PiXOptimization_Sweep', config_file='..\\config.json', **kwargs):
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
        self.pulse_length = self.extra_args['pulse_length']
        self.pulse_amp = self.extra_args['pulse_amp']

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=PiXOptimizationSweepSequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        pass

    def post_run(self, expt_pts, expt_avg_data):
        slab_file = SlabFile(self.data_file)
        with slab_file as f:
            f.append_pt('pulse_length', self.pulse_length)
            f.append_pt('pulse_amp', self.pulse_amp)
            f.append_line('sweep_expt_avg_data', expt_avg_data)
            f.append_line('sweep_expt_pts', expt_pts)
            f.close()


class RabiSweepExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='Rabi_Sweep', config_file='..\\config.json', **kwargs):
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
        self.drive_freq = self.extra_args['drive_freq']
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=RabiSweepSequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)


    def pre_run(self):
        self.drive.set_frequency(self.drive_freq-self.cfg['rabi_sweep']['iq_freq'])


    def post_run(self, expt_pts, expt_avg_data):
        # print self.data_file
        slab_file = SlabFile(self.data_file)
        with slab_file as f:
            f.append_pt('drive_freq', self.drive_freq)
            f.append_line('sweep_expt_avg_data', expt_avg_data)
            f.append_line('sweep_expt_pts', expt_pts)

            f.close()


class RabiRamseyT1FluxSweepExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='rabi_ramsey_t1_flux_sweep', config_file='..\\config.json', **kwargs):
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value


        self.drive_freq = self.extra_args['drive_freq']
        if 'exp' in self.extra_args:
            self.exp = self.extra_args['exp']
        else:
            self.exp = 'rabi'
        self.flux = self.extra_args['flux']

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=RabiRamseyT1FluxSweepSequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)


    def pre_run(self):
        self.drive.set_frequency(self.drive_freq - self.cfg[self.expt_cfg_name]['iq_freq'])


    def post_run(self, expt_pts, expt_avg_data):
        # print self.data_file

        if self.exp == 'rabi' or self.exp == 'ef_rabi':
            print "Analyzing Rabi Data"
            fitdata = fitdecaysin(expt_pts, expt_avg_data)

            # determine the start location of the rabi oscillation. +1 if it starts top; -1 if it starts bottom.

            ### get's excited signal by looking at 1st signal
            start_signal = np.sign(mean(expt_avg_data) - expt_avg_data[0])
            self.excited_signal = start_signal

            # ### get's excited signal by looking at phase
            # start_signal = np.sign(180 - fitdata[2] % 360)
            # self.excited_signal = start_signal

            # time takes to have the rabi oscillation phase to +/- pi/2
            # pi_length = (self.excited_signal * np.sign(fitdata[0]) * 0.5 * np.pi - fitdata[
            #     2] * np.pi / 180.) / (2 * np.pi * fitdata[1]) % (1 / fitdata[1])
            # time takes to have the rabi oscillation phase to 0/ pi
            # half_pi_length = (self.excited_signal * np.sign(fitdata[0]) * np.pi - fitdata[
            #     2] * np.pi / 180.) / (2 * np.pi * fitdata[1]) % (0.5 / fitdata[1])
            pi_length = 0.5 / fitdata[1]
            half_pi_length = pi_length / 2

            self.pi_length = pi_length
            self.half_pi_length = half_pi_length

            print "Pi length: " + str(self.pi_length)
            print "Half pi: " + str(self.half_pi_length)

        if self.exp == 'ramsey' or self.exp == 'ramsey_long':
            print "Analyzing Ramsey Data"
            fitdata = fitdecaysin(expt_pts, expt_avg_data)

            self.offset_freq = self.cfg['ramsey']['ramsey_freq'] - fitdata[1] * 1e9

            self.flux = self.cfg['freq_flux']['flux']
            self.freq_flux_slope = self.cfg['freq_flux']['freq_flux_slope']

            self.suggested_qubit_freq = self.drive_freq - (
            fitdata[1] * 1e9 - self.cfg[self.expt_cfg_name]['ramsey']['ramsey_freq'])
            print "Oscillation frequency: " + str(fitdata[1] * 1e3) + " MHz"
            print "T2*: " + str(fitdata[3]) + " ns"
            self.t2 = fitdata[3]

            print "Suggested Qubit Frequency: " + str(self.suggested_qubit_freq)
            print "Or Suggested Flux: " + str(self.flux - self.offset_freq / self.freq_flux_slope)

        if self.exp == 't1' or self.exp == 'ef_t1':
            print "Analyzing T1 Data"
            fitdata = fitexp(expt_pts, expt_avg_data)
            print "T1: " + str(fitdata[3]) + " ns"
            self.t1 = fitdata[3]

        if self.exp == 'ef_ramsey' or self.exp == 'ef_ramsey_long':
            self.alpha = self.extra_args['alpha']
            print "Analyzing EF Ramsey Data"
            fitdata = fitdecaysin(expt_pts, expt_avg_data)

            self.suggested_qubit_alpha = self.alpha - (
            fitdata[1] * 1e9 - self.cfg[self.expt_cfg_name]['ef_ramsey']['ramsey_freq'])
            print "Oscillation frequency: " + str(fitdata[1] * 1e3) + " MHz"
            print "T2*: " + str(fitdata[3]) + " ns"
            self.t2 = fitdata[3]

        if self.exp == 'half_pi_phase_sweep':
            print "Analyzing Half Pi Phase Sweep Data"
            fitdata = fitdecaysin(expt_pts, expt_avg_data)

            self.half_pi_phase = fitdata[2] % 360
            print "Oscillation phase: " + str(self.half_pi_phase)

        slab_file = SlabFile(self.data_file)
        with slab_file as f:
            f.append_pt('flux', self.flux)
            f.append_pt('drive_freq', self.drive_freq)
            f.append_line(self.exp + '_expt_avg_data', expt_avg_data)
            f.append_line(self.exp + '_expt_pts', expt_pts)
            if self.exp == 'ramsey_long':
                f.append_pt("t2", self.t2)
            if self.exp == 't1':
                f.append_pt("t1", self.t1)
            if self.exp == 'ef_ramsey_long':
                f.append_pt("ef_t2", self.t2)
                f.append_pt("alpha", self.suggested_qubit_alpha)
            # if self.exp == 'ef_t1':
            #     f.append_pt("ef_t1",self.t1)
            if self.exp == 'half_pi_phase_sweep':
                f.append_pt("half_pi_phase", self.half_pi_phase)

            f.close()

class SingleQubitRandomizedBenchmarkingExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='SingleQubit_RB', config_file='..\\config.json', **kwargs):

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=RandomizedBenchmarkingSequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        pass

    def post_run(self, expt_pts, expt_avg_data):
        pass
        # slab_file = SlabFile(self.data_file)
        # with slab_file as f:
        #     f.append_pt('pulse_length', self.pulse_length)
        #     f.append_pt('pulse_amp', self.pulse_amp)
        #     f.append_line('sweep_expt_avg_data', expt_avg_data)
        #     f.append_line('sweep_expt_pts', expt_pts)
        #     f.close()