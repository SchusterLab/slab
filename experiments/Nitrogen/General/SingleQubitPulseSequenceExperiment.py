__author__ = 'Nelson'

from slab import *
from slab.instruments.Alazar import Alazar
from slab.experiments.ExpLib.QubitPulseSequenceExperiment import *
from slab.dsfit import *
from numpy import mean, arange,around


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
            xdata = expt_pts
            ydata = expt_avg_data
            FFT=scipy.fft(ydata)
            fft_freqs=scipy.fftpack.fftfreq(len(ydata),xdata[1]-xdata[0])
            max_ind=np.argmax(abs(FFT[2:len(ydata)/2]))+2
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
            print 'Rabi pi: %s ns' % (self.pi_length)
            print 'Rabi pi/2: %s ns' % (self.half_pi_length)
            print 'T1*: %s ns' % (fitdata[3])
            if (self.cfg['pulse_info']['save_to_file']):

                if (self.cfg['pulse_info']['calibrate_amp']):
                     temp =  around((self.cfg['pulse_info']['gauss']['a'])*(self.pi_length/(self.cfg['pulse_info']['gauss']['pi_length'])),decimals=4)
                     error = 0.005
                     if abs(temp-self.cfg['pulse_info']['gauss']['a'])< error:
                         print "Pulse well calibrated!"
                     if temp > 1 + error:
                        print "Increase pi length"
                        pass
                     else:
                        self.cfg['pulse_info']['gauss']['a'] =  temp
                        self.cfg['pulse_info']['gauss']['pi_a'] = temp
                        self.cfg['pulse_info']['gauss']['half_pi_a'] = temp
                else:
                    self.cfg['pulse_info']['gauss']['pi_length'] = self.pi_length
                    self.cfg['pulse_info']['gauss']['half_pi_length'] =  self.half_pi_length

class EFRabiExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='EF_Rabi', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=EFRabiSequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        self.drive.set_frequency(
            self.cfg['qubit']['frequency'] - self.cfg['pulse_info'][self.pulse_type]['iq_freq'])


    def post_run(self, expt_pts, expt_avg_data):
        print "Analyzing ef Rabi Data"
        self.pi_ef_length = self.cfg['pulse_info']['gauss']['pi_ef_length']
        self.half_pi_ef_length = self.cfg['pulse_info']['gauss']['half_pi_ef_length']

        xdata = expt_pts
        ydata = expt_avg_data
        FFT=scipy.fft(ydata)
        fft_freqs=scipy.fftpack.fftfreq(len(ydata),xdata[1]-xdata[0])
        max_ind=np.argmax(abs(FFT[2:len(ydata)/2]))+2
        fft_val=FFT[max_ind]

        fitparams=[0,0,0,0,0]
        fitparams[4]=np.mean(ydata)
        fitparams[0]=(max(ydata)-min(ydata))/2.#2*abs(fft_val)/len(fitdatay)
        fitparams[1]=fft_freqs[max_ind]
        fitparams[2]=-90.0
        fitparams[3]=(max(xdata)-min(xdata))
        fitdata=fitdecaysin(xdata[:],ydata[:],fitparams=fitparams,showfit=False)

        self.pi_ef_length = around(1/fitdata[1]/2,decimals=2)
        self.half_pi_ef_length =around(1/fitdata[1]/4,decimals=2)
        print 'ef Rabi pi: %s ns' % (self.pi_ef_length)
        print 'ef Rabi pi/2: %s ns' % (self.half_pi_ef_length)
        print 'ef T1*: %s ns' % (fitdata[3])
        if (self.cfg['pulse_info']['save_to_file']):

            if (self.cfg['pulse_info']['calibrate_ef_amp']):
                 temp =  around((self.cfg['pulse_info']['gauss']['pi_ef_a'])*(self.pi_ef_length/(self.cfg['pulse_info']['gauss']['pi_ef_length'])),decimals=4)
                 error = 0.005
                 if abs(temp-self.cfg['pulse_info']['gauss']['pi_ef_a'])< error:
                     print "Pulse well calibrated!"
                 if temp > 1 + error:
                    print "Increase ef pi length"
                    pass
                 else:
                    self.cfg['pulse_info']['gauss']['pi_ef_a'] =  temp
                    self.cfg['pulse_info']['gauss']['half_pi_ef_a'] = temp
            else:

                self.cfg['pulse_info']['gauss']['pi_ef_length'] = self.pi_ef_length
                self.cfg['pulse_info']['gauss']['half_pi_ef_length'] =  self.half_pi_ef_length

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



class T1rhoExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='T1rho', config_file='..\\config.json', **kwargs):
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=T1rhoSequence, pre_run=self.pre_run,
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
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value

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
        self.offset_freq =fitdata[1] * 1e9 - self.cfg['ef_ramsey']['ramsey_freq']
        # Sign for offset frequency changed on 4/28/2016
        self.suggested_anharm = self.cfg['qubit']['alpha'] -self.offset_freq
        print "Oscillation frequency: " + str(fitdata[1] * 1e3) + " MHz"
        print "T2*ef: " + str(fitdata[3]) + " ns"
        print "Suggested Anharmonicity: " + str(self.suggested_anharm)
        if (self.cfg['pulse_info']['save_to_file']):
                self.cfg['qubit']['alpha'] = self.suggested_anharm

        if self.data_file:
            slab_file = SlabFile(self.data_file)
            with slab_file as f:
                f.append_pt('suggested_anharm', self.suggested_anharm)
                f.close()

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
        xdata = expt_pts
        ydata = expt_avg_data
        fitparams = [(max(ydata)-min(ydata))/(2.0),1/360.0,90,mean(ydata)]
        fitdata=fitsin(xdata[:],ydata[:],fitparams=fitparams,showfit=False)
        self.cfg['pulse_info']['gauss']['offset_phase'] = around((-(fitdata[2]%180) + 90),2)
        print "Offset Phase = %s" %(self.cfg['pulse_info']['gauss']['offset_phase'])


class EFPulsePhaseOptimizationExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='EFPulsePhaseOptimization', config_file='..\\config.json', **kwargs):
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value


        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=EFPulsePhaseOptimizationSequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        pass


    def post_run(self, expt_pts, expt_avg_data):
        pass
        # if 'qubit_dc_offset' in self.extra_args:
        #     self.cfg['pulse_info']['fix_phase'] =  False
        # xdata = expt_pts
        # ydata = expt_avg_data
        # fitparams = [(max(ydata)-min(ydata))/(2.0),1/360.0,90,mean(ydata)]
        # fitdata=fitsin(xdata[:],ydata[:],fitparams=fitparams,showfit=False)
        # self.cfg['pulse_info']['gauss']['offset_phase'] = around((-(fitdata[2]%180) + 90),2)
        # print "Offset Phase = %s" %(self.cfg['pulse_info']['gauss']['offset_phase'])
        #



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
        pass


class PulseProbeIQExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='pulse_probe_iq', config_file='..\\config.json', **kwargs):
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
        # self.drive_freq = self.extra_args['drive_freq']
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=PulseProbeIQSequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)


    def pre_run(self):
        pass

    def post_run(self, expt_pts, expt_avg_data):
        pass


class EFRabiSweepExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='EF_Rabi_Sweep', config_file='..\\config.json', **kwargs):

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=EFRabiSweepSequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)


    def pre_run(self):
        pass


    def post_run(self, expt_pts, expt_avg_data):
        # print self.data_file
        pass


class RabiRamseyT1FluxSweepExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='rabi_ramsey_t1_flux_sweep', config_file='..\\config.json', **kwargs):
        self.extra_args = {}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
        self.drive_freq = self.extra_args['drive_freq']
        self.exp = self.extra_args['exp']
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

class SingleQubitRandomizedBenchmarkingPhaseOffsetExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='SingleQubit_RB_Phase_Offset', config_file='..\\config.json', **kwargs):

        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=RandomizedBenchmarkingPhaseOffsetSequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        pass

    def post_run(self, expt_pts, expt_avg_data):
        pass

class SingleQubitErrorAmplificationExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='SingleQubit_Error_Amplification', config_file='..\\config.json', **kwargs):
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)
        if 'c0' in self.extra_args:
            self.c0 = self.extra_args['c0']
            self.ci = self.extra_args['ci']
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=SingleQubitErrorAmplifcationSequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        pass

    def post_run(self, expt_pts, expt_avg_data):
        if 'c0' in self.extra_args:
            self.extra_args['c0'] = self.c0
            self.extra_args['ci'] = self.ci
        else:
            pass

class SingleQubitErrorAmplificationPhaseOffsetExperiment(QubitPulseSequenceExperiment):
    def __init__(self, path='', prefix='SingleQubit_Error_Amplification_Phase_Offset', config_file='..\\config.json', **kwargs):
        self.extra_args={}
        for key, value in kwargs.iteritems():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)
        if 'c0' in self.extra_args:
            self.c0 = self.extra_args['c0']
            self.ci = self.extra_args['ci']
        QubitPulseSequenceExperiment.__init__(self, path=path, prefix=prefix, config_file=config_file,
                                              PulseSequence=SingleQubitErrorAmplificationPhaseOffsetSequence, pre_run=self.pre_run,
                                              post_run=self.post_run, **kwargs)

    def pre_run(self):
        pass

    def post_run(self, expt_pts, expt_avg_data):
        if 'c0' in self.extra_args:
            self.extra_args['c0'] = self.c0
            self.extra_args['ci'] = self.ci
        else:
            pass