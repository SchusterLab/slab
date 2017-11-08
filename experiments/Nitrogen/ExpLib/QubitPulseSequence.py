__author__ = 'Nelson'

from slab.instruments.awg.PulseSequence import *
from numpy import arange, linspace,arctan
from slab.experiments.ExpLib.PulseSequenceBuilder import *
from slab import *
from slab.dsfit import*
from liveplot import LivePlotClient
from scipy.fftpack import *
from numpy import*

class QubitPulseSequence(PulseSequence):
    '''
    Parent class for all the single qubit pulse sequences.
    '''
    def __init__(self, name, cfg, expt_cfg, define_points, define_parameters, define_pulses, **kwargs):

        self.expt_cfg = expt_cfg
        define_points()
        define_parameters()
        sequence_length = len(self.expt_pts)

       # if "multimode" not in name.lower():
       #     cfg['awgs'][1]['upload'] = False
       # else:
       #     cfg['awgs'][1]['upload'] = True

        if (expt_cfg['use_pi_calibration']):
            sequence_length+=2

        PulseSequence.__init__(self, name, cfg['awgs'], sequence_length)

        self.psb = PulseSequenceBuilder(cfg)
        self.pulse_sequence_matrix = []
        total_pulse_span_length_list = []
        self.total_flux_pulse_span_length_list = []

        for ii, pt in enumerate(self.expt_pts):

            ## sideband cool
            #if 'sb_cool' in kwargs:
            #    if kwargs['sb_cool']:
            #        self.psb.append('q,mm'+str(8),'pi_ge')
            #self.psb.append('q,mm'+str(8),'pi_ge')

            # obtain pulse sequence for each experiment point
            define_pulses(pt)
            self.pulse_sequence_matrix.append(self.psb.get_pulse_sequence())
            total_pulse_span_length_list.append(self.psb.get_total_pulse_span_length())
            self.total_flux_pulse_span_length_list.append(self.psb.get_total_flux_pulse_span_length())

        if (expt_cfg['use_pi_calibration']):
            calibration_pts = [0,1]

            for jj, pt in enumerate(calibration_pts):
                if jj ==0:
                    #if 'sb_cool' in kwargs:
                    #    if kwargs['sb_cool']:
                    #        self.psb.append('q,mm'+str(8),'pi_ge')
                    #self.psb.append('q,mm'+str(8),'pi_ge')
                    self.psb.idle(10)
                if jj ==1:
                    #if 'sb_cool' in kwargs:
                    #    if kwargs['sb_cool']:
                    #        self.psb.append('q,mm'+str(8),'pi_ge')
                    #self.psb.append('q,mm'+str(8),'pi_ge')
                    # self.psb.append('q','cal_pi', self.pulse_type)
                    # print self.pulse_type
                    # use pi
                    #self.psb.append('q','cal_pi', self.pulse_type)

                    # use two half - pi
                    self.psb.append('q','half_pi', self.pulse_type)
                    self.psb.append('q','half_pi', self.pulse_type, phase = cfg['pulse_info'][self.pulse_type]['offset_phase'])

                self.pulse_sequence_matrix.append(self.psb.get_pulse_sequence())
                total_pulse_span_length_list.append(self.psb.get_total_pulse_span_length())
                self.total_flux_pulse_span_length_list.append(self.psb.get_total_flux_pulse_span_length())

        max_length = self.psb.get_max_length(total_pulse_span_length_list)
        max_flux_length = self.psb.get_max_flux_length(self.total_flux_pulse_span_length_list)
        self.set_all_lengths(max_length)
        self.set_waveform_length("qubit 1 flux", max_flux_length)

    def build_sequence(self):

        PulseSequence.build_sequence(self)
        wtpts = self.get_waveform_times('qubit drive I')
        mtpts = self.get_marker_times('qubit buffer')
        ftpts = self.get_waveform_times('qubit 1 flux')
        markers_readout = self.markers['readout pulse']
        markers_card = self.markers['card trigger']
        waveforms_qubit_I = self.waveforms['qubit drive I']
        waveforms_qubit_Q = self.waveforms['qubit drive Q']
        waveforms_qubit_flux = self.waveforms['qubit 1 flux']
        markers_qubit_buffer = self.markers['qubit buffer']
        markers_ch3m1 = self.markers['ch3m1']
        markers_ch4m1 = self.markers['ch4m1']
        self.psb.prepare_build(wtpts, mtpts, ftpts, markers_readout, markers_card, waveforms_qubit_I, waveforms_qubit_Q, waveforms_qubit_flux,
                              markers_qubit_buffer, markers_ch3m1, markers_ch4m1)
        generated_sequences = self.psb.build(self.pulse_sequence_matrix,self.total_flux_pulse_span_length_list)



        self.markers['readout pulse'], self.markers['card trigger'], self.waveforms['qubit drive I'], self.waveforms[
            'qubit drive Q'], self.waveforms['qubit 1 flux'], self.markers['qubit buffer'], self.markers['ch3m1'], self.markers['ch4m1'] = generated_sequences

        # # Testing inversion of line profile
        #
        # fitdata = array([  1.95119318e-01,   1.48711751e-01,   1.53483539e+00,
        # 1.52158201e-04,   1.19684011e-01,   7.25849447e-02,
        # 1.62684835e+00,   3.41254602e-03,   3.58586680e-02,
        # 1.21194486e-01,   1.71463776e+00,   3.14296448e-03,
        # -1.97552714e-03,   8.68741763e-02,   1.80050879e+00,
        # 3.57667674e-03,  -1.14605700e-01,   2.59936204e-02,
        # 1.88897670e+00,   3.43332822e-03,   9.77503349e-02,
        # 1.31902880e-01,   1.97485817e+00,   3.47424343e-03,
        # 5.16104275e-02,   1.15657190e-01,   2.06144752e+00,
        # 4.17123577e-03,   2.71910473e-02,   1.05153977e-01,
        # 2.14098839e+00,   4.51683433e-03,  -7.09183086e-03,
        # 1.72537625e-01,   2.23595112e+00,   4.54133763e-03,
        # -1.11403387e-01,   1.61313254e-01,   2.32372700e+00,
        # 5.04861443e-03,  -2.22026862e-01,   5.77600276e-02,
        # 2.40967747e+00,   3.39506542e-03,  -1.52292961e-01,
        # 4.03222692e-02,   2.49721818e+00,   4.51533849e-03,
        # -1.67667628e-01,   1.86954073e-02,   2.58312924e+00,
        # 4.10638807e-03,  -1.20331256e-01,  -7.97484012e-02,
        # 2.67375167e+00,   5.09114072e-03,  -4.48743454e-01,
        # -2.11385267e-01,   2.75991459e+00,   4.65346217e-03,
        # -2.76690604e-01,  -1.38046523e-01])
        # #
        #
        #
        # padding = 500.0
        #
        # ftime = concatenate((arange(-padding,0,0.02),ftpts),axis=0)
        # ftime = concatenate((ftime,arange(ftime[-1],ftime[-1]+padding,0.02)),axis=0)
        # T = max(ftime) - min(ftime)
        # freq1 = linspace(1/(T),len(ftime)/(2*T),len(ftime)/2)
        # freq2 = linspace(-len(ftime)/(2*T),-1/(T),len(ftime)/2)
        # freq = concatenate((freq1,freq2),axis=0)
        # transform = []
        # for i in range(len(self.waveforms['qubit 1 flux'])):
        #     pulse = concatenate((zeros(int(padding/0.02)),self.waveforms['qubit 1 flux'][i]),axis=0)
        #     pulse = concatenate((pulse,zeros(int(padding/0.02))),axis=0)
        #     yFFT = scipy.fftpack.fft(pulse)
        #     yFFTc = concatenate((zeros(len(yFFT)/2),2*yFFT[len(yFFT)/2:len(yFFT)]),axis=0)
        #     transform.append(real(scipy.fftpack.ifft(yFFTc/harmfunccomplexsum2(fitdata, freq))))
        #
        # self.waveforms['qubit 1 flux'] = array(transform)
        # ftpts = ftime



        # np.save('Rabi_R',self.markers['readout pulse'])
        # np.save('Rabi_Rtime',mtpts)
        # np.save('Rabi_I',self.waveforms['qubit drive I'])
        # np.save('Rabi_Q',self.waveforms['qubit drive I'])
        # np.save('Rabi_Ptime',wtpts)


        # with SlabFile('sequences\pulse_sequence.h5') as f:
        #     f.add('readout pulse', self.markers['readout pulse'])
        #     f.add('card trigger', self.markers['card trigger'])
        #     f.add('qubit drive I', self.waveforms['qubit drive I'])
        #     f.add('qubit drive Q', self.waveforms['qubit drive Q'])
        #     f.add('qubit 1 flux', self.waveforms['qubit 1 flux'])
        #     # f.add('f pts', ftpts)
        #     f.add('f time', ftpts)
        #     f.add('q time', wtpts)
        #     f.add('m time', mtpts)
        #     f.add('qubit buffer', self.markers['qubit buffer'])
        #     f.add('ch3m1', self.markers['ch3m1'])
        #     f.close()

    def reshape_data(self, data):
        return np.reshape(data, (self.sequence_length, self.waveform_length))