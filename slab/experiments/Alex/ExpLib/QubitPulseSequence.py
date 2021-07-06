__author__ = 'Nelson'

from slab.instruments.awg.PulseSequence import *
from numpy import arange, linspace, array
from slab.experiments.Alex.ExpLib.PulseSequenceBuilder import *
import time
import visdom
from liveplot import LivePlotClient

class QubitPulseSequence(PulseSequence):
    '''
    Parent class for all the single qubit pulse sequences.
    '''
    def __init__(self, name, cfg, expt_cfg, define_points, define_parameters, define_pulses, **kwargs):

        self.expt_cfg = expt_cfg
        self.cfg = cfg
        define_points()
        define_parameters()
        sequence_length = len(self.expt_pts)

        # if "multimode" not in name.lower():
        #     cfg['awgs'][1]['upload'] = False
        # else:
        #     cfg['awgs'][1]['upload'] = True
        calibration_pts = []

        if (expt_cfg['use_g-e-f_calibration']):
            calibration_pts = list(range(3))
            sequence_length+=3
        elif (expt_cfg['use_pi_calibration']):
            calibration_pts = list(range(2))
            sequence_length+=2
        print('calibration_pts =', calibration_pts)

        PulseSequence.__init__(self, name, cfg['awgs'], sequence_length)

        self.psb = PulseSequenceBuilder(cfg)
        self.pulse_sequence_matrix = []
        self.total_pulse_span_length_list = []
        self.total_flux_pulse_span_length_list = []
        self.flux_pulse_span_list = []

        for ii, pt in enumerate(self.expt_pts):
            # obtain pulse sequence for each experiment point
            define_pulses(pt)

            self.pulse_sequence_matrix.append(self.psb.get_pulse_sequence())
            self.total_pulse_span_length_list.append(self.psb.get_total_pulse_span_length())
            self.total_flux_pulse_span_length_list.append(self.psb.get_total_flux_pulse_span_length())

        if len(calibration_pts) > 0:

            for jj, pt in enumerate(calibration_pts):

                if self.name == 'rabi_thermalizer':
                    define_pulses(jj+len(self.expt_pts))

                else:
                    if jj ==0:
                        self.psb.idle(10)
                    if jj ==1:
                        self.psb.append('q','cal_pi', self.pulse_type)
                    if jj ==2:
                        self.psb.append('q','cal_pi', self.pulse_type)
                        self.psb.append('q', 'pi_q_ef', self.pulse_type)

                self.pulse_sequence_matrix.append(self.psb.get_pulse_sequence())
                self.total_pulse_span_length_list.append(self.psb.get_total_pulse_span_length())
                self.total_flux_pulse_span_length_list.append(self.psb.get_total_flux_pulse_span_length())

        print('total_pulse_span_length_list', self.total_pulse_span_length_list)

        max_length = self.psb.get_max_length(self.total_pulse_span_length_list)
        max_flux_length = self.psb.get_max_flux_length(self.total_flux_pulse_span_length_list)
        self.set_all_lengths(max_length)

        ###

        # flux pulse
        temp_seq_matrix = self.pulse_sequence_matrix[:]
        self.pulse_sequence_matrix = []
        # clears
        dummy = self.psb.get_total_pulse_span_length()
        dummy = self.psb.get_total_flux_pulse_span_length()

        flux_max_area_list = np.zeros(8)
        flux_max_power_list = np.zeros(8)
        for ii in range(len(self.expt_pts) + len(calibration_pts)):

            if self.name == 'rabi_thermalizer' or self.name == 'histogram_rabi_thermalizer':

               # print 'add rabi_thermalizer flux pulse'
                flux_total_span, flux_area_list, flux_power_list = define_pulses(ii, isFlux = True)

                flux_max_area_list = np.where(np.abs(flux_max_area_list) >= np.abs(flux_area_list), flux_max_area_list, flux_area_list)
                flux_max_power_list = np.where(flux_max_power_list >= flux_power_list, flux_max_power_list, flux_power_list)
                # flux_total_span = self.psb.get_total_pulse_span_length() # also clears

            # 0426 hack for ramsey fast flux scope, ignore cal pts
            # elif self.name == 'ramsey':
            #     flux_total_span = self.add_flux_pulses_hack(idx=ii, pulse_span_length = self.total_pulse_span_length_list[ii])

            else:
                flux_total_span = self.add_flux_pulses(pulse_span_length = self.total_pulse_span_length_list[ii])
            temp_seqs = self.psb.get_pulse_sequence() # also clears seq

            self.flux_pulse_span_list.append(flux_total_span)
            dummy = self.psb.get_total_flux_pulse_span_length()

            #print 'flux_pulse_span_list', self.flux_pulse_span_list

            self.pulse_sequence_matrix.append(temp_seq_matrix[ii] + temp_seqs)  # concatenate hetero/flux pulse

        max_flux_length = round_samples( max(self.flux_pulse_span_list)+ 2 * self.psb.start_end_buffer)
        self.set_all_lengths(max(max_flux_length,max_length))

        print('max length =', max_length, 'ns')
        print('max flux length =', max_flux_length, 'ns')
        if max(max_flux_length,max_length) >= self.cfg["expt_trigger"]["period_ns"]:
            print('Error!! Max sequence length larger than Exp period! ')
        print('flux_max_area_list = [', ', '.join(map(str, flux_max_area_list)), ']')
        print('flux_max_power_list = [', ', '.join(map(str, flux_max_power_list)), ']')

        # import csv
        # with open(r'C:\slab_data_temp\fast_flux_kernels\flux_max_area.csv', 'a') as csvfile:
        #     ww = csv.writer(csvfile, delimiter=',')
        #     ww.writerow(map(str, flux_max_area_list))
        #     csvfile.close()
        #
        # with open(r'C:\slab_data_temp\fast_flux_kernels\flux_max_power.csv', 'a') as csvfile:
        #     ww = csv.writer(csvfile, delimiter=',')
        #     ww.writerow(map(str, flux_max_power_list))
        #     csvfile.close()

        ###

        ###
        # heterodyne pulse - hack: max_length = 0
        if (self.name == 'vacuum_rabi') or (self.name[0:9] == 'histogram') :
            # vacuum_rabi : heterodyne pulses in SingleQubitPulseSeq
            #print 'skip adding heterodyne pulse in QubitPulseSeq'
            pass

        else:
            temp_seq_matrix = self.pulse_sequence_matrix[:]
            self.pulse_sequence_matrix = []
            self.add_heterodyne_pulses()
            #self.add_flux_pulses(length=500)
            temp_seqs = self.psb.get_pulse_sequence()

            # clears
            dummy = self.psb.get_total_pulse_span_length()
            dummy = self.psb.get_total_flux_pulse_span_length()

            for ii in range(len(self.expt_pts)+len(calibration_pts)):
                self.pulse_sequence_matrix.append(temp_seq_matrix[ii] + temp_seqs) # concatenate hetero/flux pulse
        ###


    def add_flux_pulses(self, pulse_span_length):

        #print pulse_span_length

        # this is to align flux pulse to readout? (diff in 2 pxdac cards)
        hw_delay = self.cfg['flux_pulse_info']['pxdac_hw_delay']

        if self.cfg['flux_pulse_info']['on_during_drive'] and self.cfg['flux_pulse_info']['on_during_readout']:

            flux_width = max(pulse_span_length + self.cfg['readout']['delay'] + self.cfg['readout']['width'] \
                             - self.cfg['flux_pulse_info']['flux_drive_delay'], 0)
            flux_comp_width = flux_width  # self.cfg['flux_pulse_info']['dc_comp_pulse_length']
            flux_delay = self.cfg['flux_pulse_info']['flux_drive_delay'] + hw_delay
            flux_idle = 100.0

        elif (self.cfg['flux_pulse_info']['on_during_drive']) and (
                not self.cfg['flux_pulse_info']['on_during_readout']):

            flux_width = max(pulse_span_length - self.cfg['flux_pulse_info']['flux_drive_delay'], 0)
            flux_comp_width = flux_width  # self.cfg['flux_pulse_info']['dc_comp_pulse_length']
            flux_delay = self.cfg['flux_pulse_info']['flux_drive_delay'] + hw_delay
            flux_idle = self.cfg['readout']['delay'] + self.cfg['readout']['width'] + 100

        elif (not self.cfg['flux_pulse_info']['on_during_drive']) and (
        self.cfg['flux_pulse_info']['on_during_readout']):

            flux_width = self.cfg['readout']['delay'] + self.cfg['readout']['width']
            flux_comp_width = flux_width  # self.cfg['flux_pulse_info']['dc_comp_pulse_length']
            flux_delay = hw_delay + pulse_span_length
            flux_idle = 100.0

        else:
            flux_width = 0
            flux_comp_width = 0
            flux_delay = 0
            flux_idle = 0

        flux_a = self.cfg['flux_pulse_info']['flux_a']
        flux_freq = self.cfg['flux_pulse_info']['flux_freq']

        flux_mod_freq = self.cfg['flux_pulse_info']['flux_mod_freq']
        flux_mod_a = self.cfg['flux_pulse_info']['flux_mod_a']

        #print 'desired flux_width= ', flux_width, ', delay=', flux_delay

        flux_total_span_list = []
        for ii in range(8):

            flux_comp_a = - flux_a[ii]  # flux_area/float(flux_comp_width)

            if flux_width > 0:
                self.psb.append('flux_'+str(ii), 'general', 'square', amp=flux_a[ii],
                            length = flux_width, freq = flux_freq[ii],
                            delay = flux_delay)

                # # hard rise/end for now
                # self.psb.append('flux_'+str(ii), 'general', 'linear_ramp_with_mod', start_amp=flux_a[ii],
                #                 stop_amp=flux_a[ii], length=flux_width, \
                #                 mod_amp=flux_mod_a[ii], mod_freq=flux_mod_freq[ii], mod_start_phase=0.0, delay = flux_delay)

            if flux_comp_width > 0:
                self.psb.append('flux_' + str(ii), 'general', 'square', amp=flux_comp_a,
                            length = flux_comp_width, freq = flux_freq[ii],
                            delay = flux_delay + flux_idle)

            # also clears
            flux_total_span_list.append(self.psb.get_total_pulse_span_length())

        return max(flux_total_span_list)

    def add_flux_pulses_hack(self, idx, pulse_span_length):

        # only for ramsey fast flux scope

        if idx >= len(self.expt_pts):
            # cal pts
            return 0.0

        else:
            # this is to align flux pulse to readout? (diff in 2 pxdac cards)
            hw_delay = self.cfg['flux_pulse_info']['pxdac_hw_delay']

            flux_width = self.expt_cfg['flux_width']
            flux_comp_width = flux_width

            self.cfg['pulse_info']['square']['ramp_sigma'] = 0.0

            gap = 50
            flux_delay = self.cfg['pulse_info']['gauss']['half_pi_length']*4 + gap + hw_delay
            flux_idle = (pulse_span_length - flux_delay - flux_width) \
                        + self.cfg['readout']['delay'] + self.cfg['readout']['width'] + 100

            flux_a = self.cfg['flux_pulse_info']['flux_a']
            flux_freq = self.cfg['flux_pulse_info']['flux_freq']
            flux_mod_freq = self.cfg['flux_pulse_info']['flux_mod_freq']
            flux_mod_a = self.cfg['flux_pulse_info']['flux_mod_a']

            flux_total_span_list = []
            for ii in range(8):

                flux_comp_a = - flux_a[ii]  # flux_area/float(flux_comp_width)

                if flux_width > 0:
                    self.psb.append('flux_'+str(ii), 'general', 'square', amp=flux_a[ii],
                                length = flux_width, freq = flux_freq[ii],
                                delay = flux_delay)

                if flux_comp_width > 0:
                    self.psb.append('flux_' + str(ii), 'general', 'square', amp=flux_comp_a,
                                length = flux_comp_width, freq = flux_freq[ii],
                                delay = flux_delay + flux_idle)

                # also clears
                flux_total_span_list.append(self.psb.get_total_pulse_span_length())

            return max(flux_total_span_list)

    def add_heterodyne_pulses(self, hetero_read_freq = None, hetero_a = None):

        # todo: seems to have bug here? not the same w/ or w/o hetero_read_freq & hetero_a
        # print hetero_read_freq
        if hetero_read_freq is not None:

            het_carrier_freq = hetero_read_freq - self.cfg['readout']['heterodyne_freq']
            het_read_freq_list = array([hetero_read_freq])
            if hetero_a is None:
                het_a_list = array([self.cfg['readout']['heterodyne_a']])
            else:
                het_a_list = array([hetero_a])
            het_IFreqList = het_read_freq_list - het_carrier_freq

        else:
            if self.cfg['readout']['is_multitone_heterodyne']:
                het_carrier_freq = self.cfg['readout']['heterodyne_carrier_freq']
                het_read_freq_list = array(self.cfg['readout']['heterodyne_freq_list'])
                het_a_list = array(self.cfg['readout']['heterodyne_a_list'])
                het_IFreqList = het_read_freq_list - het_carrier_freq
            else:
                het_carrier_freq = self.cfg['readout']['frequency'] - self.cfg['readout']['heterodyne_freq']
                het_read_freq_list = array([self.cfg['readout']['frequency']])
                het_a_list = array([self.cfg['readout']['heterodyne_a']])
                het_IFreqList = het_read_freq_list - het_carrier_freq

        # print('het_carrier_freq', het_carrier_freq)
        # print('het_read_freq_list', het_read_freq_list)
        # print('het_a_list', het_a_list)
        # print('het_IFreqList', het_IFreqList)

        het_phase_list = [(self.cfg['readout']['start_phase'] + self.cfg['readout']['phase_slope'] * ii )%360
                          for ii in het_read_freq_list]
        # het_phase_list = [0.0 for ii in het_IFreqList]

        if sum(het_a_list) > 1:
            print('Warning! Sum of heterodyne amplitudes > 1 in QubitPulseSequence.')

        if not self.cfg['readout']['is_fast_awg']:

            for ii in range(len(het_IFreqList)):
                # q2 pulses are hacked to be fixed in time, so can append multiple pulses for heterodyne readout
                self.psb.append('hetero', 'general', 'square', amp= het_a_list[ii],
                                length=self.cfg['readout']['width'],
                                freq= het_IFreqList[ii],
                                phase=het_phase_list[ii],
                                delay=self.cfg['readout']['width']/2.0 + self.cfg['readout']['delay'])
        else:

            # heterodyne carrier - LO
            self.psb.append('hetero_carrier', 'general', 'square', amp=self.cfg['readout']['hetero_carrier_a'],
                            length=self.cfg['readout']['width'],
                            freq=het_carrier_freq,
                            delay=self.cfg['readout']['width'] / 2.0 + self.cfg['readout']['delay'])
            if self.cfg['readout']['is_hetero_phase_ref']:
                # hetero phase reference to solve alazar jitter
                self.psb.append('hetero_carrier', 'general', 'square', amp=self.cfg['readout']['hetero_phase_ref_a'],
                                length=self.cfg['readout']['width'],
                                freq=self.cfg['readout']['hetero_phase_ref_freq'],
                                delay=self.cfg['readout']['width'] / 2.0 + self.cfg['readout']['delay'])
            # fast awg: read_freq
            for ii in range(len(het_IFreqList)):
                # pulses are hacked to be fixed in time, so can append multiple pulses for heterodyne readout
                self.psb.append('hetero', 'general', 'square', amp= het_a_list[ii],
                                length=self.cfg['readout']['width'],
                                freq=het_read_freq_list[ii],
                                phase=het_phase_list[ii],
                                delay=self.cfg['readout']['width'] / 2.0 + self.cfg['readout']['delay'])

    def build_sequence(self):

        PulseSequence.build_sequence(self)

        self.waveforms_dict = {}
        self.waveforms_tpts_dict = {}

        for awg in self.awg_info:
            for waveform in awg['waveforms']:
                self.waveforms_dict[waveform['name']] = self.waveforms[waveform['name']]
                self.waveforms_tpts_dict[waveform['name']] = self.get_waveform_times(waveform['name'])

        start_time = time.time()
        print('\nStart building sequences...(QubitPulseSequence.py)')

        self.psb.prepare_build(self.waveforms_tpts_dict,self.waveforms_dict)

        generated_sequences = self.psb.build(self.pulse_sequence_matrix,self.total_flux_pulse_span_length_list)
        self.waveforms_dict = generated_sequences

        for waveform_key in self.waveforms_dict:
            self.waveforms[waveform_key] = self.waveforms_dict[waveform_key]

        end_time = time.time()
        print('Finished building sequences in', end_time - start_time, 'seconds.\n')

        # np.save('S:\\_Data\\160711 - Nb Tunable Coupler\\data\\waveform.npy',self.waveforms['qubit drive Q'])
        ### in ipython notebook: call np.load('file_path/file_name.npy')

        if self.cfg["visdom_plot_seq"]:

            print('Plotting sequences in Visdom...')
            print('Plot seq list (cfg):', self.cfg["visdom_plot_seq_list"])

            viz = visdom.Visdom()
            assert viz.check_connection(), "Visdom server not connected!"
            # added two environments "seq_builder.json", and "live_plot.json" in C:\Users\slab\.visdom
            eid = "seq_builder"
            viz.close(win=None, env=eid)

            for sequence_id in self.cfg["visdom_plot_seq_list"]: #range(len(self.pulse_sequence_matrix)):

                win = viz.line(
                    X=np.arange(0, 1),
                    Y=np.arange(0, 1),
                    env=eid,
                    opts=dict( height=750, width=1800, title='Seq #%d' % sequence_id, showlegend=True, xlabel='Time to origin (ns)'))

                for waveform_key in self.waveforms_dict:

                    viz.updateTrace(
                        X= self.waveforms_tpts_dict[waveform_key] - self.psb.origin,
                        Y= self.waveforms_dict[waveform_key][sequence_id] ,
                        env=eid, win = win, name = waveform_key, append = False)


    def reshape_data(self, data):
        return np.reshape(data, (self.sequence_length, self.waveform_length))