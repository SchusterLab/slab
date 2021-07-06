__author__ = 'Nelson'

from slab import *
from slab.instruments.Alazar import Alazar
from slab.experiments.Alex.General.PulseSequences.SingleQubitPulseSequences import *
from slab.experiments.Alex.Multimode.PulseSequences.MultimodePulseSequence import *
from numpy import mean, arange
import numpy as np
from tqdm import tqdm
from slab.instruments.awg.PXDAC4800 import PXDAC4800
from slab.instruments.pulseblaster.pulseblaster_alex import *
import visdom


class QubitPulseSequenceExperiment(Experiment):
    '''
    Parent class for all the single qubit pulse sequence experiment.
    '''

    def __init__(self, path='', prefix='SQPSE', config_file=None, PulseSequence=None, pre_run=None, post_run=None,
                 **kwargs):

        self.extra_args = {}

        for key, value in kwargs.items():
            self.extra_args[key] = value
            # print str(key) + ": " + str(value)

        if 'liveplot_enabled' in self.extra_args:
            self.liveplot_enabled = self.extra_args['liveplot_enabled']
        else:
            self.liveplot_enabled = False

        if 'data_prefix' in self.extra_args:
            data_prefix = self.extra_args['data_prefix']
        else:
            data_prefix = prefix

        Experiment.__init__(self, path=path, prefix=data_prefix, config_file=config_file, **kwargs)

        if 'prep_tek2' in self.extra_args:
            self.prep_tek2 = self.extra_args['prep_tek2']
        else:
            self.prep_tek2 = False

        if 'adc' in self.extra_args:
            self.adc = self.extra_args['adc']
        else:
            self.adc = None

        if 'data_file' in self.extra_args:
            self.data_file = self.extra_args['data_file']
        else:
            self.data_file = None
        self.slab_file = self.datafile(data_file=self.data_file)

        if 'flux_freq' in self.extra_args:
            self.flux_freq = self.extra_args['flux_freq']
        else:
            self.flux_freq = None

        if 'readout_freq' in self.extra_args:
            self.readout_freq = self.extra_args['readout_freq']
        else:
            self.readout_freq = self.cfg['readout']['frequency']

        self.prefix = prefix
        self.expt_cfg_name = prefix.lower()

        self.pre_run = pre_run
        self.post_run = post_run

        self.pulse_type = self.cfg[self.expt_cfg_name]['pulse_type']
        self.pulse_sequence = PulseSequence(prefix, self.cfg, self.cfg[self.expt_cfg_name], **kwargs)

        self.pulse_sequence.build_sequence()

        self.expt_pts = self.pulse_sequence.expt_pts
        self.cfg['alazar']['samplesPerRecord'] = 2 ** (self.cfg['readout']['width'] - 1).bit_length()
        self.cfg['alazar']['recordsPerBuffer'] = self.pulse_sequence.sequence_length
        self.cfg['alazar']['recordsPerAcquisition'] = int(
            self.pulse_sequence.sequence_length * min(self.cfg[self.expt_cfg_name]['averages'], 100))

        if self.cfg["upload_exp"]:
            print('upload_exp = True, awg seq upload started...\n')
            self.pulse_sequence.write_sequence(os.path.join(self.path, '../sequences/'), prefix, upload=True)
        else:
            print('upload_exp = False, awg seq upload skipped.\n')

        self.ready_to_go = True

        return self.ready_to_go

    def go(self):

        # if self.liveplot_enabled:
        #     self.plotter.clear()

        print("Prep Instruments")

        try:
            if self.cfg['readout']['is_multitone_heterodyne']:
                self.readout.set_frequency(self.cfg['readout']['heterodyne_carrier_freq'])
            else:
                self.readout.set_frequency(self.readout_freq - self.cfg['readout']['heterodyne_freq'])
            self.readout.set_power(self.cfg['readout']['power'])
            self.readout.set_ext_pulse(mod=self.cfg['readout']['mod'])
            self.readout.set_output(True)
        except:
            print("No readout found.")

        try:
            if self.cfg['readout']['is_multitone_heterodyne']:
                self.readout_shifter.set_phase(self.cfg['readout']['start_phase'], self.cfg['readout']['heterodyne_carrier_freq'])
            else:
                self.readout_shifter.set_phase(self.cfg['readout']['start_phase'] + self.cfg['readout']['phase_slope'] * (
                    self.cfg['readout']['frequency'] - self.cfg['readout']['bare_frequency']),
                                               self.cfg['readout']['frequency'])
        except:
            print("Digital phase shifter not loaded.")

        try:
            self.drive.set_frequency(
                self.cfg['qubit']['frequency'] - self.cfg['pulse_info'][self.pulse_type]['iq_freq'])
            self.drive.set_power(self.cfg['drive']['power'])
            self.drive.set_ext_pulse(mod=self.cfg['drive']['mod'])
            self.drive.set_output(True)
        except:
            print("No drive found")

        try:
            self.readout_atten.set_attenuator(self.cfg['readout']['dig_atten'])
            print("Readout digital attenuator set to:", self.cfg['readout']['dig_atten'])
        except:
            print("Digital attenuator not loaded.")

        try:
            self.cfg['freq_flux']['flux'] = self.extra_args['flux']
        except:
            pass

        try:
            self.cfg['freq_flux']['freq_flux_slope'] = self.extra_args['freq_flux_slope']
        except:
            pass

        try:
            self.cfg['freq_flux']['flux_offset'] += self.extra_args['flux_offset']
        except:
            pass

        try:
            self.awg.set_amps_offsets(self.cfg['cal']['iq_amps'], self.cfg['cal']['iq_offsets'])
        except:
            print("self.awg not loaded.")

        if self.cfg['take_data']:
            print('\nTaking data with filename:', self.slab_file.filename, '\n')
            self.take_data()
        else:
            print('\ntake_data = False, data acquisition skipped.')

        if self.cfg['stop_awgs'] == True:
            print('stop_awg = True, seqs stopped.')
        else:
            self.awg_run()
            print('stop_awg = False, seqs left running.')


    def take_data(self):

        ####
        if self.cfg["visdom_plot_livedata"]:

            viz = visdom.Visdom()
            assert viz.check_connection(), "Visdom server not connected!"
            # added two environments "seq_builder.json", and "live_plot.json" in C:\Users\slab\.visdom
            eid = "live_plot"
            viz.close(win=None, env=eid)

            win1 = viz.line( X=np.arange(0, 1), Y=np.arange(0, 1), env=eid,
                opts=dict(height=400, width=700, title='expt_avg_data', showlegend=True, xlabel='expt_pts'))
            win2 = viz.line( X=np.arange(0, 1), Y=np.arange(0, 1), env=eid,
                opts=dict(height=400, width=700, title='expt_avg_data2', showlegend=True, xlabel='expt_pts'))
            win3 = viz.line( X=np.arange(0, 1), Y=np.arange(0, 1), env=eid,
                opts=dict(height=400, width=700, title='single_record (of only first run)', showlegend=True, xlabel='time ns'))
            win4 = viz.scatter(X=array([[1,2,3],[1,2,3]]).transpose(), env=eid,
                        opts=dict(height=400, width=700, title='g/e/f cal single shot', showlegend=True, xlabel=''))

        ####

        if self.pre_run is not None:
            self.pre_run()

        TEST_REDPITAYA = False

        if TEST_REDPITAYA:
            self.awg_run()

        if not TEST_REDPITAYA:

            if self.adc == None:
                print("Prep Card")
                adc = Alazar(self.cfg['alazar'])
            else:
                adc = self.adc

            # debug_alazer = False
            # if debug_alazer:
            #
            #     # save raw time trace, 100 runs only
            #     tpts, single_data1, single_data2 = adc.acquire_singleshot_data2()
            #
            #     self.slab_file = self.datafile(data_file=self.data_file)
            #     with self.slab_file as f:
            #         f.add('tpts', tpts)
            #         f.add('single_data1', single_data1)
            #         f.add('single_data2', single_data2)
            #         f.close()
            #
            #     # hack for post_run
            #     expt_avg_data = self.expt_pts
            #
            # else:

            if not self.cfg['readout']['save_single-shot_data']:

                expt_data = None
                current_data = None
                for ii in tqdm(arange(max(1, self.cfg[self.expt_cfg_name]['averages'] / 100))):
                    tpts, ch1_pts, ch2_pts = adc.acquire_avg_data_by_record(prep_function=self.awg_prep,
                                                                            start_function=self.awg_run,
                                                                            excise=self.cfg['readout']['window'])

                    if not self.cfg[self.expt_cfg_name]['use_pi_calibration']:

                        if expt_data is None:
                            if self.cfg['readout']['channel'] == 1:
                                expt_data = ch1_pts
                            elif self.cfg['readout']['channel'] == 2:
                                expt_data = ch2_pts
                        else:
                            if self.cfg['readout']['channel'] == 1:
                                expt_data = (expt_data * ii + ch1_pts) / (ii + 1.0)
                            elif self.cfg['readout']['channel'] == 2:
                                expt_data = (expt_data * ii + ch2_pts) / (ii + 1.0)

                        if self.cfg['readout']['heterodyne_freq'] == 0:
                            # homodyne
                            expt_avg_data = mean(expt_data, 1)
                        else:
                            # heterodyne
                            heterodyne_freq = self.cfg['readout']['heterodyne_freq']
                            # ifft by numpy default has the correct 1/N normalization
                            expt_data_fft_amp = np.abs(np.fft.ifft(expt_data))
                            hetero_f_ind = int(round(heterodyne_freq * tpts.size * 1e-9))  # position in ifft

                            # todo: do the proper sum here
                            # expt_avg_data = np.average(expt_data_fft_amp[:, (hetero_f_ind - 1):(hetero_f_ind + 1)], axis=1)
                            expt_avg_data = expt_data_fft_amp[:, hetero_f_ind]

                    else:
                        # average first, then divide by pi_calibration values
                        if expt_data is None:
                            if self.cfg['readout']['channel'] == 1:
                                expt_data = ch1_pts[:-2]
                                zero_amp_curr = mean(ch1_pts[-2])
                                pi_amp_curr = mean(ch1_pts[-1])
                            elif self.cfg['readout']['channel'] == 2:
                                expt_data = ch2_pts[:-2]
                                zero_amp_curr = mean(ch2_pts[-2])
                                pi_amp_curr = mean(ch2_pts[-1])
                            zero_amp = zero_amp_curr
                            pi_amp = pi_amp_curr
                        else:
                            if self.cfg['readout']['channel'] == 1:
                                expt_data = (expt_data * ii + ch1_pts[:-2]) / (ii + 1.0)
                                zero_amp_curr = mean(ch1_pts[-2])
                                pi_amp_curr = mean(ch1_pts[-1])
                            elif self.cfg['readout']['channel'] == 2:
                                expt_data = (expt_data * ii + ch2_pts[:-2]) / (ii + 1.0)
                                zero_amp_curr = mean(ch2_pts[-2])
                                pi_amp_curr = mean(ch2_pts[-1])
                            zero_amp = (zero_amp * ii + zero_amp_curr) / (ii + 1.0)
                            pi_amp = (pi_amp * ii + pi_amp_curr) / (ii + 1.0)

                        # todo: add heterodyne with pi_cal
                        expt_avg_data = mean((expt_data - zero_amp) / (pi_amp - zero_amp), 1)

                    # self.slab_file = self.datafile(data_file=self.data_file)

                    with self.slab_file as f:
                        f.add('expt_2d', expt_data)
                        f.add('expt_avg_data', expt_avg_data)
                        f.add('expt_pts', self.expt_pts)

                        # save pi_cal amps, to be able to monitor fluctuations
                        if self.cfg[self.expt_cfg_name]['use_pi_calibration']:
                            f.append_pt('zero_amps', zero_amp_curr)
                            f.append_pt('pi_amps', pi_amp_curr)

                        f.close()


            else:  # here saves all single shot data

                if self.prefix == 'Vacuum_Rabi':

                    print('vacuum_rabi')
                    het_IFreqList = [self.cfg['readout']['heterodyne_freq']]
                    het_read_freq_list = [0]

                elif self.cfg['readout']['is_multitone_heterodyne']:
                    het_carrier_freq = self.cfg['readout']['heterodyne_carrier_freq']
                    het_read_freq_list = array(self.cfg['readout']['heterodyne_freq_list'])
                    het_IFreqList = het_read_freq_list - het_carrier_freq
                else:
                    het_carrier_freq = self.readout_freq - self.cfg['readout']['heterodyne_freq']
                    het_read_freq_list = array([self.readout_freq])
                    het_IFreqList = het_read_freq_list - het_carrier_freq

                avgPerAcquisition = int(min(self.cfg[self.expt_cfg_name]['averages'], 100))
                numAcquisition = int(np.ceil(self.cfg[self.expt_cfg_name]['averages'] / 100))

                # (ch1/2, exp_pts, heterodyne_freq, cos/sin, all averages)
                ss_data = zeros((2, len(self.expt_pts), len(het_IFreqList), 2, avgPerAcquisition * numAcquisition))

                # (ch1/2, heterodyne_freq, cos/sin, all averages)
                ss_cal_g = zeros((2, len(het_IFreqList), 2, avgPerAcquisition * numAcquisition))
                ss_cal_e = zeros((2, len(het_IFreqList), 2, avgPerAcquisition * numAcquisition))
                ss_cal_f = zeros((2, len(het_IFreqList), 2, avgPerAcquisition * numAcquisition))

                if self.cfg['readout']['save_trajectory_data']:

                    # (ch1/2, exp_pts, heterodyne_freq, cos/sin, all averages, traj)
                    traj_data = zeros((2, len(self.expt_pts), len(het_IFreqList), 2, avgPerAcquisition * numAcquisition, self.cfg['readout']['pts_per_traj']))

                    # (ch1/2, heterodyne_freq, cos/sin, all averages, traj)
                    traj_cal_g = zeros((2, len(het_IFreqList), 2, avgPerAcquisition * numAcquisition, self.cfg['readout']['pts_per_traj']))
                    traj_cal_e = zeros((2, len(het_IFreqList), 2, avgPerAcquisition * numAcquisition, self.cfg['readout']['pts_per_traj']))
                    traj_cal_f = zeros((2, len(het_IFreqList), 2, avgPerAcquisition * numAcquisition, self.cfg['readout']['pts_per_traj']))

                for ii in tqdm(arange(numAcquisition)):

                    if not self.cfg['readout']['is_hetero_phase_ref']:

                        # single_data1/2: index: (hetero_freqs, cos/sin, all_seqs)
                        single_data1, single_data2, single_record1, single_record2 = \
                            adc.acquire_singleshot_heterodyne_multitone_data(het_IFreqList, prep_function=self.awg_prep,
                                                                             start_function=self.awg_run,
                                                                             excise=self.cfg['readout']['window'])

                        # saving the raw time traces
                        # single_data1, single_data2, single_record1, single_record2 = \
                        #     adc.acquire_singleshot_heterodyne_multitone_data(het_IFreqList, prep_function=self.awg_prep,
                        #                                                      start_function=self.awg_run,
                        #                                                      excise=None, save_raw_data=True)

                    elif self.cfg['readout']['save_trajectory_data']:

                        excise = [ 0, self.cfg['readout']['window'][1] ]
                        single_data1, single_data2, single_record1, single_record2, single_traj1, single_traj2= \
                            adc.acquire_singleshot_heterodyne_multitone_data_phase_ref_save_traj(het_IFreqList,
                                                                                       self.cfg['readout'][
                                                                                           'hetero_phase_ref_freq'],
                                                                                       prep_function=self.awg_prep,
                                                                                       start_function=self.awg_run,
                                                                                       excise=excise,
                                                                                       isCompensatePhase=True,
                                                                                       save_raw_data=False, pts_per_traj=self.cfg['readout']['pts_per_traj'])
                    else:

                        # single_data1/2: index: (hetero_freqs, cos/sin, all_seqs)
                        single_data1, single_data2, single_record1, single_record2 = \
                            adc.acquire_singleshot_heterodyne_multitone_data_phase_ref(het_IFreqList,
                                                                             self.cfg['readout']['hetero_phase_ref_freq'],
                                                                             prep_function=self.awg_prep,
                                                                             start_function=self.awg_run,
                                                                             excise=self.cfg['readout']['window'],
                                                                             isCompensatePhase=True,
                                                                             save_raw_data=False)

                    single_data = array([single_data1, single_data2])
                    # index: (ch1/2, hetero_freqs, cos / sin, avgs, seq(exp_pts))
                    single_data = np.reshape(single_data,
                                             (single_data.shape[0], single_data.shape[1], single_data.shape[2],
                                              int(self.cfg['alazar'][
                                                  'recordsPerAcquisition'] / self.pulse_sequence.sequence_length),
                                              self.pulse_sequence.sequence_length))

                    # index: (ch1/2, exp_pts, hetero_freqs, cos / sin, avgs)
                    single_data = np.transpose(single_data, (0, 4, 1, 2, 3))

                    if (self.cfg[self.expt_cfg_name]['use_g-e-f_calibration']):

                        ss_data[:, :, :, :, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition)] = single_data[:,
                                                                                                       0:-3]
                        ss_cal_g[:, :, :, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition)] = single_data[:, -3]
                        ss_cal_e[:, :, :, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition)] = single_data[:, -2]
                        ss_cal_f[:, :, :, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition)] = single_data[:, -1]

                    elif (self.cfg[self.expt_cfg_name]['use_pi_calibration']):

                        ss_data[:, :, :, :, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition)] = single_data[:,
                                                                                                       0:-2]
                        ss_cal_g[:, :, :, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition)] = single_data[:, -2]
                        ss_cal_e[:, :, :, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition)] = single_data[:, -1]

                    else:
                        ss_data[:, :, :, :, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition)] = single_data[:, :]

                    ####
                    if self.cfg['readout']['save_trajectory_data']:

                        # index: (hetero_freqs, cos/sin, all_seqs, traj)
                        single_traj = array([single_traj1, single_traj2])

                        # index: (ch1/2, hetero_freqs, cos / sin, avgs, seq(exp_pts), traj)
                        single_traj = np.reshape(single_traj,
                                                 (single_traj.shape[0], single_traj.shape[1], single_traj.shape[2],
                                                  self.cfg['alazar']['recordsPerAcquisition'] / self.pulse_sequence.sequence_length,
                                                  self.pulse_sequence.sequence_length, single_traj.shape[-1]))

                        # index: (ch1/2, exp_pts, hetero_freqs, cos / sin, avgs, traj)
                        single_traj = np.transpose(single_traj, (0, 4, 1, 2, 3, 5))
                        # print single_traj.shape

                        if (self.cfg[self.expt_cfg_name]['use_g-e-f_calibration']):

                            traj_data[:, :, :, :, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition), :] = single_traj[
                                                                                                           :,
                                                                                                           0:-3, :]
                            traj_cal_g[:, :, :, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition), :] = single_traj[:,
                                                                                                         -3, :]
                            traj_cal_e[:, :, :, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition), :] = single_traj[:,
                                                                                                         -2, :]
                            traj_cal_f[:, :, :, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition), :] = single_traj[:,
                                                                                                         -1, :]

                        elif (self.cfg[self.expt_cfg_name]['use_pi_calibration']):

                            traj_data[:, :, :, :, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition), :] = single_traj[
                                                                                                           :,
                                                                                                           0:-2, :]
                            traj_cal_g[:, :, :, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition), :] = single_traj[:,
                                                                                                         -2, :]
                            traj_cal_e[:, :, :, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition), :] = single_traj[:,
                                                                                                         -1, :]

                        else:
                            traj_data[:, :, :, :, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition), :] = single_traj[
                                                                                                           :, :, :]

                    ####

                    # old way for easy plotting
                    # only calculate for 1st het_readout_freq
                    if het_IFreqList[0] == 0:
                        # take cos of ch1/ch2
                        expt_avg_data = mean(ss_data[0, :, 0, 0, 0:((ii + 1) * avgPerAcquisition)], -1)
                        expt_avg_data2 = mean(ss_data[1, :, 0, 0, 0:((ii + 1) * avgPerAcquisition)], -1)

                        if self.cfg[self.expt_cfg_name]['use_pi_calibration']:
                            zero_amp_avg = mean(ss_cal_g[0, 0, 0, 0:((ii + 1) * avgPerAcquisition)], -1)
                            zero_amp_avg2 = mean(ss_cal_g[1, 0, 0, 0:((ii + 1) * avgPerAcquisition)], -1)
                            pi_amp_avg = mean(ss_cal_e[0, 0, 0, 0:((ii + 1) * avgPerAcquisition)], -1)
                            pi_amp_avg2 = mean(ss_cal_e[1, 0, 0, 0:((ii + 1) * avgPerAcquisition)], -1)

                            zero_amp_curr = mean(
                                ss_cal_g[0, 0, 0, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition)], -1)
                            zero_amp_curr2 = mean(
                                ss_cal_g[1, 0, 0, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition)], -1)
                            pi_amp_curr = mean(
                                ss_cal_e[0, 0, 0, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition)], -1)
                            pi_amp_curr2 = mean(
                                ss_cal_e[1, 0, 0, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition)], -1)

                            expt_avg_data = (expt_avg_data - zero_amp_avg) / (pi_amp_avg - zero_amp_avg)
                            expt_avg_data2 = (expt_avg_data2 - zero_amp_avg2) / (pi_amp_avg2 - zero_amp_avg2)

                    else:
                        # take cos/sin of ch1
                        expt_avg_data = mean(ss_data[0, :, 0, 0, 0:((ii + 1) * avgPerAcquisition)], -1)
                        expt_avg_data2 = mean(ss_data[0, :, 0, 1, 0:((ii + 1) * avgPerAcquisition)], -1)

                        if self.cfg[self.expt_cfg_name]['use_pi_calibration']:
                            zero_amp_avg = mean(ss_cal_g[0, 0, 0, 0:((ii + 1) * avgPerAcquisition)], -1)
                            zero_amp_avg2 = mean(ss_cal_g[0, 0, 1, 0:((ii + 1) * avgPerAcquisition)], -1)
                            pi_amp_avg = mean(ss_cal_e[0, 0, 0, 0:((ii + 1) * avgPerAcquisition)], -1)
                            pi_amp_avg2 = mean(ss_cal_e[0, 0, 1, 0:((ii + 1) * avgPerAcquisition)], -1)

                            zero_amp_curr = mean(
                                ss_cal_g[0, 0, 0, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition)], -1)
                            zero_amp_curr2 = mean(
                                ss_cal_g[0, 0, 1, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition)],
                                -1)
                            pi_amp_curr = mean(
                                ss_cal_e[0, 0, 0, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition)], -1)
                            pi_amp_curr2 = mean(
                                ss_cal_e[0, 0, 1, (ii * avgPerAcquisition):((ii + 1) * avgPerAcquisition)], -1)

                            expt_avg_data = (expt_avg_data - zero_amp_avg) / (pi_amp_avg - zero_amp_avg)
                            expt_avg_data2 = (expt_avg_data2 - zero_amp_avg2) / (pi_amp_avg2 - zero_amp_avg2)

                    # this needs to stay here
                    # if self.data_file != None:
                    #     self.slab_file = SlabFile(self.data_file)
                    # else:
                    #     self.slab_file = self.datafile()
                    self.slab_file = self.datafile(data_file=self.data_file)

                    with self.slab_file as f:
                        f.add('ss_data', ss_data[:, :, :, :, 0:((ii + 1) * avgPerAcquisition)])
                        f.add('single_record1', single_record1)
                        f.add('single_record2', single_record2)

                        if (self.cfg[self.expt_cfg_name]['use_pi_calibration']):
                            f.add('ss_cal_g', ss_cal_g[:, :, :, 0:((ii + 1) * avgPerAcquisition)])
                            f.add('ss_cal_e', ss_cal_e[:, :, :, 0:((ii + 1) * avgPerAcquisition)])
                        if (self.cfg[self.expt_cfg_name]['use_g-e-f_calibration']):
                            f.add('ss_cal_g', ss_cal_g[:, :, :, 0:((ii + 1) * avgPerAcquisition)])
                            f.add('ss_cal_e', ss_cal_e[:, :, :, 0:((ii + 1) * avgPerAcquisition)])
                            f.add('ss_cal_f', ss_cal_f[:, :, :, 0:((ii + 1) * avgPerAcquisition)])

                        if self.cfg['readout']['save_trajectory_data']:
                            f.add('traj_data', traj_data[:, :, :, :, 0:((ii + 1) * avgPerAcquisition), :])
                            if (self.cfg[self.expt_cfg_name]['use_pi_calibration']):
                                f.add('traj_cal_g', traj_cal_g[:, :, :, 0:((ii + 1) * avgPerAcquisition), :])
                                f.add('traj_cal_e', traj_cal_e[:, :, :, 0:((ii + 1) * avgPerAcquisition), :])
                            if (self.cfg[self.expt_cfg_name]['use_g-e-f_calibration']):
                                f.add('traj_cal_g', traj_cal_g[:, :, :, 0:((ii + 1) * avgPerAcquisition), :])
                                f.add('traj_cal_e', traj_cal_e[:, :, :, 0:((ii + 1) * avgPerAcquisition), :])
                                f.add('traj_cal_f', traj_cal_f[:, :, :, 0:((ii + 1) * avgPerAcquisition), :])

                        f.add('expt_avg_data', expt_avg_data.flatten())
                        f.add('expt_avg_data2', expt_avg_data2.flatten())

                        f.add('expt_pts', self.expt_pts)
                        f.add('het_read_freq_list', het_read_freq_list)

                        if self.cfg[self.expt_cfg_name]['use_pi_calibration']:
                            f.append_pt('zero_amps', zero_amp_curr)
                            f.append_pt('pi_amps', pi_amp_curr)
                            f.append_pt('zero_amps2', zero_amp_curr2)
                            f.append_pt('pi_amps2', pi_amp_curr2)

                        f.close()

                    ####
                    if self.cfg["visdom_plot_livedata"]:
                        viz.updateTrace(X=self.expt_pts, Y=expt_avg_data, env=eid, win=win1, append=False)
                        viz.updateTrace(X=self.expt_pts, Y=expt_avg_data2, env=eid, win=win2, append=False)

                        # if (self.cfg[self.expt_cfg_name]['use_pi_calibration']):
                        #
                        #     f_idx = 0
                        #     viz.scatter(X=array([ss_cal_g[0, f_idx, 0, :],ss_cal_g[0, f_idx, 1, :]]).transpose(),
                        #                     Y=None, env=eid, win=win4, name='test')
                        #
                        # if (self.cfg[self.expt_cfg_name]['use_g-e-f_calibration']):
                        #     pass

                        if ii==0:
                            viz.updateTrace(X=array(range(len(single_record1))), Y=single_record2, env=eid, win=win3, name='2', append=False)
                            viz.updateTrace(X=array(range(len(single_record1))), Y=single_record1, env=eid, win=win3, name='1', append=False)
                    ####

            if self.post_run is not None:
                self.post_run(self.expt_pts, expt_avg_data)

            if self.cfg['stop_awgs'] == True:
                self.awg_prep()
                print('stop_awg = True, seqs stopped.')
            else:
                print('stop_awg = False, seqs left running.')

            # closes Alazar card and releases buffer
            adc.close()

    def awg_prep(self):

        stop_pulseblaster()

        self.im['M8195A'].stop_output()

        for key, value in LocalInstruments().inst_dict.items():
            value.stop()

    def awg_run(self):

        # # hack to fix attenuator jumping
        # try:
        #     im['atten2'].set_attenuator(self.cfg['readout']['dig_atten'])
        #     time.sleep(1.0)
        # except:
        #     print "Digital attenuator not loaded."

        self.im['M8195A'].start_output()

        for key, value in LocalInstruments().inst_dict.items():
            value.run_experiment()

        time.sleep(1)
        run_pulseblaster()