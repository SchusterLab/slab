__author__ = 'Nelson'

from slab import *
from slab.instruments.Alazar import Alazar
from slab.experiments.Nitrogen.General.PulseSequences.SingleQubitPulseSequences import *
from slab.experiments.Nitrogen.Multimode.PulseSequences.MultimodePulseSequence import *
from numpy import mean, arange
from tqdm import tqdm


class QubitPulseSequenceExperiment(Experiment):
    '''
    Parent class for all the single qubit pulse sequence experiment.
    '''
    def __init__(self, path='', prefix='SQPSE', config_file=None, PulseSequence=None, pre_run=None, post_run=None,
                 **kwargs):

        self.extra_args={}

        for key, value in kwargs.items():
            self.extra_args[key] = value
            #print str(key) + ": " + str(value)

        if 'liveplot_enabled' in self.extra_args:
            self.liveplot_enabled = self.extra_args['liveplot_enabled']
        else:
            self.liveplot_enabled = True

        if 'data_prefix' in self.extra_args:
            data_prefix = self.extra_args['data_prefix']
        else:
            data_prefix = prefix

        Experiment.__init__(self, path=path, prefix=data_prefix, config_file=config_file, **kwargs)



        if 'prep_tek2' in self.extra_args:
            self.prep_tek2 = self.extra_args['prep_tek2']
        else:
            self.prep_tek2 = False

        if 'trigger_period' in self.extra_args:
            self.trigger_period = self.extra_args['trigger_period']
            print("Trigger period has been set to %s microseconds"%(self.trigger_period*1e6))
        else:
            try:
                self.trigger_period = self.cfg['expt_trigger']['period']
            except:
                print("error in setting trigger time")

        self.adc_predefined = False
        if 'adc' in self.extra_args:
            self.adc = self.extra_args['adc']
            self.adc_predefined = True
        else:
            self.adc = None

        if 'data_file' in self.extra_args:
            self.data_file = self.extra_args['data_file']
        else:
            self.data_file = None

        if 'flux_freq' in self.extra_args:
            self.flux_freq = self.extra_args['flux_freq']
        else:
            self.flux_freq = None

        self.prefix = prefix
        self.expt_cfg_name = prefix.lower()

        self.pre_run = pre_run
        self.post_run = post_run

        self.pulse_type = self.cfg[self.expt_cfg_name]['pulse_type']

        self.pulse_sequence = PulseSequence(prefix, self.cfg, self.cfg[self.expt_cfg_name],**kwargs)
        self.pulse_sequence.build_sequence()
        self.pulse_sequence.write_sequence(os.path.join(self.path, '../sequences/'), prefix, upload=True)

        self.expt_pts = self.pulse_sequence.expt_pts
        self.cfg['alazar']['samplesPerRecord'] = 2 ** (self.cfg['readout']['width'] - 1).bit_length()
        self.cfg['alazar']['recordsPerBuffer'] = self.pulse_sequence.sequence_length

        self.cfg['alazar']['recordsPerAcquisition'] = int(
            self.pulse_sequence.sequence_length * min(self.cfg[self.expt_cfg_name]['averages'], 100))

        self.ready_to_go = True
        return

    def go(self):
        if self.liveplot_enabled:
            self.plotter.clear()

        print("Prep Instruments")
        self.readout.set_frequency(self.cfg['readout']['frequency'])
        self.readout.set_power(self.cfg['readout']['power'])
        self.readout.set_ext_pulse(mod=True)
        # self.readout_shifter.set_phase(self.cfg['readout']['start_phase'] + self.cfg['readout']['phase_slope'] * (
        #     self.cfg['readout']['frequency'] - self.cfg['readout']['bare_frequency']), self.cfg['readout']['frequency'])

        self.drive.set_frequency(self.cfg['qubit']['frequency'] - self.cfg['pulse_info'][self.pulse_type]['iq_freq'])
        self.drive.set_power(self.cfg['drive']['power'])
        self.drive.set_ext_pulse(mod=True)
        self.drive.set_output(True)
        self.readout_atten.set_attenuator(self.cfg['readout']['dig_atten'])

        self.trigger.set_period(self.trigger_period)

        try:
            self.cfg['freq_flux']['flux']=self.extra_args['flux']
        except:
            pass

        try:
            self.cfg['freq_flux']['freq_flux_slope']=self.extra_args['freq_flux_slope']
        except:
            pass

        try:
            self.cfg['freq_flux']['flux_offset']+=self.extra_args['flux_offset']
        except:
            pass


        if self.cfg['freq_flux']['current']:
            try: self.flux_volt.ramp_current(self.cfg['freq_flux']['flux'])
            except:pass
        elif self.cfg['freq_flux']['voltage']:
            try: self.flux_volt.ramp_volt(self.cfg['freq_flux']['flux'])
            except: pass

        self.awg.set_amps_offsets(self.cfg['cal']['iq_amps'], self.cfg['cal']['iq_offsets'])

        if self.pre_run is not None:
            self.pre_run()

        if self.adc==None:
            print("Prep Card")
            adc = Alazar(self.cfg['alazar'])
        else:
            adc = self.adc

        expt_data = None
        current_data = None
        for ii in tqdm(arange(max(1, self.cfg[self.expt_cfg_name]['averages'] / 100))):
            tpts, ch1_pts, ch2_pts = adc.acquire_avg_data_by_record(prep_function=self.awg_prep,
                                                                    start_function=self.awg.run,
                                                                    excise=self.cfg['readout']['window'])

            ### to skip the first 100 averages data point, because we saw a weird ramp in the RF output
            if ii == 0:
                time.sleep(0.1)
                continue

            mag = sqrt(ch1_pts**2+ch2_pts**2)
            if not self.cfg[self.expt_cfg_name]['use_pi_calibration']:

                if expt_data is None:
                    if self.cfg['readout']['channel']==1:
                        expt_data = ch1_pts
                    elif self.cfg['readout']['channel']==2:
                        expt_data = ch2_pts
                else:
                    if self.cfg['readout']['channel']==1:
                        expt_data = (expt_data * ii + ch1_pts) / (ii + 1.0)
                    elif self.cfg['readout']['channel']==2:
                        expt_data = (expt_data * ii + ch2_pts) / (ii + 1.0)

                expt_avg_data = mean(expt_data, 1)


            else:

                if expt_data is None:
                    if self.cfg['readout']['channel']==1:
                        expt_data = ch1_pts
                    elif self.cfg['readout']['channel']==2:
                        expt_data = ch2_pts
                else:
                    if self.cfg['readout']['channel']==1:
                        expt_data = (expt_data * ii + ch1_pts) / (ii + 1.0)
                    elif self.cfg['readout']['channel']==2:
                        expt_data = (expt_data * ii + ch2_pts) / (ii + 1.0)

                expt_avg_data = mean(expt_data, 1)

                expt_avg_data = (expt_avg_data[:-2]-expt_avg_data[-2])/(expt_avg_data[-1]-expt_avg_data[-2])

            # temporary hack. Alazar card seems to shift two data points (problem disappeared after reboot computer)
            # expt_avg_data = np.concatenate((expt_avg_data[2:],expt_avg_data[:2]))
            # if self.cfg[self.expt_cfg_name]['use_pi_calibration']:
            #     expt_avg_data = (expt_avg_data[:-2]-expt_avg_data[-2])/(expt_avg_data[-1]-expt_avg_data[-2])

            # else:
            #     if self.cfg['readout']['channel']==1:
            #         zero_amp = mean(ch1_pts[-2])
            #         pi_amp = mean(ch1_pts[-1])
            #         current_data= (ch1_pts[:-2]-zero_amp)/(pi_amp-zero_amp)
            #     elif self.cfg['readout']['channel']==2:
            #         zero_amp = mean(ch2_pts[-2])
            #         pi_amp = mean(ch2_pts[-1])
            #         current_data= (ch2_pts[:-2]-zero_amp)/(pi_amp-zero_amp)
            #     if expt_data is None:
            #         expt_data = current_data
            #     else:
            #         expt_data = (expt_data * ii + current_data) / (ii + 1.0)


            # expt_avg_data = mean(expt_data, 1)


            if self.liveplot_enabled:
                self.plotter.plot_z(self.prefix + ' Data', expt_data.T)
                self.plotter.plot_xy(self.prefix + ' XY', self.pulse_sequence.expt_pts, expt_avg_data)

            # print ii * min(self.cfg[self.expt_cfg_name]['averages'], 100)

            if self.data_file == None:
                self.slab_file = self.datafile()
                with self.slab_file as f:
                    f.add('expt_2d', expt_data)
                    f.add('expt_avg_data', expt_avg_data)
                    f.add('expt_pts', self.expt_pts)
                    f.close()

        if self.data_file != None:
            self.slab_file = SlabFile(self.data_file)
            with self.slab_file as f:
                f.append_line('expt_avg_data', expt_avg_data)
                f.append_line('expt_pts', self.expt_pts)
                f.close()


        if not self.adc_predefined:
            adc.close()


        if self.post_run is not None:
            self.post_run(self.expt_pts, expt_avg_data)


    def awg_prep(self):
        self.awg.stop_and_prep()
        if self.prep_tek2:
            self.tek2.stop()
            self.tek2.prep_experiment()
            self.tek2.run()