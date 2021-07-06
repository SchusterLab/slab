__author__ = 'dave'

from slab import *

from slab.instruments.awg.MultimodePulseFluxRabiSequences import MultimodeFluxSideBandRabiSequence
from numpy import mean, arange


class MultimodeFluxSideBandRabiExperiment(Experiment):
    def __init__(self, path='', prefix='MM_Flux_Sideband_Rabi', config_file=None, use_cal=False, **kwargs):
        Experiment.__init__(self, path=path, prefix=prefix, config_file=config_file, **kwargs)


        self.pulse_type = self.cfg['mm_flux_sideband_rabi']['pulse_type']

        self.mm_flux_sideband_cfg = self.cfg['mm_flux_sideband']

        if self.mm_flux_sideband_cfg['freq_step'] is not None:
            self.mm_flux_sideband_freq_pts = arange(self.mm_flux_sideband_cfg['start_freq'], self.mm_flux_sideband_cfg['stop_freq'], self.mm_flux_sideband_cfg['freq_step'])
        else:
            self.mm_flux_sideband_freq_pts = linspace(self.mm_flux_sideband_cfg['start_freq'], self.mm_flux_sideband_cfg['stop_freq'], self.mm_flux_sideband_cfg['freq_num_pts'])

        if self.cfg['pulse_info'][self.pulse_type] is None:
            print("This pulse type is not valid.")
            self.ready_to_go = False
            return

        pulse_calibrated = self.cfg['pulse_info'][self.pulse_type]['rabi_calibrated']

        if not pulse_calibrated:
            print("This pulse type has not been calibrated.")
            self.ready_to_go = False
            return

        self.pulse_sequence = MultimodeFluxSideBandRabiSequence(self.cfg['awgs'], self.cfg['mm_flux_sideband_rabi'], self.cfg['readout'],self.cfg['pulse_info'][self.pulse_type])
        self.pulse_sequence.build_sequence()
        self.pulse_sequence.write_sequence(os.path.join(self.path, 'sequences/'), prefix, upload=True)

        self.mm_flux_sideband_pts = self.pulse_sequence.mm_flux_sideband_pts
        #self.cfg['alazar']['samplesPerRecord'] = self.pulse_sequence.waveform_length
        self.cfg['alazar']['recordsPerBuffer'] = self.pulse_sequence.sequence_length
        self.cfg['alazar']['recordsPerAcquisition'] = int(
            self.pulse_sequence.sequence_length * min(self.cfg['mm_flux_sideband']['averages'], 100))

        self.ready_to_go = True

        ### Hard coding TEK7
        self.tek7 = InstrumentManager()["TEK2"]

        return

    def go(self,adc):
        self.plotter.clear('MM Flux Sideband Rabi Data')
        self.plotter.clear('MM Flux Sideband Rabi XY')

        # self.save_config()

        print("Prep Instruments")
        self.readout.set_frequency(self.cfg['readout']['frequency'])
        self.readout.set_power(self.cfg['readout']['power'])
        self.readout.set_ext_pulse(mod=True)
        self.readout_shifter.set_phase(self.cfg['readout']['start_phase'] + self.cfg['readout']['phase_slope'] * (
            self.cfg['readout']['frequency'] - self.cfg['readout']['bare_frequency']), self.cfg['readout']['frequency'])

        self.drive.set_frequency(self.cfg['qubit']['frequency'] - self.cfg['mm_flux_sideband']['freq'])
        self.drive.set_power(self.cfg['mm_flux_sideband']['power'])
        self.drive.set_ext_pulse(mod=True)
        self.drive.set_output(True)
        self.readout_atten.set_attenuator(self.cfg['readout']['dig_atten'])

        self.awg.set_amps_offsets(self.cfg['cal']['iq_amps'], self.cfg['cal']['iq_offsets'])


        mm_flux_sideband_data = None

        #print "Doing Flux Sideband with frequency: " + str(self.flux_freq)
        for ii in arange(max(1, self.cfg['mm_flux_sideband_rabi']['averages'] / 100)):
            ### putting TEK2 to first waveform before awg runs, not sure if this is a good way to do so
            tpts, ch1_pts, ch2_pts = adc.acquire_avg_data_by_record(prep_function=self.awgs_prep,
                                                                    start_function=self.awg.run,
                                                                    excise=self.cfg['readout']['window'])
            # tpts, ch1_pts, ch2_pts = adc.acquire_data_by_record(start_function=awg.run, excise=None)
            # if self.cfg.alazar["ch1_enabled"]: self.plotter.plot_xy('current ch1', tpts, ch1_pts)
            # if self.cfg.alazar["ch1_enabled"]: self.plotter.plot_xy('current ch2', tpts, ch2_pts)
            if mm_flux_sideband_data is None:
                mm_flux_sideband_data = ch1_pts
            else:
                mm_flux_sideband_data = (mm_flux_sideband_data * ii + ch1_pts) / (ii + 1.0)

            self.plotter.plot_z('MM Flux Sideband Rabi Data', mm_flux_sideband_data.T)
            mm_flux_sideband_avg_data = mean(mm_flux_sideband_data, 1)
            self.plotter.plot_xy('MM Flux Sideband Rabi XY', self.pulse_sequence.mm_flux_sideband_pts, mm_flux_sideband_avg_data)
            print(ii * min(self.cfg['mm_flux_sideband']['averages'], 100))

        #self.plotter.append_z('MM Flux Sideband Rabi Freq Sweep',mm_flux_sideband_avg_data)
        with self.datafile() as f:
            #f.append_pt('flux_freq',self.flux_freq)
            f.append_line('mm_flux_sideband_avg_data', mm_flux_sideband_avg_data)
            f.append_line('mm_flux_sideband_pts', self.mm_flux_sideband_pts)
            f.close()


    ## hard coding tek7 preparation, why need to stop and run, but not just preping it??
    def awgs_prep(self):
        print("Preparing TEKs")
        self.awg.stop_and_prep()
        self.tek7.stop()
        self.tek7.prep_experiment()
        self.tek7.run()




