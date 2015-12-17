# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 16:14:17 2014

@author: Gerwin K. 
"""
from slab import *
from slab.instruments import InstrumentManager, Alazar
from slab.instruments.Alazar import AlazarConfig
from numpy import *
from data_cache import dataCacheProxy
import shutil, time
import msvcrt
try:
    from liveplot import LivePlotClient
except:
    print "Error in loading liveplot"

class wigglewiggle():
    def __init__(self, msmtname, datapath, scriptname,
                 scriptpath=r'S:\_Data\141224 - M011 Helium Channels v4\experiment_M011_HeliumChannels'):

        self.im = InstrumentManager()
        #self.plotter = LivePlotClient()

        # ############################################
        # ### SETTINGS FOR THE PATH, SAVING ETC. #####
        # ############################################

        self.datapath = datapath
        self.msmtname = msmtname
        self.scriptname = scriptname
        self.scriptpath = scriptpath

        # ############################################
        ################ NWA SETTINGS ###############
        #############################################

        self.na_cfg = {
            'na_power': -30,
            'na_ifbw': 1000,
            'na_start': 4.0E9,
            'na_stop': 8.0E9,
            'na_sweep_pts': 1601,
            'na_avgs': 20,
            'na_delay': 0,
            'na_power_threshold': 0
        }

        self.sa_cfg = {
            'sa_center': 500E3,
            'sa_span': 1000E3,
            'sa_resbw': None,
            'sa_vidbw': 1E3,
            'sa_sweep_pts': 1601,
            'sa_avgs': 10,
            'sa_remote': False}

        self.labbrick_cfg = {
            'lb_freq': 7.785E9,
            'lb_power': -30
        }

        self.jpa_cfg = {
            'flux_bias_limit': 2E-3
        }

        #self.jpa_bias = im['JPA_bias']
        #self.na = im['NWA']
        #self.fridge = im['FRIDGE']
        #self.heman = im['heman']
        #self.lb = im['labbrick']

        #self.sa = im['SA']

        #Hacky solution to get the spectrum analyzer to work
        #from slab.instruments import E4440
        #self.sa = E4440("E4440", address="192.168.14.152")

        self.today = time.strftime('%y%m%d')
        self.timestamp = time.strftime('%H%M%S')

    def find_nearest(self, array, value):
        """
        Finds the nearest value in array. Returns index of array for which this is true.
        """
        idx=(np.abs(array-value)).argmin()
        return idx

    def initiate_instrument(self, name, cfg_name):
        im = InstrumentManager()
        setattr(self, name, im[cfg_name])
        return True

    def create_new_datafolder(self, datapath, measurement_name, date, timestamp):

        if not date in os.listdir(os.path.join(datapath)):
            os.mkdir(os.path.join(datapath, date))
            time.sleep(0.1)

        datafoldername = timestamp + '_' + measurement_name
        os.mkdir(os.path.join(datapath, date, datafoldername))

        return os.path.join(datapath, date, datafoldername)


    def create_new_file(self, datapath, measurement_name):
        datafoldername = self.create_new_datafolder(datapath, measurement_name, self.today, self.timestamp)

        FileHandler, fname = self.create_file_for_write(datafoldername, measurement_name)

        shutil.copy(os.path.join(self.scriptpath, self.scriptname),
                    os.path.join(datafoldername, self.scriptname))

        return datafoldername, fname, FileHandler


    def create_file_for_write(self, datafoldername, measurement_name):
        fname = get_next_filename(datafoldername + '\\', measurement_name, suffix='.h5')
        print 'Created file %s...' % fname
        FileHandler = dataCacheProxy(expInst=measurement_name, filepath=os.path.join(datafoldername, fname + '.h5'))
        return FileHandler, fname


    def test_instruments(self, do_not_test=None):
        """
        Test if instruments are responding
        """
        heman_query = 'get_manifold_status'
        fridge_query = 'get_temperatures'
        bnc845_query = 'get_frequency'
        nwa_query = 'get_start_frequency'
        bncawg_query = 'get_frequency'
        digatt_query = 'get_attenuation'

        instruments = ['NWA', 'heman', 'FRIDGE', 'digi_att', 'BNC845_RF1', 'BNC_sync', 'BNC_drive']
        queries = [nwa_query, heman_query, fridge_query, digatt_query, bnc845_query, bncawg_query, bncawg_query]
        no_error = True

        for instrument, query in zip(instruments, queries):
            if instrument not in do_not_test:
                try:
                    getattr(self.im[instrument], query)()
                    print "Established connection with {}".format(instrument)
                    time.sleep(0.1)
                except:
                    print "ERROR for {}".format(instrument)
                    no_error = False

        if no_error:
            return True
        else:
            return False

    def power_ok(self, power, threshold=0):
        if power > threshold:
            print "Power exceeded threshold of %d dBm. Aborting measurement." % threshold
            return False
        else:
            return True


    def flux_bias_ok(self, current, threshold):
        if np.abs(current) > threshold:
            print "Current exceeds threshold of %d mA. Aborting measurement." % threshold
            return False
        else:
            return True


    def setup_alazar(self, config_dict):
        self.alazar = Alazar()

        try:
            print "new configuration file"
            cfg = AlazarConfig(config_dict)
            print "config file has to pass through the AlazarConfig middleware."
            self.alazar.configure(cfg)
            return True
        except:
            return False


    def take_single_trace(self, FileHandler, fstart, fstop, power, ifbw, sweep_pts,
                          noof_avgs, verbose=True, save=True):
        """
        Uses dataCacheProxy as FileHandler or None if saving is not required.

        Take a single trace with the network analyzer. Specify inputs in the
        dictionary na_cfg. Verbose gives response
        """

        if self.power_ok(power, threshold=self.na_cfg['na_power_threshold']):

            if verbose: print "Configuring the NWA"
            self.na.set_start_frequency(fstart)
            self.na.set_stop_frequency(fstop)
            self.na.set_power(power)
            self.na.set_ifbw(ifbw)
            self.na.set_sweep_points(sweep_pts)
            self.na.set_averages(noof_avgs)

            if noof_avgs > 1: self.na.set_average_state(True)

            time.sleep(0.1)

            if verbose: print "Taking data"
            fpoints, mags, phases = self.na.take_one_averaged_trace()

            if save:
                FileHandler.post('fpoints', fpoints)
                FileHandler.post('mags', mags)
                FileHandler.post('phases', phases)
                self.save_nwa_config(FileHandler)
            else:
                return fpoints, mags, phases


    def nwa_power_sweep(self, FileHandler, pwr_start, pwr_stop, steps, save=True):
        """
        Uses dataCacheProxy as FileHandler

        Sweep the power of the network analyzer using 
        the function take_single_trace. Specify a starting, ending point
        for the power and the number of steps. 
        """
        p = np.linspace(pwr_start, pwr_stop, steps)
        print "Starting NWA powersweep..."

        fpoints = np.empty((steps, self.na_cfg['na_sweep_pts']), dtype=np.float64)
        mags = np.empty((steps, self.na_cfg['na_sweep_pts']), dtype=np.float64)
        phases = np.empty((steps, self.na_cfg['na_sweep_pts']), dtype=np.float64)

        self.plotter.clear()

        for sweep_idx, P in enumerate(p):
            print "\tP = %.2f dBm (%.1f%%)" % (P, sweep_idx / float(steps) * 100)
            f, db, phi = self.take_single_trace(FileHandler,
                                                self.na_cfg['na_start'],
                                                self.na_cfg['na_stop'],
                                                P,
                                                self.na_cfg['na_ifbw'],
                                                self.na_cfg['na_sweep_pts'],
                                                self.na_cfg['na_avgs'],
                                                verbose=False, save=False)

            self.plotter.append_z('phase', phi, ((p[0], p[1]-p[0]), (f[0], f[1] - f[0])))
            self.plotter.append_z('mag', db, ((p[0], p[1]-p[0]), (f[0], f[1] - f[0])))

            fpoints[sweep_idx, :] = f
            mags[sweep_idx, :] = db
            phases[sweep_idx, :] = phi

            if save:
                FileHandler.post('fpoints', f)
                FileHandler.post('mags', db)
                FileHandler.post('phases', phi)
                FileHandler.post('powers', P)

            if msvcrt.kbhit() and msvcrt.getch() == 'q':
                break

        self.na.set_power(min(p))

        if save:
            self.save_nwa_config(FileHandler)
        else:
            return fpoints, mags, phases


    def flux_bias_sweep(self, FileHandler, current_start, current_stop, steps, save=True):
        """
        Uses dataCacheProxy as FileHandler

        Sweep the current from the yokogawa sourc e. The
        linear sweep starts at current_start and stop at 
        current_stop. Resolution is determined by steps. 
        Flag save to save the data. 

        All currents in units of A, i.e. 1 mA = 1E-3
        """

        i = np.linspace(current_start, current_stop, steps)
        print "Starting Yoko currentsweep..."

        self.jpa_bias.set_mode(mode='curr')
        self.jpa_bias.set_current(0)
        self.jpa_bias.set_output()

        fpoints = np.empty((steps, self.na_cfg['na_sweep_pts']), dtype=np.float64)
        mags = np.empty((steps, self.na_cfg['na_sweep_pts']), dtype=np.float64)
        phases = np.empty((steps, self.na_cfg['na_sweep_pts']), dtype=np.float64)

        self.plotter.clear()

        for sweep_idx, I in enumerate(i):

            if self.flux_bias_ok(I, self.jpa_cfg['flux_bias_limit']):
                self.jpa_bias.set_current(I)
                time.sleep(0.2)
                print "\tI = %.6f mA (%.1f%%)" % (I * 1E3, sweep_idx / float(steps) * 100)
                f, db, phi = self.take_single_trace(FileHandler,
                                                    self.na_cfg['na_start'],
                                                    self.na_cfg['na_stop'],
                                                    self.na_cfg['na_power'],
                                                    self.na_cfg['na_ifbw'],
                                                    self.na_cfg['na_sweep_pts'],
                                                    self.na_cfg['na_avgs'],
                                                    verbose=False, save=False)

                fpoints[sweep_idx, :] = f
                mags[sweep_idx, :] = db
                phases[sweep_idx, :] = phi

                if save:
                    FileHandler.append('fpoints', f)
                    FileHandler.append('mags', db)
                    FileHandler.append('phases', phi)
                    FileHandler.append('currents', I)

                self.plotter.append_z('phase', phi, ((1, 1), (f[0], f[1] - f[0])))
                self.plotter.append_z('mag', db, ((1, 1), (f[0], f[1] - f[0])))

                if msvcrt.kbhit() and msvcrt.getch() == 'q':
                    break
            else:
                break

        self.jpa_bias.set_output(0)
        self.jpa_bias.set_current(0)

        if save:
            self.save_nwa_config(FileHandler)

        return fpoints, mags, phases


    def level_meter_expt(self, datapath, measurement_name, noof_puffs, start_stop_pairs=None, manual=True, save=True,
                         pressure_limit=1.5E-2, reps_per_puff=1, puff_pressure=0.25, max_temp=60E-3, pressure_drop_timeout=600):
        """
        Fills the helium lines with a number of puffs specified by noof_puffs. 
        The time between each puff is specified by wait_time, and is 1s by default. 
        After each puff a spectrum is taken and after noof_puffs puffs the file is saved.
        """
        user_is_awake = raw_input("Starting helium puff sequence. Press 'c' to continue.")
        datafoldername = self.create_new_datafolder(datapath, measurement_name, self.today, self.timestamp)
        shutil.copy(os.path.join(self.scriptpath, self.scriptname),
                    os.path.join(datafoldername, self.scriptname))

        self.plotter.clear()
        if user_is_awake == 'c' and self.heman.get_pressure() < pressure_limit:
            # Prepare file for write
            if save:
                FileHandler = dataCacheProxy(expInst=measurement_name,
                                             filepath=os.path.join(datafoldername, '%s.h5' % measurement_name))
                time.sleep(0.1)

            print "Helium puff sequence about to start..."

            puff = np.arange(noof_puffs)

            for puff_idx in puff:

                if manual:
                    another_puff = raw_input("Press 'c' for another puff")
                    if another_puff == 'c':
                        pass
                    else:
                        break
                else:
                    settled = False
                    start_time = time.time()

                    while not settled:
                        print "...",
                        settled = (self.fridge.get_mc_temperature() < max_temp) and ((time.time() - start_time) > 60)
                        #start pumping
                        self.heman.set_cryostat(False)
                        self.heman.set_gas(False)
                        self.heman.set_pump(True)
                        time.sleep(1)

                self.heman.puff(pressure=puff_pressure, min_time=60, timeout=pressure_drop_timeout)
                print "\n%s Puff %d (%.1f%% done)" % (time.strftime("%H%M%S"), puff_idx + 1,
                                                      (puff_idx + 1) / float(noof_puffs) * 100)

                if start_stop_pairs is None:

                    points, mags, phases = self.take_single_trace(None,
                                                                  self.na_cfg['na_start'],
                                                                  self.na_cfg['na_stop'],
                                                                  self.na_cfg['na_power'],
                                                                  self.na_cfg['na_ifbw'],
                                                                  self.na_cfg['na_sweep_pts'],
                                                                  self.na_cfg['na_avgs'],
                                                                  verbose=False, save=False)

                    if save:
                        FileHandler.new_stack()
                        for r in range(reps_per_puff):
                            FileHandler.post('fpoints', points)
                            FileHandler.post('mags', mags)
                            FileHandler.post('phases', phases)

                            points, mags, phases = self.take_single_trace(None,
                                                                          self.na_cfg['na_start'],
                                                                          self.na_cfg['na_stop'],
                                                                          self.na_cfg['na_power'],
                                                                          self.na_cfg['na_ifbw'],
                                                                          self.na_cfg['na_sweep_pts'],
                                                                          self.na_cfg['na_avgs'],
                                                                          verbose=False, save=False)



                        FileHandler.post('puff_nr', self.heman.get_puffs())
                        FileHandler.set_dict('Temperatures', self.fridge.get_temperatures())

                    self.plotter.append_z('mags', mags, ((1, 1), (points[0], points[1] - points[0])))
                    self.plotter.append_z('phases', phases, ((1, 1), (points[0], points[1] - points[0])))
                    self.plotter.append_y('puffs', self.heman.get_puffs())

                else:
                    for idx, f_zoom in enumerate(start_stop_pairs):
                        self.na_cfg['na_start'] = f_zoom[0]
                        self.na_cfg['na_stop'] = f_zoom[1]
                        points, mags, phases = self.take_single_trace(None,
                                                                      self.na_cfg['na_start'],
                                                                      self.na_cfg['na_stop'],
                                                                      self.na_cfg['na_power'],
                                                                      self.na_cfg['na_ifbw'],
                                                                      self.na_cfg['na_sweep_pts'],
                                                                      self.na_cfg['na_avgs'],
                                                                      verbose=False, save=False)

                        if save:
                            FileHandler.new_stack() if idx == 0 else None
                            FileHandler.post('fpoints_%d' % idx, points)
                            FileHandler.post('mags_%d' % idx, mags)
                            FileHandler.post('phases_%d' % idx, phases)
                            FileHandler.post('puff_nr_%d' % idx, self.heman.get_puffs())
                            FileHandler.set_dict('Temperatures', self.fridge.get_temperatures())

                if save:
                    self.save_nwa_config(FileHandler)
                    print "Done writing to stack %s..." % FileHandler.current_stack
                else:
                    return points, mags, phases

        else:
            if self.heman.get_pressure() > pressure_limit:
                print "Measurement aborted, pressure > %.2e mbar" % pressure_limit
            else:
                print "Measurement aborted by user."


    def get_sa_trace(self):
        """
        Get single trace from the spectrum analyzer
        :return: fpoints, magnitude
        """

        if self.sa_cfg['sa_avgs'] > 1:
            self.sa.set_average_state(True)
        else:
            self.sa.set_average_state(False)

        self.sa.configure(start=None, stop=None,
                          center=self.sa_cfg['sa_center'],
                          span=self.sa_cfg['sa_span'],
                          resbw=self.sa_cfg['sa_resbw'],
                          vidbw=self.sa_cfg['sa_vidbw'],
                          sweep_pts=self.sa_cfg['sa_sweep_pts'],
                          avgs=self.sa_cfg['sa_avgs'],
                          remote=self.sa_cfg['sa_remote'])

        return self.sa.take_one()


    def fire_filament(self, amplitude, offset, frequency, pulse_length):
        self.initiate_instrument('fil', 'fil')
        self.fil.setup_driver(amplitude, offset, frequency, pulse_length)
        self.fil.fire_filament()


    def monitor_manifold_pressure(self, heman_name = 'heman', runtime = None, dt = 1):
        """

        :param heman_name: Name as it appears in the instrument manager
        :param runtime: For how long should the pressure be monitored?
        :param dt: Timestep for the recording
        :return:
        """
        self.initiate_instrument('heman', heman_name)
        self.plotter.clear()

        t0 = time.time()
        while time.time()-t0 < runtime:
            self.heman.get_pressure()
            time.sleep(dt)

            self.plotter.append_xy('Puffs', time.time()-t0, self.heman.get_puffs())
            self.plotter.append_xy('Pressure (mbar)', time.time()-t0, self.heman.get_pressure()*1E3)
            self.plotter.append_xy('Pump valve state', time.time()-t0, self.heman.get_manifold_status_bits()[1])
            self.plotter.append_xy('Cryostat valve state', time.time()-t0, self.heman.get_manifold_status_bits()[2])
            self.plotter.append_xy('Gas valve state', time.time()-t0, self.heman.get_manifold_status_bits()[0])


    def monitor_cryostat(self, runtime, dt):
        import time
        self.initiate_instrument('fridge', 'FRIDGE')
        self.plotter.clear()

        t0 = time.time()
        while time.time()-t0 < runtime:
            pressures = self.fridge.get_pressures()
            temperatures = self.fridge.get_temperatures()

            time.sleep(dt)
            T = time.time()

            # Pressures
            self.plotter.append_xy('Condense pressure', T-t0, pressures['Condense'])
            self.plotter.append_xy('Forepump pressure', T-t0, pressures['Forepump'])
            self.plotter.append_xy('Tank pressure', T-t0, pressures['Tank'])

            # Temperatures
            self.plotter.append_xy('PT2 plate', T-t0, temperatures['PT2 Plate'])
            self.plotter.append_xy('Still', T-t0, temperatures['Still'])
            self.plotter.append_xy('100 mK plate', T-t0, temperatures['100mK Plate'])
            self.plotter.append_xy('MC RuO2', T-t0, temperatures['MC RuO2'])
            self.plotter.append_xy('MC cernox', T-t0, temperatures['MC cernox'])


    def save_sa_config(self, FileHandler):
        """
        Uses datacache
        :param FileHandler: dataCacheProxy instance
        :return: None
        """
        FileHandler.set_dict("sa_config", self.sa_cfg)


    def save_nwa_config(self, FileHandler):
        """
        Uses datacache
        :param FileHandler: dataCacheProxy instance
        :return: None
        """
        FileHandler.set_dict("nwa_config", self.na_cfg)

