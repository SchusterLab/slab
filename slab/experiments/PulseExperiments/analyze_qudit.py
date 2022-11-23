import numpy as np
# import matplotlib.pyplot as plt
from copy import deepcopy
# import os


class Spliner:
    def __init__(self, t, Ns):
        self.t = t
        self.Ns = Ns
        t0 = np.min(t)
        self._dtknot = (np.max(t)-t0)/(Ns - 2.0)
        self._width = 3.0 * self._dtknot
        self._tcenter = self._dtknot * (np.linspace(1, Ns, Ns) - 1.5) + t0
        self.bsplines = self._get_bsplines()
    
    def bspline_base(tau):
        if tau < -0.5 or tau >= 0.5:
            return 0.0
        if -0.5 <= tau and tau < -1.0/6.0:
            return 9.0/8.0 + 9.0/2.0*tau + 9.0/2.0*tau*tau
        if -1.0/6.0 <= tau and tau < 1.0/6.0:
            return 3.0/4.0 - 9.0*tau*tau
        if 1.0/6.0 <= tau and tau < 0.5:
            return 9.0/8.0 - 9.0/2.0*tau + 9.0/2.0*tau*tau    
        
    vbspline_base = np.vectorize(bspline_base)
    
    # def bspline(self, s):
    #     tau = (self.t - self._tcenter[s]) / self._width
    #     return Spliner.vbspline_base(tau)

    def bspline(self, t0, s):
        tau = (t0 - self._tcenter[s]) / self._width
        return Spliner.bspline_base(tau)
    
    def _get_bsplines(self):
        times_mesh, tcenter_mesh = np.meshgrid(self.t, self._tcenter)
        tau_mesh = (times_mesh - tcenter_mesh) / self._width
        return Spliner.vbspline_base(tau_mesh).T
    
    def assemble_spline(self, c):
        assert len(c) == self.Ns, "one control param per Bspline, len(c) == Ns"
        return self.bsplines @ c
    
    def decompose_spline(self, S):
        """ least square fit """
        assert len(self.t) == len(S), "t and S must have same length: S(t)"
        assert len(self.t) >= self.Ns, "should have more than Ns t values (so full rank of B^T B)"
        B = self.bsplines
        # return np.linalg.inv(B.T @ B) @ B.T @ S
        return np.linalg.lstsq(B, S)


class Pulser:
    """
    Class to build and decompose pulses on Q qubits
    """
    def __init__(self, carrier_freq, t=None, Ns=None, spliner=None, rot_freq=None):
        """
        carrier_freq: nested list: Q sublists, kth list contains carrier freqs (NOT ANGULAR, in GHz) for kth qubit
        rot_freq: array of rotation frequencies, length = Q. If None, then all zeros.
        t: points in time to represent pulse (in ns)
        Ns: number of Bspline basis functions
        spliner: Spliner object for building and decomposing splines
        
        Need to define either t and Ns or spliner
        """ 
        if isinstance(carrier_freq, list) and not (isinstance(carrier_freq[0], list) or isinstance(carrier_freq[0], np.ndarray)):
            # carrier_freq is 1d list with carrier frequencies
            carrier_freq = [carrier_freq]
        self.Omega_carr = [2 * np.pi * np.array(freq) for freq in carrier_freq]
        if rot_freq is not None:
            self.omega_rot = 2 * np.pi * np.array(rot_freq)
        else:
            self.omega_rot = np.zeros(len(self.Omega_carr))
        if t is not None and Ns is not None:
            self.spliner = Spliner(t, Ns)
        elif spliner is not None:
            self.spliner = spliner
        else:
            raise Exception("must either specifiy t and Ns or spliner")

    def apply_filter_correction(self, alpha, filter_func):
        """
        alpha: complex parameter array as defined in Quandary (however, here real and imaginary parts already reduced to complex numbers), length = Ns * total number of carrier frequencies over all qubits
        filter_func: function gamma(k, w) that maps drive frequency (in GHz) to amplitude correction for qudit k=0,1,...; e.g. A cos(wt) -> A gamma(k, w) cos(wt)
        """
        Q = len(self.Omega_carr)
        Nf = [len(omegas) for omegas in self.Omega_carr]
        Ns = self.spliner.Ns

        if alpha.size == 2*Q*Ns*Nf[0]:
            alpha = alpha[::2] + 1j*alpha[1::2]

        assert alpha.size == Q*Ns*Nf[0], "alpha has wrong size"

        alpha_ = np.empty(alpha.size, dtype=complex)
    
        for k in range(Q):
            sum_Nf_less_k = int(np.sum(Nf[0:k]))
            alpha_k = deepcopy(alpha[(Ns * sum_Nf_less_k):(Ns * (sum_Nf_less_k + Nf[k]))])
            alpha_k = alpha_k.reshape((Ns, Nf[k]))
            for j in range(Nf[k]):
                alpha_k[:,j] *= filter_func(k, (self.Omega_carr[k][j] + self.omega_rot[k])/2.0/np.pi)
            alpha_[(Ns * sum_Nf_less_k):(Ns * (sum_Nf_less_k + Nf[k]))] = alpha_k.reshape(Ns*Nf[k])
        
        return alpha_

    
    def params_to_controls(self, alpha, phase_shift=None, print_maxctrl=False, return_subwaves=False, rot_frame_invariant=False, combine_controls=False):
        """
        Assemble pulse from parameters.
        alpha: complex parameter array as defined in Quandary (however, here real and imaginary parts already reduced to complex numbers), length = Ns * total number of carrier frequencies over all qubits
        phase_shift: phase offset phi for each control and each carrier wave, e.g. f_k(t) = A cos(wt + phi)
                     has same dimensions as self.Omega_carr, list of Q items where each item is a list of phases for each carrier wave of that control
        """
        Q = len(self.Omega_carr)
        Nf = [len(omegas) for omegas in self.Omega_carr]

        if phase_shift is None:
            phase_shift = [np.zeros(Nf[k]) for k in range(Q)]
        else:
            phase_shift = [np.array(phases) for phases in phase_shift]
            
        Ns = self.spliner.Ns
        Nt = self.spliner.t.size
        if rot_frame_invariant:
            t0 = self.spliner.t.min()
        else:
            t0 = 0

        if alpha.size == 2*Q*Ns*Nf[0]:
            alpha = alpha[::2] + 1j*alpha[1::2]

        assert alpha.size == Q*Ns*Nf[0], "alpha has wrong size"

        if return_subwaves:
            Nf0 = max(Nf)
            d = np.zeros((Q, Nt, Nf0), dtype=complex)
        else:
            d = np.zeros((Q, Nt), dtype=complex)

        for k in range(Q):
            sum_Nf_less_k = int(np.sum(Nf[0:k]))
            alpha_k = alpha[(Ns * sum_Nf_less_k):(Ns * (sum_Nf_less_k + Nf[k]))]
            alpha_k = alpha_k.reshape((Ns, Nf[k])) 

            if print_maxctrl:
                print(f"k = {k}")
                for i, Om in enumerate(self.Omega_carr[k]):
                    max_re = np.abs(alpha_k[:,i].real).max()
                    max_im = np.abs(alpha_k[:,i].imag).max()
                    print(f"   {Om}: {max(max_re, max_im)}")

            S_k = self.spliner.assemble_spline(alpha_k)
            osc_k = np.exp(1j * np.outer(self.spliner.t - t0, self.Omega_carr[k]))

            if return_subwaves:
                d[k,:,:Nf[k]] = osc_k * S_k * np.exp(1j * phase_shift[k])
            else:
                d[k] = np.sum(osc_k * S_k * np.exp(1j * phase_shift[k]), axis=1)

        if return_subwaves:
            f = 2 * np.real(d.swapaxes(1,2).swapaxes(0,1) * np.exp(1j * np.outer(self.omega_rot, self.spliner.t))).swapaxes(0,1).swapaxes(1,2)
        else:
            f = 2 * np.real(d * np.exp(1j * np.outer(self.omega_rot, self.spliner.t)))

        if combine_controls:
            f_ = np.zeros((1, Nt))
            d_ = np.zeros((1, Nt), dtype=complex)
            f_[0] = f.sum(0)
            d_[0] = d.sum(0)
            f = f_
            d = d_

        return f

    def params_to_control_functions(self, alpha):
        """
        Assemble pulse functions from parameters.
        alpha: complex parameter array as defined in Quandary (however, here real and imaginary parts already reduced to complex numbers), length = Ns * total number of carrier frequencies over all qubits
        """
        Q = len(self.Omega_carr)
        Nf = [len(omegas) for omegas in self.Omega_carr]
        Ns = self.spliner.Ns

        def d(t, k):
            s_start = int(t/self.spliner._dtknot)
            sum_Nf_less_k = int(np.sum(Nf[0:k]))
            range_start = Ns * sum_Nf_less_k + Nf[k] * s_start
            range_end = range_start + 3*Nf[k]
            alpha_nonzero = alpha[range_start:range_end]
            alpha_nonzero = alpha_nonzero.reshape((3, Nf[k])) 
            spline_vec = np.array([self.spliner.bspline(t, s) for s in np.arange(3) + s_start])
            osc_k = np.exp(1j * self.Omega_carr[k] * t)
            return np.dot(spline_vec, np.dot(alpha_nonzero, osc_k))
        
        def p(t, k):
            return np.real(d(t, k))
        def q(t, k):
            return np.imag(d(t, k))
        def f(t, k):
            return 2*np.real(d(t, k) * np.exp(1j * self.omega_rot[k] * t))
        return np.vectorize(p, excluded=['k']), np.vectorize(q, excluded=['k']), np.vectorize(f, excluded=['k'])

    def controls_to_params(self, d=None, p=None, q=None):
        """
        Decompose pulse (rotating frame!) in parameters.
        d = p + iq: complex 2D array, shape (Q, number of time points). d[k] is pulse for kth qubit
        p: real part of d, same shape
        q: imag part of d, same shape

        Need to define either p and q or d.
        """
        if d is None:
            if p is not None and q is not None:
                d = p + 1j * q
            else:
                raise Exception("need to define either p and q or d")
        # d is not None
        if len(d.shape) == 1:
            d = np.array([d])
        Q = len(self.Omega_carr)
        Nf = [len(omegas) for omegas in self.Omega_carr]
        Ns = self.spliner.Ns
        Nt = self.spliner.t.size

        alpha = np.empty(Ns*np.sum(Nf), dtype=complex)
        for k in range(Q):
            osc_k = np.exp(1j * np.outer(self.spliner.t, self.Omega_carr[k])) 
            X = np.empty((Nt, Ns*Nf[k]), dtype=complex)
            for s in range(Ns):
                X[:,s*Nf[k]:(s+1)*Nf[k]] = (osc_k.T * self.spliner.bsplines[:,s]).T

            sum_Nf_less_k = int(np.sum(Nf[0:k]))
            # use numpy least squares here, because alpha = (X^T X)^(-1) X^T d produces too much error
            alpha[(Ns * sum_Nf_less_k):(Ns * (sum_Nf_less_k + Nf[k]))] = np.linalg.lstsq(X, d[k], rcond=None)[0]
        return alpha

    def shrextend_pulse(self, t_new, Ns_new, alpha=None, d=None, p=None, q=None, fit_alpha=False, return_pulser=False):
        """
        Build new shrunk or extended (shrextended) pulse from this optimized pulse.
        Will have same carrier frequencies and time discretization.
        Nt_new: new number of time steps 
        Ns_new: new number of splines for parametrizing new pulse
        fit_alpha: If False, just append/remove Bsplines and do not perform lstsq fit
        """
        if alpha is None:
            if d is None:
                if p is not None and q is not None:
                    d = p + 1j * q
                else:
                    raise Exception("need to define either alpha or p and q or d")
            alpha = self.controls_to_params(d=d)

        if fit_alpha:
            # copy pulser for modeling new pulse
            pulser = deepcopy(self)
            dt_new = t_new[1] - t_new[0]
            Nt_new = t_new.size
            T = self.spliner.t.max()
            _Nt = int(np.round(T / dt_new, decimals=0)) + 1
            _t = np.linspace(0, T, _Nt)
            pulser.spliner = Spliner(_t, pulser.spliner.Ns)
            _p, _q, _ = pulser.params_to_controls(alpha)
            _d = _p + 1j * _q

            if Nt_new <= _Nt:
                # shrink
                _d = _d[:,:Nt_new]
            else:
                # extend = append 0's to pulse
                zero_add = np.zeros((_d.shape[0], Nt_new-_Nt))
                _d = np.concatenate((_d.T, zero_add.T)).T
            pulser.spliner = Spliner(t_new, Ns_new)
            alpha_new = pulser.controls_to_params(d=_d)
        else:
            Q = len(self.Omega_carr)
            Nf = [len(omegas) for omegas in self.Omega_carr]
            Ns = self.spliner.Ns
            _Ns = min(Ns, Ns_new)

            alpha_new = np.zeros(int(alpha.size/(Ns*Ns_new)), dtype=complex)

            for k in range(Q):
                sum_Nf_less_k = int(np.sum(Nf[0:k]))
                alpha_new[(Ns_new*sum_Nf_less_k):(Ns_new*sum_Nf_less_k + _Ns*Nf[k])] = alpha[(Ns*sum_Nf_less_k):(Ns*sum_Nf_less_k + _Ns*Nf[k])]

            pulser = deepcopy(self)
            pulser.spliner = Spliner(t_new, Ns_new)

        if return_pulser:
            return alpha_new, pulser
        else:
            return alpha_new


# class Analyzer:
#     def __init__(self, data_dir, config_file="config_log.dat", param_file="params.dat", control_file="control.dat"):
#         self.data_dir = data_dir if data_dir[-1] == "/" else data_dir + "/"
#         self.config_file = self.data_dir + config_file
#         self.param_file = self.data_dir + param_file
#         self.control_file = self.data_dir + control_file
#         self.pulser = None
#         self.Q = None
#         self.configs = {}
#         self._get_configs()
    
#     def _get_configs(self):
#         with open(self.config_file) as cf:
#             for line in cf.readlines():
#                 # beware of case that right side has = signs
#                 parts = line.split('=')
#                 left = parts[0]
#                 right = "=".join(parts[1:])
#                 self.configs[left.strip(' \t\n')] = right.strip(' \t\n')  
#         self.Q = np.fromstring(self.configs['nlevels'], sep=',').size
#         Nt = int(self.configs['ntime']) + 1
#         T = (Nt - 1) * float(self.configs['dt'])
#         Ns = int(self.configs['nspline'])
#         t = np.linspace(0, T, Nt)
#         rot_freq = np.fromstring(self.configs['rotfreq'], sep=',')
#         carrier_freq = [None] * self.Q
#         for k in range(self.Q):
#             carrier_freq[k] = np.fromstring(self.configs[f'carrier_frequency{k}'], sep=',')
#         self.pulser = Pulser(carrier_freq, rot_freq, t, Ns)
    
#     def params_to_controls(self, save_file=None):
#         """returns t, p(t), q(t), f(t)"""    
#         alpha = np.loadtxt(self.param_file)
#         alpha = alpha[::2] + 1j * alpha[1::2]

#         p, q, f = self.pulser.params_to_controls(alpha)
        
#         if save_file is not None:
#             assert save_file[-4] == '.', "file name must have 3 character extension"
#             for k in range(self.Q):
#                 sf_k = self.data_dir + save_file[:-4] + str(k) + save_file[-4:]
#                 header = "# time\tp(t) (rotating)\tq(t) (rotating)\tf(t) (labframe)"
#                 matrix = np.vstack((t, p[k], q[k], f[k])).T
#                 np.savetxt(sf_k, matrix, delimiter='\t', header=header, fmt='%.14e')
#         return self.pulser.spliner.t, p, q, f

#     def controls_to_params(self):
#         control_path = self.control_file[:-4] + str(0) + self.control_file[-4:]
#         p, q = np.loadtxt(control_path, skiprows=1, usecols=(1,2), unpack=True)
#         p = np.array([p])
#         q = np.array([q])
#         for k in range(1, self.Q):
#             control_path = self.control_file[:-4] + str(k) + self.control_file[-4:]
#             _p, _q = np.loadtxt(control_path, skiprows=1, usecols=(1,2), unpack=True)
#             p = np.vstack((p, _p))
#             q = np.vstack((q, _q))
#         d = p + 1j * q
#         return self.pulser.controls_to_params(d)

#     def shrextend_pulse(self, t_new, Ns_new, return_pulser=False):
#         """
#         Build new shrunk or extended (shrextended) pulse from this optimized pulse.
#         Will have same carrier frequencies and time discretization.
#         Nt_new: new number of time steps 
#         Ns_new: new number of splines for parametrizing new pulse
#         """
#         alpha = np.loadtxt(self.param_file)
#         alpha = alpha[::2] + 1j * alpha[1::2]
#         return self.pulser.shrextend_pulse(t_new, Ns_new, alpha=alpha, return_pulser=return_pulser)
    
#     def plot_pulses(self, from_params=False, fig_title=None, save_file=None):
#         if from_params:
#             t, p, q, f = self.params_to_controls()
#         else:
#             control_path = self.control_file[:-4] + str(0) + self.control_file[-4:]
#             t, p, q, f = np.loadtxt(control_path, skiprows=1, unpack=True)
#             for k in range(1, self.Q):
#                 control_path = self.control_file[:-4] + str(k) + self.control_file[-4:]
#                 _p, _q, _f = np.loadtxt(control_path, skiprows=1, usecols=(1,2,3), unpack=True)
#                 p = np.vstack((p, _p))
#                 q = np.vstack((q, _q))
#                 f = np.vstack((f, _f))
        
#         plt.rc('text', usetex=True)
#         plt.rc('font', family='serif')
#         fig = plt.figure(figsize=(16,10))
#         gs = fig.add_gridspec(ncols=2, nrows=2)
#         if fig_title is not None:
#             fig.suptitle(fig_title, fontsize=16)

#         ax_p = fig.add_subplot(gs[0,0])
#         ax_p.set_title(r"rot. frame $p(t)$")
#         ax_p.set_xlabel(r"$t$ in ns")
#         ax_p.set_ylabel(r"$p(t)$ in ?")
#         ax_q = fig.add_subplot(gs[0,1])
#         ax_q.set_title(r"rot. frame $q(t)$")
#         ax_q.set_xlabel(r"$t$ in ns")
#         ax_q.set_ylabel(r"$q(t)$ in ?")
#         ax_f = fig.add_subplot(gs[1,:])
#         ax_f.set_title(r"lab frame $f(t)$")
#         ax_f.set_xlabel(r"$t$ in ns")
#         ax_f.set_ylabel(r"$f(t)$ in ?")
#         ax_p.plot(t, p.T, label=[f"Q{k}" for k in range(self.Q)])
#         ax_p.legend()
#         ax_p.grid()
#         ax_q.plot(t, q.T, label=[f"Q{k}" for k in range(self.Q)])
#         ax_q.legend()
#         ax_q.grid()
#         ax_f.plot(t, f.T, label=[f"Q{k}" for k in range(self.Q)])
#         ax_f.legend()
#         ax_f.grid()
        
#         if save_file is not None:
#             fig.savefig(self.data_dir + save_file)
#         else:
#             return fig
    
#     def plot_population(self, plot_dims=(8,5), fig_title=None, save_file=None):
#         nlevels = np.fromstring(self.configs['nlevels'], sep=',', dtype=int)
#         max_levels = nlevels.max()

#         pop_files = [f for f in os.listdir(self.data_dir) if f[:10] == "population"]
#         if not pop_files:
#             raise Exception("no population files")
#         t = np.loadtxt(self.data_dir + pop_files[0], skiprows=1, usecols=0)
#         file_parts = np.array([f.split('.') for f in pop_files], dtype=str)
#         unique_iinits = sorted(list(set(file_parts[:,1])))

#         plt.rc('text', usetex=True)
#         plt.rc('font', family='serif')
#         nr_plots_v = len(unique_iinits)
#         fig = plt.figure(figsize=np.array(plot_dims) * np.array([max_levels, nr_plots_v]))
#         gs = fig.add_gridspec(ncols=max_levels, nrows=nr_plots_v)
#         if fig_title is not None:
#             fig.suptitle(fig_title, fontsize=16)

#         prop_cycle = plt.rcParams['axes.prop_cycle']
#         colors = prop_cycle.by_key()['color']

#         for i, iinit in enumerate(unique_iinits):
#             for lvl in range(max_levels):
#                 ax = fig.add_subplot(gs[i,lvl])
#                 ax.set_title(f"pop lvl {lvl}, {iinit}")
#                 ax.set_xlabel(r"$t$ in ns")
#                 ax.set_ylabel("pop")
#                 ax.set_ylim(-0.1, 1.1)
#                 for k in range(self.Q):
#                     path = self.data_dir + f"population{k}.{iinit}.dat"
#                     try:
#                         data_k = np.loadtxt(path, skiprows=1, usecols=lvl+1)
#                         ax.plot(t, data_k, label=f"Q{k}", c=colors[k])
#                     except:
#                         pass
#                 ax.grid()
#                 ax.legend()

#         if save_file is not None:
#             fig.savefig(self.data_dir + save_file)
#         else:
#             return fig

#     def plot_energy(self, plot_dims=(6,5), from_pops=False, fig_title=None, save_file=None):
#         nlevels = np.fromstring(self.configs['nessential'], sep=',', dtype=int)
#         if len(nlevels) == 1:
#             nlevels = np.concatenate(([1], nlevels))
#         nlevels_all = np.fromstring(self.configs['nlevels'], sep=',', dtype=int)
#         maxlevel = nlevels_all.max()

#         energy_files = [f for f in os.listdir(self.data_dir) if f[:8] == "expected"]
#         if not energy_files and not from_pops:
#             print("no energy files, fall back on population files")
#             from_pops = True
        
#         t = self.pulser.spliner.t
#         if from_pops:
#             pop_files = [f for f in os.listdir(self.data_dir) if f[:10] == "population"]
#             if not pop_files:
#                 raise Exception("no population files")
#             t = np.loadtxt(self.data_dir + pop_files[0], skiprows=1, usecols=0)
#             file_parts = np.array([f.split('.') for f in pop_files], dtype=str)
#         else:
#             t = np.loadtxt(self.data_dir + energy_files[0], skiprows=1, usecols=0)
#             file_parts = np.array([f.split('.') for f in energy_files], dtype=str)
#         unique_iinits = sorted(list(set(file_parts[:,1])))

#         fig, axs = plt.subplots(nlevels[0], nlevels[1], figsize=np.array(plot_dims) * nlevels[::-1])
#         fig.subplots_adjust(hspace=0.3)
            
#         plt.rc('text', usetex=True)
#         plt.rc('font', family='serif')
#         if fig_title is not None:
#             fig.suptitle(fig_title, fontsize=16)

#         prop_cycle = plt.rcParams['axes.prop_cycle']
#         colors = prop_cycle.by_key()['color']

#         for i, iinit in enumerate(unique_iinits):
#             if nlevels[0] == 1:
#                 ax = axs[i]
#                 ax.set_title(r"Evolution of $|{}\rangle$".format(f"{i}"))
#             else:
#                 ax = axs[i//nlevels[1], i%nlevels[1]]
#                 ax.set_title(r"Evolution of $|{}\rangle$".format(f"{i//nlevels[1]}{i%nlevels[1]}"))
#             ax.set_xlabel(r"$t$ [ns]")
#             ax.set_ylabel("energy state")
#             ax.set_ylim(-0.1, maxlevel-1+0.1)
#             ax.set_yticks(np.arange(0, maxlevel, 1))
#             ax.set_yticklabels(np.arange(0, maxlevel, 1))
#             for k in range(self.Q):
#                 try:
#                     if from_pops:
#                         path = self.data_dir + f"population{k}.{iinit}.dat"
#                         nl = nlevels_all[k]
#                         data_k = np.loadtxt(path, skiprows=1, usecols=np.arange(1,nl+1,1)) @ np.arange(0,nl,1)
#                     else:
#                         path = self.data_dir + f"expected{k}.{iinit}.dat"
#                         data_k = np.loadtxt(path, skiprows=1, usecols=1)
#                     ax.plot(t, data_k, label=f"Q{k}", c=colors[k])
#                 except:
#                     pass
#             ax.grid()
#             ax.legend()

#         if save_file is not None:
#             fig.savefig(self.data_dir + save_file)
#         else:
#             return fig