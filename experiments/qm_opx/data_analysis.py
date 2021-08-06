from slab.dsfit import*
import matplotlib.pyplot as plt
from numpy import*

class PostExperiment:

    def __init__(self, experiment_name, I , Q, P = 'Q', show=True):
        self.quantum_device_cfg = quantum_device_cfg
        self.experiment_cfg = experiment_cfg
        self.hardware_cfg = hardware_cfg

        self.exptname = experiment_name
        self.I = I
        self.Q = Q
        self.P = P
        self.show = show

        try:
            temp = eval('self.' + experiment_name)()
        except:
            print("No post experiment analysis yet")

    def resonator_spectroscopy(self):
        """fit to the power"""
        amps = Q**2 + I**2
        ph = np.arctan2(np.array(Q), np.array(I))
        ph = np.unwrap(ph, discont=3.141592653589793, axis=-1)
        m = (ph[-1]-ph[0])/(x[-1] - x[0])
        ph = ph - m*x*0.95
        ph = ph -np.mean(ph)

        if self.show:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.35)
            axs[0].plot(x, I)
            axs[0].plot(x, Q)
            axs[0].set_xlabel('Freq (GHz)')
            axs[0].set_ylabel('I/Q')
            axs[1].plot(x, amps, 'b*')
            p = fitlor(x, amps, showfit=False)
            q = p[2]/(2*p[3])
            axs[1].plot(x, lorfunc(p, x), label=r'$\nu_{r}$ = %.6f GHz, Q = %.2f'% (p[2], q), color='r')
            print ("fits = ", p)
            ax2  = axs[1].twinx()
            ax2.plot(x, ph, 'g*')
            axs[1].set_xlabel('Freq (GHz)')
            axs[1].set_ylabel('Amp (a.u.)')
            ax2.set_ylabel('$\\varphi$')
            axs[1].legend(loc='center')
            plt.show()
        else: pass

        print("Resonant frequency from fitting mag squared: ", p[2], "GHz")
        print("Resonant frequency from I peak : ", f[argmax(abs(I))], "GHz")
