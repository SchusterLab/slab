from StateDiscriminator import StateDiscriminator
import numpy as np
from qm.qua import *


class TwoStateDiscriminator(StateDiscriminator):

    def __init__(self, qmm, config, update_tof, rr_qe, path):
        super().__init__(qmm, config, update_tof, rr_qe, path)
        self.num_of_states = 2

    def _update_config(self):
        weights = self.saved_data['weights']
        b_vec = weights[0, :] - weights[1, :]
        self.config['integration_weights']['demod1_iw'] = {
            'cosine': np.real(b_vec).tolist(),
            'sine': (-np.imag(b_vec)).tolist()
        }
        self._add_iw_to_all_pulses('demod1_iw')
        self.config['integration_weights'][f'demod2_iw'] = {
            'cosine': np.imag(b_vec).tolist(),
            'sine': np.real(b_vec).tolist()
        }
        self._add_iw_to_all_pulses('demod2_iw')
        if self.update_tof or self.finish_train == 1:
            self.config['elements'][self.rr_qe]['time_of_flight'] = self.config['elements'][self.rr_qe]['time_of_flight'] - \
                                                                    self.config['elements'][self.rr_qe]['smearing']

    def get_threshold(self):
        bias = self.saved_data['bias']
        return bias[0]-bias[1]

    def measure_state(self, pulse, out1, out2, res, adc=None, statistic=None):
        """
        This procedure generates a macro of QUA commands for measuring the readout resonator and discriminating between
        the states of the qubit its states.
        :param pulse: A string with the readout pulse name.
        :param out1: A string with the name first output of the readout resonator (corresponding to the real part of the
         complex IN(t) signal).
        :param out2: A string with the name second output of the readout resonator (corresponding to the imaginary part
        of the complex IN(t) signal).
        :param res: A boolean QUA variable that will receive the discrimination result (0 or 1)
        :param adc: (optional) the stream variable which the raw ADC data will be saved and will appear in result
        analysis scope.
        """

        R1 = declare(fixed)
        R2 = declare(fixed)

        measure(pulse, self.rr_qe, adc, demod.full('demod1_iw', R1, out1),
                                        demod.full('demod2_iw', R2, out2))

        assign(res, R1 + R2 < self.get_threshold())
        if statistic is not None:
            assign(statistic, R1 + R2)
