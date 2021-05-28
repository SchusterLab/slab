"""
backend_interface.py
"""

from qiskit.providers.backend import BackendV1 as BackendInterface
from qiskit.providers.options import Options
from qiskit.qobj.utils import MeasLevel, MeasReturnType

class SLabBackendInterface(BackendInterface):
    """
    Talk to a backend through this class.

    References
    [0] https://github.com/Qiskit/qiskit-ibmq-provider/blob/master/
        qiskit/providers/ibmq/ibmqbackend.py
    """
    
    def __init__(self, configuration, defaults, ip, provider):
        super().__init__(configuration=configuration, provider=provider)
        self._defaults = defaults
        self.ip = ip
    #ENDDEF

    def defaults(self):
        return self._defaults
    #ENDDEF

    @classmethod
    def _default_options(cls):
        return Options(
            shots=1024, memory=False,
            qubit_lo_freq=None, meas_lo_freq=None,
            schedule_los=None,
            meas_level=MeasLevel.KERNELED,
            meas_return=MeasReturnType.AVERAGE,
            memory_slots=None, memory_slot_size=100,
            rep_time=None, rep_delay=None,
            init_qubits=True
        )
    #ENDDEF

    def run(self, qobj):
        qobj_dict = qobj.to_dict()
        # TODO: post to actual backend
        return qobj_dict
    #ENDDEF
#ENDCLASS

