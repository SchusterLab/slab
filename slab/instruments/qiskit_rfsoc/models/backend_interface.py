"""
backend_interface.py
"""

import os

from qiskit.providers.backend import BackendV1 as BackendInterface
from qiskit.providers.options import Options
from qiskit.qobj.utils import MeasLevel, MeasReturnType

from .json_util import PulseEncoder
from .backend import JOB_ID_ERROR

QUEUE_KEYS = [
    "job_id"
]

class SLabBackendInterface(BackendInterface):
    """
    Talk to a backend through this class.

    References
    [0] https://github.com/Qiskit/qiskit-ibmq-provider/blob/master/
        qiskit/providers/ibmq/ibmqbackend.py
    """
    
    def __init__(self, provider, url):
        # get config
        config_url = os.path.join(url, "pulse_backend_configuration.json")
        config_res = requests.get(config_url)
        defaults_res.raise_for_status()
        config_dict = json.loads(config_res.json())
        config = PulseBackendConfiguration.from_dict(config_dict)
        
        # get defaults
        defaults_url = os.path.join(url, "pulse_defaults.json")
        defaults_res = requests.get(defaults_url)
        defaults_res.raise_for_status()
        defaults_dict = json.loads(defaults_res.json())
        defaults = PulseDefaults.from_dict(defaults_dict)

        # initialize
        super().__init__(configuration=config, provider=provider)
        self._defaults = defaults
        self.url = url
        self.job_queue_url = os.path.join(url, "job-queue")
        self.job_retrieve_url = os.path.join(url, "job-retrieve")
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
        # post to /job-queue
        out_dict = qobj.to_dict()
        qobj_id = out_dict["qobj_id"]
        out_payload = json.dumps(qobj_dict, ensure_ascii=False, cls=PulseEncoder)
        headers = {"Content-Type": "application/json"}
        res = requests.post(self.job_queue_url, headers=headers, data=out_payload)
        
        # get job_id
        res.raise_for_status()
        in_dict = json.loads(res.json())
        job_id = in_dict["job_id"]
        if job_id == JOB_ID_ERROR:
            raise Exception("Backend errored in job_queue POST.")
        #ENDIF
        job = SLabJob(self, job_id, qobj_id)

        return job
    #ENDDEF
#ENDCLASS

