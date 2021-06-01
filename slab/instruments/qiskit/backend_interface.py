"""
backend_interface.py

References:
[0] https://github.com/Qiskit/qiskit-ibmq-provider/blob/176cc07f556c5c026a15b1aec8860bdd003733e9/qiskit/providers/ibmq/ibmqbackend.py
"""

import copy
import json
import os
import requests

from qiskit.compiler import assemble
from qiskit.providers.backend import BackendV1 as BackendInterface
from qiskit.providers.models import (PulseBackendConfiguration, PulseDefaults)
from qiskit.providers.options import Options
from qiskit.qobj import PulseQobj
from qiskit.qobj.utils import MeasLevel, MeasReturnType

from .json_util import PulseEncoder
from .backend import JOB_ID_ERROR

QUEUE_KEYS = [
    "job_id"
]

class SLabBackendInterface(BackendInterface):
    """
    Talk to a backend through this class. See [0].
    """
    
    def __init__(self, provider, url):
        # get config
        config_url = os.path.join(url, "pulse_backend_configuration.json")
        config_res = requests.get(config_url)
        config_res.raise_for_status()
        config_dict = config_res.json()
        config = PulseBackendConfiguration.from_dict(config_dict)
        
        # get defaults
        defaults_url = os.path.join(url, "pulse_defaults.json")
        defaults_res = requests.get(defaults_url)
        defaults_res.raise_for_status()
        defaults_dict = defaults_res.json()
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

    def _get_run_config(self, **kwargs):
        run_config_dict = copy.copy(self.options.__dict__)
        for key, val in kwargs.items():
            if val is not None:
                run_config_dict[key] = val
            #ENDIF
        #ENDFOR
        return run_config_dict
    #ENDDEF

    def run(self, circuits, job_name=None, job_share_level=None,
            job_tags=None, experiment_id=None, validate_qobj=None,
            header=None, shots=None, memory=None, qubit_lo_freq=None,
            meas_lo_freq=None, schedule_los=None, meas_level=None,
            meas_return=None, memory_slots=None, memory_slot_size=None,
            rep_time=None, rep_delay=None, init_qubits=None, parameter_binds=None,
            **run_config):
        # create qobj
        sim_method = None
        if self.configuration().simulator:
            sim_method = getattr(self.configuration(), 'simulation_method', None)
        #ENDIF
        qobj_header = run_config.pop('qobj_header', None)
        header = header or qobj_header
        run_config_dict = self._get_run_config(
            qobj_header=header, shots=shots, memory=memory, qubit_lo_freq=qubit_lo_freq,
            meas_lo_freq=meas_lo_freq, schedule_los=schedule_los, meas_level=meas_level,
            meas_return=meas_return, memory_slots=memory_slots, memory_slot_size=memory_slot_size,
            rep_time=rep_time, rep_delay=rep_delay, init_qubits=init_qubits, **run_config)
        if parameter_binds:
            run_config_dict['parameter_binds'] = parameter_binds
        #ENDIF
        if sim_method and 'method' not in run_config_dict:
            run_config_dict['method'] = sim_method
        #ENDIF
        qobj = assemble(circuits, self, **run_config_dict)
        qobj_id = qobj.qobj_id
        
        # post to /job-queue
        out_dict = qobj.to_dict()
        out_payload = json.dumps(out_dict, ensure_ascii=False, cls=PulseEncoder)
        headers = {"Content-Type": "application/json"}
        res = requests.post(self.job_queue_url, headers=headers, data=out_payload)
        
        # get job_id
        res.raise_for_status()
        in_dict = res.json()
        job_id = in_dict["job_id"]
        if job_id == JOB_ID_ERROR:
            raise Exception("Backend errored in job_queue POST.")
        #ENDIF
        job = SLabJob(self, job_id, qobj_id)

        return job
    #ENDDEF
#ENDCLASS

