"""
job.py
"""

import json
import requests
import time

from qiskit.providers.job import JobV1 as Job
from qiskit.providers.jobstatus import JobStatus, JOB_FINAL_STATES
from qiskit.result import Result

from .json_util import PulseEncoder

class SLabJob(Job):
    def __init__(self, backend, job_id, qobj_id, **kwargs):
        super().__init__(backend, job_id, **kwargs)
        bc = self._backend.configuration()
        self._backend_name = bc.backend_name
        self._backend_version = bc.backend_version
        self._status = JobStatus.INITIALIZING
        self._qobj_id = qobj_id
        self._results = list()
    #ENDDEF
    
    def submit(self):
        raise NotImplementedError("job.submit() is not supported. Please use "
                                  "SLabBackendInterface.run() to submit a job.")
    #ENDDEF

    def result(self):
        # post to /job-retrieve
        out_dict = {
            "job_id": self._job_id
        }
        out_payload = json.dumps(out_dict, cls=PulseEncoder)
        headers = {"Content-Type": "application/json"}
        res = requests.post(self._backend.job_retrieve_url, headers=headers, data=out_payload)

        # get result
        res.raise_for_status()
        in_dict = res.json()
        result_ = Result.from_dict(in_dict)
        self._status = JobStatus(result_.status)
        
        return result_
    #ENDDEF

    def status(self):
        self.result()
        return self._status
    #ENDDEF
#ENDCLASS
