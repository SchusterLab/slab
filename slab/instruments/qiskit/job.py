"""
job.py
"""

import requests
import time

from qiskit.providers.job import JobV1 as Job
from qiskit.providers.jobstatus import JobStatus, JOB_FINAL_STATES
from qiskit.result import Result

RETRIEVE_KEYS = [
    # "backend_name", "backend_version", "qobj_id", "job_id", "header", "date"
    "success", "results", "status",
]

class SLabJob(Job):
    def __init__(self, backend, job_id, qobj_id, **kwargs):
        super().__init__(backend, job_id, **kwargs)
        bc = self._backend.configuration()
        self._backend_name = bc.backend_name
        self._backend_version = bc.backend_version
        self._status = JobStatus.INITIALIZED
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
        out_payload = json.dumps(out_dict, cls=json.PulseEncoder)
        headers = {"Content-Type": "application/json"}
        res = requests.post(self._backend.job_retireve_url, headers=headers, data=out_payload)

        # validate response
        res.raise_for_status()
        if not ("Content-Type" in res.headers
                and res.headers["Content-Type"] == "application/json"):
            raise Exception("Expected Content-Type: application/json in "
                            "job_retrieve POST response.")
        #ENDIF
        # exception may be raised if json is invalid
        in_dict = json.loads(res.json())
        for key in RETRIEVE_KEYS:
            if not key in in_dict:
                raise Exception("Expected key \"{}\" in job_retrieve POST response."
                                "".format(key))
            #ENDIF
        #ENDFOR

        # construct result
        results = in_dict["results"]
        for i in range(len(results)):
            # TODO concatenate results appropriately
            self._results[i] += results[i]
        #ENDFOR
        self._status = JobStatus(in_dict["status"])
        success = in_dict["success"]
        result_ = Result.from_dict({
            "backend_name": self._backend_name,
            "backend_version": self._backend_version,
            "qobj_id": self._qobj_id,
            "job_id": self._job_id,
            "success": success,
            "results": self._results,
            "status": self._status
        })
        
        return result_
    #ENDDEF

    def status(self):
        self.result()
        return self._status
    #ENDDEF
#ENDCLASS
