"""
backend.py

References:
[0] https://github.com/Qiskit/qiskit-terra/blob/2eee56616d50a9e26756f855ef4aa0135920ad78/qiskit/providers/models/backendconfiguration.py#L493
[1] https://github.com/Qiskit/qiskit-terra/issues/6469
[2] https://github.com/Qiskit/qiskit-terra/blob/2eee56616d50a9e26756f855ef4aa0135920ad78/qiskit/providers/models/backendconfiguration.py#L197
[3] https://github.com/Qiskit/qiskit-terra/blob/2eee56616d50a9e26756f855ef4aa0135920ad78/qiskit/providers/models/pulsedefaults.py#L164
[4] https://github.com/Qiskit/qiskit-terra/blob/2eee56616d50a9e26756f855ef4aa0135920ad78/qiskit/result/result.py#L29
[5] https://gist.github.com/nitaku/10d0662536f37a087e1b
"""

import argparse
import cgi
import collections
import http.server
import json
import logging
import os
import socket
import socketserver
import threading
import time

import yaml
from qiskit.providers.jobstatus import JobStatus
from qiskit.providers.models import PulseBackendConfiguration, PulseDefaults

from .json_util import PulseEncoder

SLAB_BACKEND_KEYS_REQUIRED = [
    "result_timeout",
    "result_timeout_sleep",
    "default_shots_per_set",
]

PULSE_BACKEND_CONFIGURATION_KEYS_REQUIRED = [
    "backend_name", "backend_version", "n_qubits",
    "basis_gates", "gates", "local", "simulator",
    "conditional", "open_pulse", "memory", "max_shots",
    "coupling_map", "n_uchannels", "u_channel_lo",
    "meas_levels", "qubit_lo_range", "meas_lo_range",
    "dt", "dtm", "rep_times", "meas_kernels", "discriminators",
]
PULSE_BACKEND_CONFIGURATION_KEYS = PULSE_BACKEND_CONFIGURATION_KEYS_REQUIRED + [
    "hamiltonian", "channel_bandwidth", "acquisition_latency",
    "conditional_latency", "meas_map", "max_experiments", "sample_name",
    "n_registers", "register_map", "configurable",
    "credits_required", "online_date", "display_name", "description",
    "tags", "channels", "parametric_pulses", "supported_instructions",
    "dynamic_reprate_enabled", "rep_delay_range", "default_rep_delay",
    "max_experiments", "processor_type"
]

PULSE_DEFAULTS_KEYS_REQUIRED = [
    "qubit_freq_est", "meas_freq_est", "buffer",
    "pulse_library", "cmd_def"
]
PULSE_DEFAULTS_KEYS = PULSE_DEFAULTS_KEYS_REQUIRED + [
    "meas_kernel", "discriminator"
]

JOB_ID_ERROR = "-1"

# TODO: Default Python on RFSoC is 3.6.5,
# if we ever upgrade to 3.7+ use ThreadingHTTPServer from http.server
# https://github.com/python/cpython/blob/3.9/Lib/http/server.py#L144
# https://pymotw.com/2/BaseHTTPServer/index.html#threading-and-forking
class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    
    def __init__(self, address, handler, backend):
        super().__init__(address, handler)
        self.backend = backend
    #ENDDEF
#ENDCLASS

class Handler(http.server.BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
    #ENDDEF

    def do_GET(self):
        caddr_str = "{}:{}".format(self.client_address[0], self.client_address[1])
        path = self.path
        tid = threading.get_ident()
        # log GET
        self.server.backend.logger.log(logging.INFO, "t{} received GET {} from {}"
                                       "".format(tid, path, caddr_str))
        # break on get path
        if path == "/pulse_backend_configuration.json" or path == "/pulse_defaults.json":
            if path == self.VALID_GET_PATHS[0]:
                out_payload = self.server.backend.backend_config_payload
                out_payload_name = "backend_config"
            else:
                out_payload = self.server.backend.defaults_payload
                out_payload_name = "defaults"
            #ENDIF
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out_payload)
            self.server.backend.logger.log(logging.INFO, "t{} returned {} payload"
                                           "".format(tid, out_payload_name))
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.server.backend.logger.log(logging.INFO, "t{} could not find resource"
                                           "".format(tid))
        #ENDIF
    #ENDDEF

    def do_POST(self):
        """
        References:
        see [5]
        """
        caddr_str = "{}:{}".format(self.client_address[0], self.client_address[1])
        path = self.path
        tid = threading.get_ident()
        ctype, pdict = cgi.parse_header(self.headers.getheader('content-type'))
        # log POST
        self.server.backend.logger.log(logging.INFO, "t{} received POST {} content-type {} from {}"
                                       "".format(tid, path, ctype, caddr_str))
        # break on post path
        if path == "/job-queue" or path == "/job-retrieve":
            if ctype != "application/json":
                self.server.backend.logger.log(logging.INFO, "t{} received incorrect content-type "
                                               "{} for /job-queue or /job-retrieve"
                                               "".format(tid, ctype))
            else:
                length = int(self.headers.getheader('content-length'))
                in_payload = self.rfile.read(length)
                if path == "/job-queue":
                    out_payload = self.server.backend.queue_(in_payload)
                    out_payload_name = "queue"
                elif path == "/job-retrieve":
                    out_payload = self.server.backend.retrieve(in_payload)
                    out_payload_name = "retrieve"
                #ENDIF
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.wfile.write(out_payload)
                self.end_headers()
                self.server.backend.logger.log(logging.INFO, "t{} returned {} payload"
                                               "".format(tid, out_payload_name))
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.server.backend.logger.log(logging.INFO, "t{} could not find resource"
                                           "".format(tid))
        #ENDIF        
    #ENDDEF

    def log_request(self, code="-", size="-"):
        pass
    #ENDDEF

    def log_message(self):
        pass
    #ENDDEF
#ENDCLASS

class SLabBackend(object):
    """
    This class handles most of the logic for running a backend.
    """
    
    DEFAULT_SERVER_PORT = 8555

    def __init__(self, config_file_path, log_path, experiment_class,
                 result_timeout, result_timeout_sleep,
                 default_shots_per_set, **kwargs):
        """
        Initialize the backend. The backend will not be functional until the `serve`
        method is called.

        args:
        config_file_path :: str - path to the config file, can be JSON or YAML
        log_path :: str - path to the directory where log files are stored
        experiment_class :: PulseExperiment - the parser / executor for this type of backend
        result_timeout :: int - the duration in seconds after which the
            results of a job will be flushed from the server, starting
            from the time of the job's completion. the job will no
            longer be retrievable. this policy prevents results
            from abandoned jobs taking up memory
        result_timeout_sleep :: int - this is a duration in seconds that determines
            how often `_worker2` wakes up to clean out experiment results
        default_shots_per_set :: int - a job has multiple experiments.
            each experiment should be executed
            `shots` number of times. this backend executes the experiments
            in round-robin fashion `sets` number of times, where each experiment
            is executed `shots_per_set` number of times sequentially in the
            set. `default_shots_per_set` is the default value of `shots_per_set`
            if none is given
            "sets" of round-robin `default_shots_per_set`

        kwargs:
        backend_name - see [0]
        backend_version - see [0]
        n_qubits - see [0]
        basis_gates - see [0]
        gates - see [0]
        local - see [0]
        simulator - see [0]
        conditional - see [0]
        open_pulse - see [0]
        memory - see [0]
        max_shots - see [0]
        coupling_map - see [0]
        n_uchannels - see [0]
        u_channel_lo - see [0]
        meas_levels - see [0]
        qubit_lo_range - see [0]
        meas_lo_range - see [0]
        dt - see [0]
        dtm - see [0]
        rep_times - see [0]
        meas_kernels - see [0]
        discriminators - see [0]
        hamiltonian - see [0]
        channel_bandwidth - see [0]
        acquisition_latency - see [0]
        conditional_latency - see [0]
        meas_map - see [0]
        max_experiments - see [0]
        sample_name - see [0]
        n_registers - see [0]
        register_map - see [0]
        configurable - see [0]
        credits_required - see [0]
        online_date - see [0]
        display_name - see [0]
        description - see [0]
        tags - see [0]
        channels - see [0]
        parameteric_pulses - see [0], [1]
        supported_instructions - see [2]
        dynamic_reprate_enabled - see [2]
        rep_delay_range - see [2]
        default_rep_delay - see [2]
        max_experiments - see [2]
        processor_type - see [2]
        qubit_freq_est - see [3]
        meas_freq_est - see [3]
        buffer - see [3]
        pulse_library - see [3]
        cmd_def - see [3]
        meas_kernel - see [3]
        discriminator - see [3]
        """
        # initialize fields
        super().__init__()
        self.experiment_class = experiment_class
        self.result_timeout = result_timeout
        self.result_timeout_sleep = result_timeout_sleep
        self.default_shots_per_set = default_shots_per_set
        self.backend_name = kwargs["backend_name"]
        self.backend_version = kwargs["backend_version"]
        self.queue_lock = threading.Lock()
        self.queue_job_ids = collections.deque()
        self.queue_qobj_dicts = dict()
        self.results_lock = threading.Lock()
        self.results = dict()
        self.job_id_lock = threading.Lock()
        self.job_id_counter = 0
        # see [4]
        result_failure_dict = {
            "backend_name": self.backend_name,
            "backend_version": self.backend_version,
            "success": False,
            "results": [],
            "date": None,
            "status": JobStatus.ERROR,
            "header": None,
        }
        self.result_failure_payload = bytes(json.dumps(result_failure_dict, cls=PulseEncoder,
                                                       ensure_ascii=False), "utf-8")
        
        # prep backend_config and defaults payloads
        backend_config_dict = dict()
        for key in kwargs.keys():
            if key in PULSE_BACKEND_CONFIGURATION_KEYS:
                backend_config_dict[key] = kwargs[key]
            #ENDIF
        #ENDFOR
        # check that input is well-formed
        _ = PulseBackendConfiguration.from_dict(backend_config_dict)
        self.backend_config_payload = bytes(json.dumps(backend_config_dict, cls=PulseEncoder,
                                                       ensure_ascii=False), "utf-8")
        defaults_dict = dict()
        for key in kwargs.keys():
            if key in PULSE_DEFAULTS_KEYS:
                defaults_dict[key] = kwargs[key]
            #ENDIF
        #ENDFOR
        # check that input is well-formed
        print(defaults_dict)
        _ = PulseDefaults.from_dict(defaults_dict)
        self.defaults_payload = bytes(json.dumps(defaults_dict, cls=PulseEncoder,
                                                 ensure_ascii=False), "utf-8")

        # create a log file
        lt = time.localtime()
        lt_str = "{}-{}-{}-{}:{}:{}".format(lt.tm_year, lt.tm_mon, lt.tm_mday,
                                            lt.tm_hour, lt.tm_min, lt.tm_sec)
        log_file = "{}.log".format(lt_str)
        log_file_path = os.path.abspath(os.path.join(log_path, log_file))
        logging.basicConfig(filename=log_file_path, level=logging.DEBUG)
        self.logger = logging.getLogger()
        # create latest.log, which links to log_file_path
        alias_log_file_path = os.path.abspath(os.path.join(log_path, "latest.log"))
        os.symlink(log_file_path, alias_log_file_path)

        # determine local ip
        # https://stackoverflow.com/questions/166506/
        # finding-local-ip-addresses-using-pythons-stdlib
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        self.localhost = s.getsockname()[0]
        s.close()

        # create http server
        # listen at all available addresses on designated port
        address = ("0.0.0.0", self.DEFAULT_SERVER_PORT)
        self.server = Server(address, Handler, self)

        # initial logging
        self.logger.log(logging.INFO, "started backend {}v{} at {}"
                        "".format(self.backend_name, self.backend_version, lt_str))
        self.logger.log(logging.INFO, "configuration file path: {}".format(config_file_path))
        self.logger.log(logging.INFO, "log file path: {}".format(log_file_path))
    #ENDDEF

    @classmethod
    def from_file(cls, config_file_path, log_path, experiment_class):
        # read in config file
        config_file = open(config_file_path, "r")
        config_suffix = config_file_path.split(".")[1]
        if config_suffix == "yaml":
            config_dict = yaml.load(config_file)
        elif config_suffix == "json":
            config_dict = json.load(config_file)
        else:
            raise Exception("Unrecognized config file type {}".format(config_suffix))
        #ENDIF
        config_file.close()

        # ensure config file has required keys
        for key in cls.required_config_keys():
            if not key in config_dict:
                raise Exception("Expected top-level key \"{}\" in config file. "
                                "See the file where this class is defined for "
                                "documentation on the required keys."
                                "".format(key))
            #ENDIF
        #ENDFOR

        return cls(config_file_path, log_path, experiment_class, **config_dict)
    #ENDDEF

    @classmethod
    def required_config_keys(cls):
        return (PULSE_BACKEND_CONFIGURATION_KEYS_REQUIRED
                + PULSE_DEFAULTS_KEYS_REQUIRED
                + SLAB_BACKEND_KEYS_REQUIRED)
    #ENDDEF

    def serve(self):
        """
        Enter service mode by spawning worker threads starting the HTTP server.
        """
        # spawn worker threads
        threading.Thread(target=self._worker1, daemon=True).start()
        threading.Thread(target=self._worker2, daemon=True).start()
        
        # enter service loop
        self.logger.log(logging.INFO, "starting HTTP server on {}:{}"
                        "".format(self.localhost, self.DEFAULT_SERVER_PORT))
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            self.logger.log(logging.INFO, "^C received, shutting down HTTP server")
            self.server.socket.close()
        #ENDTRY
    #ENDDEF

    def _acquire_lock(self, lock, name, tid):
        self.logger.log(logging.DEBUG, "t{} acquiring {}".format(tid, name))
        lock.acquire()
        self.logger.log(logging.DEBUG, "t{} acquired {}".format(tid, name))
    #ENDDEF

    def _release_lock(self, lock, name, tid):
        lock.release()
        self.logger.log(logging.DEBUG, "t{} released {}".format(tid, name))
    #ENDDEF

    def queue_(self, qobj_payload):
        tid = threading.get_ident()

        # validate qobj_payload
        # validation of qobj_dict is deferred to execution
        try:
            qobj_dict = json.loads(qobj_payload)
        except Error as e:
            qobj_dict = None
            self.logger.log(logging.INFO, "t{} encountered invalid JSON in job_queue.\n{}"
                            "".format(tid, e))
        #ENDTRY

        # queue the job if qobj_payload was valid
        if qobj_dict is not None:
            # create unique job id
            self._acquire_lock(self.job_id_lock, "job_id_lock", tid)
            job_id = str(self.job_id_counter)
            self.job_id_counter += 1
            self._release_lock(self.job_id_lock, "job_id_lock", tid)

            # append job to queue
            self._acquire_lock(self.queue_lock, "queue_lock", tid)
            self.queue_job_ids.append(job_id)
            self.queue_qobj_dicts[job_id] = qobj_dict
            self._release_lock(self.queue_lock, "queue_lock", tid)

            # add job to results
            self._acquire_lock(self.results_lock, "results_lock", tid)
            # see [4]
            self.results[job_id] = {
                "results": [],
                "status": JobStatus.QUEUED,
                "success": False,
                "time": time.time(),
            }
            self._release_lock(self.results_lock, "results_lock", tid)
        # else, send failure response
        else:
            job_id = JOB_ID_ERROR
        #ENDIF

        # format response
        out_dict = {
            "job_id": job_id,
        }
        out_payload = bytes(json.dumps(out_dict, ensure_ascii=False), "utf-8")

        return out_payload
    #ENDDEF

    def retrieve(self, job_id_payload):
        tid = threading.get_ident()

        # validate job_id_payload
        try:
            job_id_dict = json.loads(job_id_payload)
        except Exception as e:
            job_id_dict = None
            self.logger.log(logging.INFO, "t{} encountered invalid JSON in job_retrieve.\n{}"
                            "".format(tid, e))
        #ENDTRY
        
        # validate job_id_dict
        if job_id_dict is not None:
            if not "job_id" in job_id_dict:
                self.logger.log(logging.INFO, "t{} expected key \"job_id\" in job_retrieve POST.")
                job_id = None
            else:
                job_id = job_id_dict["job_id"]
                if not isinstance(job_id, str):
                    self.logger.log(logging.INFO, "t{} expected value of \"job_id\" to be str "
                                    "got type {}".format(tid, type(job_id)))
                    job_id = None
                #ENDIF
            #ENDIF
        else:
            job_id = None
        #ENDIF
        
        # retrieve result in results for job_id if job_id_dict was valid
        if job_id is not None:
            self._acquire_lock(self.results_lock, "results_lock", tid)
            out_dict = copy(self.results.get(job_id, None))
            self._release_lock(self.results_lock, "results_lock", tid)
            if out_dict is not None:
                del out_dict["time"]
            #ENDIF
            out_payload = bytes(json.dumps(out_dict, ensure_ascii=False), "utf-8")
        # else, send failure response
        else:
            out_payload = self.result_failure_payload
        #ENDIF

        return out_payload
    #ENDDEF

    def _worker1(self):
        """
        Take jobs from the queue, execute them, and update their results.
        """
        tid = threading.get_ident()
        self.logger.log(logging.INFO, "t{} worker1 thread spawned".format(tid))

        # work forever
        while True:
            # attempt to take a new job from the queue
            self._acquire_lock(self.queue_lock, "queue_lock", tid)
            if len(self.queue) == 0:
                job_id = None
                qobj_dict = None
            else:
                job_id = self.queue_job_ids.popleft()
                qobj_dict = self.queue_qobj_dicts.pop(job_id)
            #ENDIF
            self._release_lock(self.queue_lock, "queue_lock", tid)
            
            # sleep if a job was not present
            if job_id is None:
                time.sleep(1)
            # work if a job was present
            else:
                # update job result
                # TODO could make a more fine-grained locking structure
                # on a per-result basis for updates like this
                self._acquire_lock(self.results_lock, "results_lock", tid)
                self.results[job_id]["status"] = JobStatus.RUNNING
                self._release_lock(self.results_lock, "results_lock", tid)

                # attempt to execute the job
                self.logger.log(logging.INFO, "t{} executing j{}".format(tid, job_id))
                try:
                    # parse experiments
                    experiments = list()
                    for expt_dict in qobj_dict["experiments"]:
                        experiment = self.experiment_class(qobj_dict, expt_dict, self)
                        experiments.append(experiment)
                    #ENDFOR

                    # run experiments in round-robin fashion until they
                    # have all been exhausted
                    results = [None] * len(experiments)
                    all_exhausted = False
                    while not all_exhausted:
                        # set all_exhausted to True, if any single experiment is exhausted,
                        # `all_exhausted` will be set to False
                        all_exhausted = True
                        for (j, experiment) in enumerate(experiments):
                            all_exhausted = experiment.exhausted and all_exhausted
                            # if exhausted, skip
                            if experiment.exhausted:
                                continue
                            # if not exhausted, execute a set
                            else:
                                results[j] = experiment.run_next_set(results[j])
                            #ENDIF
                        #ENDFOR
                        # update results
                        self._acquire_lock(self.results_lock, "results_lock", tid)
                        for j in range(len(results)):
                            self.results[job_id]["results"][j] = results[j]
                        #ENDIF
                        self._release_lock(self.results_lock, "results_lock", tid)
                    #ENDFOR
                    
                    # job done
                    self._acquire_lock(self.results_lock, "results_lock", tid)
                    self.results[job_id]["status"] = JobStatus.DONE
                    self.results[job_id]["success"] = True
                    self.results[job_id]["time"] = time.time()
                    self._release_lock(self.results_lock, "results_lock", tid)
                    self.logger.log(logging.INFO, "t{} executed j{}".format(tid, job_id))
                except Exception as e:
                    # job error
                    self._acquire_lock(self.results_lock, "results_lock", tid)
                    self.results[job_id]["status"] = JobStatus.ERROR
                    self.results[job_id]["time"] = time.time()
                    self._release_lock(self.results_lock, "results_lock", tid)
                    self.logger.log(logging.INFO, "t{} encountered Exception in job j{} "
                                    "execution.\n{}".format(tid, job_id, e))
                #ENDTRY
            #ENDIF
        #ENDWHILE
    #ENDDEF

    def _worker2(self):
        """
        Delete old results.
        """
        tid = threading.get_ident()
        self.logger.log(logging.INFO, "t{} worker2 thread spawned".format(tid))
        # work forever
        while True:
            # sleep for a while
            time.sleep(self.result_timeout_sleep)
            
            # delete results that have not been claimed within self.result_timeout
            deleted_job_ids = list()
            self._acquire_lock(self.results_lock, "results_lock", tid)
            for job_id in self.results.keys():
                elapsed_time = time.time() - self.results[job_id]["time"]
                if (elapsed_time > self.result_timeout
                    and self.results[job_id]["status"] in JOB_FINAL_STATES):
                    del self.results[job_id]
                    deleted_job_ids.append(job_id)
                #ENDIF
            #ENDFOR
            self._release_lock(self.results_lock, "results_lock", tid)
            for job_id in deleted_job_ids:
                self.logger.log(logging.INFO, "t{} deleted result for j{}"
                                "".format(tid, job_id))
            #ENDFOR
        #ENDWHILE
    #ENDDEF
#ENDCLASS
