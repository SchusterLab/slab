"""
backend.py
"""

import argparse
import cgi
import http.server
import json
import logging
import os
import socket
import socketserver
import threading
import time

import yaml

# https://github.com/Qiskit/qiskit-terra/blob/main/
# qiskit/providers/models/backendconfiguration.py#L493
PULSE_BACKEND_CONFIGURATION_KEYS = [
    "backend_name", "backend_version", "n_qubits",
    "basis_gates", "gates", "local", "simulator",
    "conditional", "open_pulse", "memory", "max_shots",
    "coupling_map", "n_uchannels", "u_channel_lo",
    "meas_levels", "qubit_lo_range", "meas_lo_range",
    "dt", "dtm", "rep_times", "meas_kernels", "discriminators",
    "hamiltonian", "channel_bandwidth", "acquisition_latency",
    "conditional_latency", "meas_map", "max_experiments", "sample_name",
    "n_registers", "register_map", "configurable",
    "credits_required", "online_date", "display_name", "description",
    "tags", "channels",
    # https://github.com/Qiskit/qiskit-terra/issues/6469
    "parametric_pulses"
]

# https://github.com/Qiskit/qiskit-terra/blob/main/
# qiskit/providers/models/pulsedefaults.py#L164
PULSE_DEFAULTS_KEYS = [
    "qubit_freq_est", "meas_freq_est", "buffer",
    "pulse_library", "cmd_def", "meas_kernel",
    "discriminator",
]

SLAB_BACKEND_KEYS = [
    "result_timeout"
]

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
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            if path == self.VALID_GET_PATHS[0]:
                out_payload = self.server.backend.backend_config_payload
                out_payload_name = "backend_config_payload"
            else:
                out_payload = self.server.backend.defaults_payload
                out_payload_name = "defaults_payload"
            #ENDIF
            self.wfile.write(out_payload)
            self.server.backend.logger.log(logging.INFO, "t{} returned {}"
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
        [0] https://gist.github.com/nitaku/10d0662536f37a087e1b
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
                    out_payload = self.server.backend.queue(in_payload)
                    out_payload_name = "job-queue"
                else:
                    out_payload = self.server.backend.retrieve(in_payload)
                    out_payload_name = "job-retrieve"
                #ENDIF
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(out_payload)
                self.server.backend.logger.log(logging.INFO, "t{} returned {} payload"
                                               "".format(tid, out_payload_name))
            #ENDIF
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
    DEFAULT_SERVER_PORT = 8555

    def __init__(self, config_file_path, log_path, experiment_class):
        """
        Initialize the backend. The backend will not be functional until the `serve`
        method is called.

        Args:
        config_file_path :: str - path to the config file, can be JSON or YAML
        log_path :: str - path to the directory where log files are stored
        """
        # initialize fields
        super().__init__()
        self.queue_lock = threading.Lock()
        self.queue = list()
        self.results_lock = threading.Lock()
        self.results = dict()
        self.jid_lock = threading.Lock()
        self.jid_counter = 0
        self.experiment_class = experiment_class
        
        # read in config file
        config_file = open(config_file_path, "r")
        config_suffix = config_file_path.split(".")[1]
        if config_suffix == "yaml":
            config_dict = yaml.load(config_file)
        elif config_suffix == "json":
            config_dict = json.load(config_file)
        else:
            raise Error("Unrecognized config file type {}".format(config_suffix))
        #ENDIF
        config_file.close()
        self.config_dict = config_dict
        
        # prep backend_config and defaults json
        backend_config_dict = dict()
        for key in PULSE_BACKEND_CONFIGURATION_KEYS:
            if key in self.config_dict:
                backend_config_dict[key] = self.config_dict[key]
            else:
                raise Error("Expected top-level key {} in configuration file".format(key))
            #ENDIF
        #ENDFOR
        self.backend_config_payload = bytes(json.dumps(backend_config_dict, ensure_ascii=False),
                                            "utf-8")
        defaults_dict = dict()
        for key in PULSE_DEFAULTS_KEYS:
            if key in self.config_dict:
                defaults_dict[key] = self.config_dict[key]
            else:
                raise Error("Expected top-level key {} in configuration file".format(key))
            #ENDIF
        #ENDFOR
        self.defaults_payload = bytes(json.dumps(defaults_dict, ensure_ascii=False),
                                      "utf-8")
        # ensure config file has keys required by self.experiment_class
        for key in self.experiment_class.required_backend_config_keys():
            if not key in self.config_dict:
                raise Error("Expected top-level key {} in configuration file".format(key))
            #ENDIF
        #ENDFOR

        # grab fields from config dict
        self.backend_name = self.config_dict["backend_name"]
        self.backend_version = self.config_dict["backend_version"]
        self.result_timeout = self.config_dict["result_timeout"]

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

    def required_config_keys():
        return PULSE_BACKEND_CONFIGURATION_KEYS + PULSE_DEFAULTS_KEYS + SLAB_BACKEND_KEYS
    #ENDDEF

    def serve(self):
        """
        Enter service mode by spawning a worker thread and starting the HTTP server.
        """
        # spawn worker thread
        threading.Thread(target=self._worker, daemon=True).start()
        
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

    def queue(self, qobj_payload):
        tid = threading.get_ident()

        # create unique job id
        self._acquire_lock(self.jid_lock, "jid_lock", tid)
        jid = self.jid_counter
        self.jid_counter += 1
        self._release_lock(self.jid_lock, "jid_lock", tid)
        
        # append job to queue
        self._acquire_lock(self.queue_lock, "queue_lock", tid)
        self.queue.append({"jid": jid, "qobj_payload": qobj_payload})
        position = len(self.queue)
        self._release_lock(self.queue_lock, "queue_lock", tid)

        # TODO format response
        out_dict = {}
        out_payload = bytes(json.dumps(out_dict, ensure_ascii=False), "utf-8")
        return out_payload
    #ENDDEF

    def retrieve(self, jid_payload):
        tid = threading.get_ident()

        # get jid from payload
        jid_dict = json.loads(jid_payload)
        # TODO there might be a qiskit standard for this
        jid = jid_dict.get("jid", None)

        # check for jid in results
        if jid is not None:
            self._acquire_lock(self.results_lock, "results_lock", tid)
            results_dict = self.results.get(jid, None)
            self._release_lock(self.results_lock, "results_lock", tid)
        else:
            results_dict = None
        #ENDIF

        # TODO format response
        result = results_dict["result"]
        out_dict = {}
        out_payload = bytes(json.dumps(out_dict, ensure_ascii=False), "utf-8")
        return out_payload
    #ENDDEF

    def _worker(self):
        tid = threading.get_ident()
        self.logger.log(logging.INFO, "t{} worker thread spawned".format(tid))

        # work forever
        while True:
            # attempt to take a new job from the queue
            self._acquire_lock(self.queue_lock, "queue_lock", tid)
            if len(self.queue) == 0:
                queue_dict = None
            else:
                queue_dict = self.queue.pop(0)
            #ENDIF
            self._release_lock(self.queue_lock, "queue_lock", tid)

            # sleep if a job was not present
            if queue_dict is None:
                time.sleep(1)
            # work if a job was present
            else:
                # attempt to deserialize the qobj_payload
                qobj_payload = queue_dict["qobj_payload"]
                jid = queue_dict["jid"]
                try:
                    # TODO: custom JSON encoder
                    qobj_dict = json.loads(qobj_payload)
                except Exception as e:
                    qobj_dict = None
                #ENDTRY

                # TODO if the qobj_dict was malformed, return a failure result
                if qobj_dict is None:
                    self.logger.log(logging.INFO, "t{} received malformed j{}".format(tid, jid))
                # otherwise, run the experiments
                else:
                    self.logger.log(logging.INFO, "t{} executing j{}".format(tid, jid))
                    # parse experiments
                    shots_per_set = qobj_dict["config"]["shots_per_set"]
                    shots = qobj_dict["config"]["shots"]
                    sets = int(np.ceil(shots / shots_per_set))
                    experiments = list()
                    for expt_dict in qobj_dict["experiments"]:
                        experiment = self.experiment_class(qobj_dict, expt_dict, self)
                        experiments.append(experiment)
                    #ENDFOR

                    # run experiments in round-robin fashion until they
                    # have all been exhausted
                    for set_idx in range(self.sets):
                        # execute a set
                        result = list()
                        for experiment in experiments:
                            result_ = experiment.run_set(set_idx)
                            result.append(result_)
                        #ENDFOR
                        # make set available for fetching
                        if set_idx == (self.sets - 1):
                            shots_completed = self.shots
                        else:
                            shots_completed = set_idx * self.shots_per_set
                        #ENDIF
                        self._acquire_lock(self.results_lock, "results_lock", tid)
                        # if there is not a previous result waiting to be fetched,
                        # create a new result entry
                        if not jid in self.results:
                            results[jid] = {
                                "result": result,
                                "shots_completed": shots_completed,
                                "time": time.time()
                            }
                        # if there is a previous result waiting to be fetched,
                        # concatenate it with the new result
                        else:
                            for j in range(len(result)):
                                # TODO concatenate appropriately here
                                result[j] += results[jid]["result"][j]
                            #ENDFOR
                            results[jid]["result"] = result
                            results[jid]["shots_completed"] = shots_completed
                            results[jid]["time"] = time.time()
                        #ENDIF
                        self._release_lock(self.results_lock, "results_lock", tid)
                    #ENDWHILE
                #ENDIF
                self.logger.log(logging.INFO, "t{} completed j{}".format(tid, jid))
            #ENDIF

            # delete results that have not been claimed within self.result_timeout
            # TODO a separate worker thread could do this constantly
            deleted_jids = list()
            self._acquire_lock(self.results_lock, "results_lock", tid)
            for jid in self.results.keys():
                elapsed_time = time.time() - self.results[jid]["time"]
                if elapsed_time > self.result_timeout:
                    del self.results[jid]
                    deleted_jids.append(jid)
                #ENDIF
            #ENDFOR
            self._release_lock(self.results_lock, "results_lock", tid)
            for jid in deleted_jids:
                self.logger.log(logging.INFO, "t{} deleted result for j{}"
                                "".format(tid, jid))
            #ENDFOR
        #ENDWHILE
    #ENDDEF
#ENDCLASS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config-file-path", type=str)
    parser.add_argument("--log-path", default=".", type=str)
    args = vars(parser.parse_args())
    backend = SLabBackend(args["config-file-path"], args["log_path"])
    backend.serve()
#ENDDEF

if __name__ == "__main__":
    main()
#ENDIF
