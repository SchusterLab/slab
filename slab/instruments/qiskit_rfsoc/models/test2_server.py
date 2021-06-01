"""
test2_server.py
"""

import argparse
from importlib import reload

from slab.instruments.qiskit_rfsoc.models import RFSoCBackend

def main(config_file_path=None, log_path=None):
    if (config_file_path is None and log_path is None):
        parser = argparse.ArgumentParser()
        parser.add_argument("config-file-path", type=str)
        parser.add_argument("--log-path", default=".", type=str)
        args = vars(parser.parse_args())
        config_file_path = args["config-file-path"]
        log_path = args["log_path"]
    #ENDIF
    backend = RFSoCBackend.from_file(config_file_path, log_path)
    backend.serve()
#ENDDEF

if __name__ == "__main__":
    main()
#ENDIF
