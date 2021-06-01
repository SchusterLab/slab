"""
server.py
"""

import argparse

from slab.instruments.qiskit.rfsoc import RFSoCBackend

def main():
    # parse CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument("config-file-path", type=str)
    args = vars(parser.parse_args())
    config_file_path = args["config-file-path"]

    # start backend and serve
    backend = RFSoCBackend.from_file(config_file_path)
    backend.serve()
#ENDDEF

if __name__ == "__main__":
    main()
#ENDIF
