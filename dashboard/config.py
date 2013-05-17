import Pyro4

manager_id = 'manager'
manager_host = 'localhost'
manager_port = 5555
manager_uri = "PYRO:%s@%s:%d" % (manager_id, manager_host, manager_port)
RUNNING = True

Pyro4.config.HMAC_KEY = '6551d449b0564585a9d39c0bd327dcf1'

h5file_directory = r'S:\_Data'
h5file_filter = 'HDF5 Files (*.h5)'
