import time
DATA_DIRECTORY = r'C:\_Data'

def dataserver_client(serveraddr='127.0.0.1', serverport=55556, localaddr='127.0.0.1'):
    import objectsharer as objsh
    if objsh.helper.backend is None:
        zbe = objsh.ZMQBackend()
        zbe.start_server(addr=localaddr)
        zbe.connect_to('tcp://%s:%d' % (serveraddr, serverport))  # Data server
    return objsh.helper.find_object('dataserver')

def get_file(filename, groupname="", timestamp_group=False, **kwargs):
    if not (filename.endswith('.h5') or filename.endswith('.hdf5')):
        filename += '.h5'
    c = dataserver_client()
    f = c.get_file(filename)
    if groupname:
        groupname = ":" + groupname
    if timestamp_group:
        date = time.strftime('%y-%m-%d')
        if date in f:
            f = f[date]
        else:
            f = f.create_group(date)
        f = f.create_group(time.strftime('%H:%M:%S' + groupname))
    return f

def run_dataserver(qt=False):
    from dataserver import start
    import os
    os.chdir(DATA_DIRECTORY)
    start(qt)


def resolve_file(filename, path):
    file = get_file(filename)
    for group in path.split('/'):
        file = file.get_group(group)
    return file.get_numbered_child()