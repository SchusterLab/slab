import logging
#logging.getLogger().setLevel(logging.DEBUG)
import numpy as np
import time

import objectsharer as objsh
if objsh.helper.backend is None:
    zbe = objsh.ZMQBackend()
    zbe.start_server(addr='127.0.0.1')
zbe.connect_to('tcp://127.0.0.1:55556')     # Data server

def group_changed_cb(key, group=None):
    print 'Key %s changed in group %s' % (key, group, )

dataserver = objsh.helper.find_object('dataserver')

xs = np.linspace(-1, 1, 1001)
ys = np.linspace(-2, 2, 1001)
XS, YS = np.meshgrid(xs, ys)
RS = np.sqrt(XS**2 + YS**2)
ZS = np.sin(XS) * np.cos(YS) * np.sin(XS + YS)

start = time.time()
f = dataserver.get_file('test.hdf5')
g = f.create_group('data%s' % time.strftime('%Y%m%d_%H%M%S', time.localtime()))
g.set_attrs(instruments=('abc','def'))
g.create_dataset('xs', data=xs)
g.create_dataset('ys', data=ys)
g.connect('changed', lambda key: group_changed_cb(key, group=g))
g['xs2'] = xs**2
data = g.create_dataset('averaged', shape=(1001,1001), dtype=np.float)
data[:] = ZS
end = time.time()
print 'Sent in %.03f sec' % (end - start,)
data.set_attrs(done=True)

aslice = data[0,:]

zbe.add_qt_timer()

