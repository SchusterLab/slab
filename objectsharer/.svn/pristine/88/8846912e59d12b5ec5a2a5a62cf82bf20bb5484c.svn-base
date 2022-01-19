# ObjectSharer v2 with ZMQ communication backend
# Reinier Heeres <reinier@heeres.eu>, 2013
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

import logging
import random
import cPickle as pickle
import time
import numpy as np
import inspect
import uuid
import types
import bisect
import os
import traceback

logger = logging.getLogger("Object Sharer")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
handler.setLevel(logging.WARNING)
handler.setFormatter(formatter)
logger.addHandler(handler)

DEFAULT_TIMEOUT = 5000      # Timeout in msec
REDUCE_LATENCY  = True      # Use Voodoo latency reduction?

OS_CALL         = 'c'
OS_RETURN       = 'r'
OS_SIGNAL       = 'OS_SIG'

# List of special functions to wrap

SPECIAL_FUNCTIONS = (
    '__getitem__',
    '__setitem__',
    '__delitem__',
    '__contains__',
    '__str__',
    '__repr__',
)

def ellipsize(s):
    if len(s) > 64:
        return s[:64] + '...'
    else:
        return s

class RemoteException(Exception):
    pass

class TimeoutError(RuntimeError):
    pass

class OSSpecial(object):
    def __init__(self, **kwargs):
        for k, v in kwargs:
            setattr(self, k, v)

#########################################
# Decorators
#########################################

def cache_result(f):
    '''
    Decorator to instruct client to cache function result
    '''
    if not hasattr(f, '_share_options'):
        f._share_options = dict()
    f._share_options['cache_result'] = True
    return f

#########################################
# Asynchronous reply helpers
#########################################

class AsyncReply(object):
    '''
    Container to receive the reply from an asynchoronous call.
    Use is_valid() to see whether the value is valid.
    '''

    def __init__(self, callid, callback=None):
        self._callid = callid
        self.val_valid = False
        self.val = None
        self.callback = callback

    def set(self, val):
        self.val = val
        self.val_valid = True
        if self.callback is not None:
            logger.debug('Performing callback for call id %d' % self._callid)
            self.callback(val)

    def get(self, block=False, delay=DEFAULT_TIMEOUT, do_raise=True):
        if block and not self.is_valid():
            helper.interact(delay=delay)
        if do_raise and isinstance(self.val, Exception):
            raise self.val
        return self.val

    def is_valid(self):
        return self.val_valid

class AsyncHelloReply(object):
    def __init__(self, target):
        self.target = target
    def is_valid(self):
        self.val = helper.backend.get_uid_for_addr(self.target)
        return (self.val is not None)

#########################################
# Helper functions to replace parameters to make transmission possible
# or more efficient.
#########################################

def _walk_objects(obj, func, *args):
    if type(obj) in (types.ListType, types.TupleType):
        obj = list(obj)
        for i, v in enumerate(obj):
            obj[i] = _walk_objects(v, func, *args)
    if type(obj) is types.DictType:
        for k in sorted(obj):
            obj[k] = _walk_objects(obj[k], func, *args)
    return func(obj, *args)

def _wrap_ars_sobjs(obj, arlist=None):
    '''
    Function to wrap numpy arrays and shared objects for efficient transmission.
    '''
    def replace(o):
        if isinstance(o, np.ndarray):
            if not o.flags['C_CONTIGUOUS']:
                o = np.ascontiguousarray(o)
            assert o.flags['C_CONTIGUOUS']
            arlist.append(o)
            return dict(
                OS_AR=True,
                shape=o.shape,
                dtype=o.dtype,
            )
        elif hasattr(o, '_OS_UID'):
            ret = dict(OS_UID=o._OS_UID)
            if hasattr(o, '_OS_SRV_ID'):    # Not a local object
                ret['OS_SRV_ID'] = o._OS_SRV_ID
                ret['OS_SRV_ADDR'] = o._OS_SRV_ADDR
            return ret
#        elif isinstance(o, ObjectProxy):
#            raise ValueError('ObjectProxy without OS_UID')
        return o

    if arlist is None:
        arlist = []
    try:
        obj = _walk_objects(obj, replace)
        return obj, arlist
    except:
        return obj, arlist

def _unwrap_ars_sobjs(obj, bufs, client=None):
    '''
    Function to unwrap numpy arrays and shared objects after efficient
    transmission.
    '''

    def replace(o):
        if type(o) is types.DictType:
            if 'OS_AR' in o:
                if len(bufs) == 0:
                    raise ValueError('No buffer!')
                ar = np.frombuffer(
                    bufs.pop(0),
                    dtype=o['dtype']
                )
                ar = ar.reshape(o['shape'])
                return ar
            if 'OS_UID' in o:
                if 'OS_SRV_ID' in o and 'OS_SRV_ADDR' in o:
                    return helper.get_object_from(o['OS_UID'], o['OS_SRV_ID'], o['OS_SRV_ADDR'])
                else:
                    return helper.get_object_from(o['OS_UID'], client)
        return o

    obj = _walk_objects(obj, replace)
    return obj, bufs

#########################################
# Object sharer core
#########################################

class ObjectSharer(object):
    '''
    The object sharer core.
    '''

    def __init__(self):
        self.sock = None
        self.backend = None

        # Local objects
        self.objects = {}
        self.name_map = {}

        # Clients, proxies and remote object lists
        self.clients = {}
        self._proxy_cache = {}
        self._client_object_list_cache = {}

        self._last_call_id = 0
        self.reply_objects = {}

        # Signal related 
        self._last_hid = 0
        self._callbacks_hid = {}
        self._callbacks_name = {}
        self._signal_queue = []

    def set_backend(self, backend):
        self.backend = backend

    def interact(self, delay=DEFAULT_TIMEOUT, wait_for=None):
        self.backend.main_loop(delay=delay, wait_for=wait_for)

    def call(self, client, obj_name, func_name, *args, **kwargs):
        is_signal = kwargs.get(OS_SIGNAL, False)
        callback = kwargs.pop('callback', None)
        async = kwargs.pop('async', False) or (callback is not None) or is_signal
        timeout = kwargs.pop('timeout', DEFAULT_TIMEOUT)

        self._last_call_id += 1
        callid = self._last_call_id
        async_reply = AsyncReply(callid, callback=callback)
        self.reply_objects[callid] = async_reply
        logger.debug('Sending call %d to %s: %s.%s(%s,%s), async=%s', callid, client, obj_name, func_name, ellipsize(str(args)), ellipsize(str(kwargs)), async)

        args, arlist = _wrap_ars_sobjs(args)
        kwargs, arlist = _wrap_ars_sobjs(kwargs, arlist)
        msg = (
            OS_CALL,
            callid,
            obj_name,
            func_name,
            args,
            kwargs
        )
        try:
            msg = pickle.dumps(msg)
        except:
            raise Exception('Unable to pickle function call: %s' % str(msg))
        self.backend.send_to(client, msg, arlist)

        if async:
            return async_reply

        ret = self.backend.main_loop(delay=timeout, wait_for=async_reply)
        if ret:
            val = async_reply.get()
            return val
        else:
            raise TimeoutError('Call timed out')

    #####################################
    # Object resolving functions.
    #####################################

    def list_objects(self):
        '''
        Return a list of locally available objects, comprising of both the uids
        and the aliases.
        '''
        ret = list(self.objects.keys())
        ret.extend(self.name_map.keys())
        return ret

    def get_object(self, objname):
        '''
        Get a local object.
        <objname> can be a uid or an alias.
        '''
        # Look-up in name_map
        objname = self.name_map.get(objname, objname)
        # Look-up in objects list
        return self.objects.get(objname, None)

    def _get_object_shared_props_funcs(self, obj):
        props = []
        funcs = []
        for key, val in inspect.getmembers(obj):
            if key.startswith('_') and not key in SPECIAL_FUNCTIONS:
                continue
            if key == 'connect':
                continue
            elif callable(val):
                if hasattr(val, '_share_options'):
                    opts = val._share_options
                else:
                    opts = {}
                opts['__doc__'] = getattr(val, '__doc__', None)
                try:
                    opts['__argspec__'] = inspect.getargspec(val)
                except:
                    opts['__argspec__'] = None
                funcs.append((key, opts))
            else:
                props.append(key)

        return props, funcs

    def get_object_info(self, objname):
        '''
        Return the object info of a local object to build a proxy remotely.
        '''
        obj = self.get_object(objname)
        if obj is None:
            return None

        props, funcs = self._get_object_shared_props_funcs(obj)
        info = dict(
            uid=obj._OS_UID,
            properties=props,
            functions=funcs
        )
        return info

    def get_object_info_from(self, objname, client_id, client_addr=None):
        '''
        Get object info from a particular client.
        If not connected yet, do that first.
        '''

        # Handle clients that we are not yet connected to.
        if client_id not in self.clients:
            if client_addr is None:
                logger.warning('Object from unknown client requested')
                return None
            logger.info('Object %s requested from unconnected client %s @ %s, connecting...', objname, client_id, client_addr)
            self.backend.connect_to(client_addr, uid=client_id)

        # We should be connected now
        if client_id not in self.clients:
            logger.error('Unable to connect to client')
            return None

        try:
            return self.clients[client_id].get_object_info(objname)
        except TimeoutError:
            logger.warning('Client %s unresponsive: Removing from connected' % client_id)
            del self.clients[client_id] # I'm trying to think of a better place to do client invalidation!
            return None

    def get_object_from(self, objname, client_id, client_addr=None, no_cache=False):
        '''
        Get an object from a particular client.
        If a proxy is available in cache return that
        If not connected yet, do that first.
        '''

        # Return from cache if it is the right object (it could be an object
        # with the same alias at a different location),
        if not no_cache:
            if objname in self._proxy_cache and self._proxy_cache[objname]._OS_SRV_ID == client_id:
                return self._proxy_cache[objname]

        info = self.get_object_info_from(objname, client_id, client_addr=client_addr)
        if info is None:
            return None
        proxy = ObjectProxy(client_id, info)
        self._proxy_cache[objname] = proxy
        self._proxy_cache[proxy.os_get_uid()] = proxy
        return proxy

    def find_object(self, objname, client_id=None, client_addr=None, no_cache=False):
        '''
        Find a particular object either locally or with a client.
        '''

        if client_id is not None:
            return self.get_object_from(objname, client_id, client_addr)

        # A local object?
        obj = self.get_object(objname)
        if obj is not None:
            return obj
        if not no_cache:
            # A remote object with cached proxy?
            if objname in self._proxy_cache:
                return self._proxy_cache[objname]

            # See if we already know which client has this object
            for client_id, names in self._client_object_list_cache.iteritems():
                if objname in names:
                    return self.get_object_from(objname, client_id)

        # Query all clients
        # TODO: asynchronously
        for client_id in self.clients.keys():
            obj = self.get_object_from(objname, client_id, no_cache=no_cache)
            if obj is not None:
                return obj
        return None

    def register(self, obj, name=None):
        '''
        This function registers an object as a shared object.

        - Generates a unique id (<obj>.OS_UID)
        - Adds an emit function (<obj>.emit), which can be used to emit
        signals. A previously available emit function will still be called.
        '''

        if obj is None:
            return
        if hasattr(obj, '_OS_UID') and obj._OS_UID is not None:
            logger.warning('Object %s already registered' % obj._OS_UID)
            return

        obj._OS_UID = str(uuid.uuid4())
        if name is not None:
            if name in self.name_map:
                raise Exception('Object %s already defined' % name)
            self.name_map[name] = obj._OS_UID

        obj._OS_emit = getattr(obj, 'emit', None)
        # TODO: make obj properly assigned
        obj.emit = lambda signal, *args, **kwargs: self.emit_signal(obj._OS_UID, signal, *args, **kwargs)
        obj.connect = lambda signame, callback, *args, **kwargs: self.connect_signal(obj._OS_UID, signame, callback, *args, **kwargs)
        self.objects[obj._OS_UID] = obj

        root.emit('object-added', obj._OS_UID, name=name)

    def unregister(self, obj):
        if not hasattr(obj, '_OS_UID'):
            logger.warning('Trying to unregister an unknown object')

        if obj._OS_UID in self.objects:
            del self.objects[obj._OS_UID]
            root.emit('object-removed', obj._OS_UID)

        for name, id in self.name_map.items():
            if obj._OS_UID == id:
                del self.name_map[name]

    #####################################
    # Signal functions
    #####################################

    def connect_signal(self, uid, signame, callback, *args, **kwargs):
        '''
        Called by ObjectProxy instances to register a callback request.
        '''
        self._last_hid += 1
        info = {
                'hid': self._last_hid,
                'uid': uid,
                'signal': signame,
                'callback': callback,
                'args': args,
                'kwargs': kwargs,
        }

        self._callbacks_hid[self._last_hid] = info
        name = '%s__%s' % (uid, signame)
        if name in self._callbacks_name:
            self._callbacks_name[name].append(info)
        else:
            self._callbacks_name[name] = [info]

        return self._last_hid

    def disconnect_signal(self, hid):
        if hid in self._callbacks_hid:
            del self._callbacks_hid[hid]

        for name, info_list in self._callbacks_name.iteritems():
            for index, info in enumerate(info_list):
                if info['hid'] == hid:
                    del self._callbacks_name[name][index]
                    break

    def emit_signal(self, uid, signame, *args, **kwargs):
        logger.debug('Emitting %s(%r, %r) for %s to %d clients',
                signame, args, kwargs, uid, len(self.clients))

        kwargs[OS_SIGNAL] = True
        for client_id, client in self.clients.iteritems():
#            print 'Calling receive sig, uid=%s, signame %s, args %s, kwargs %s' % (uid, signame, args, kwargs)
            client.receive_signal(uid, signame, *args, **kwargs)
        self.receive_signal(uid, signame, *args, **kwargs)

    def receive_signal(self, uid, signame, *args, **kwargs):
        kwargs.pop(OS_SIGNAL, None)
        logger.debug('Received signal %s(%r, %r) from %s',
                signame, args, kwargs, uid)

        ncalls = 0
        start = time.time()
        name = '%s__%s' % (uid, signame)
        if name in self._callbacks_name:
            info_list = self._callbacks_name[name]
            for info in info_list:

                try:
                    fargs = list(args)
                    fargs.extend(info['args'])
                    fkwargs = kwargs.copy()
                    fkwargs.update(info['kwargs'])
                    info['callback'](*fargs, **fkwargs)
                except Exception, e:
                    logger.warning('Callback to %s failed for %s.%s: %s\n%s',
                            info.get('callback', None), uid, signame, str(e), traceback.format_exc())

        end = time.time()
        logger.debug('Did %d callbacks in %.03fms for sig %s',
                ncalls, (end - start) * 1000, signame)

    #####################################
    # Client management
    #####################################

    def _update_client_object_list(self, uid, names):
        if names is not None:
            self._client_object_list_cache[uid] = names

    def _add_client_to_list(self, uid, root_info):
        if root_info is None:
            raise Exception('Unable to retrieve root object from %s' % uid)
        logger.debug('  root@%s.get_object_info() reply: %s', uid, root_info)
        self.clients[uid] = ObjectProxy(uid, root_info)
        self.clients[uid].list_objects(callback=lambda reply, uid=uid:
            self._update_client_object_list(uid, reply))

    def request_client_proxy(self, uid, async=False):
        if not async:
            info = self.call(uid, 'root', 'get_object_info', 'root')
            self._add_client_to_list(uid, info)
        else:
            self.call(uid, 'root', 'get_object_info', 'root', callback=lambda reply, uid=uid:
                self._add_client_to_list(uid, reply))

    #####################################
    # Message processing
    #####################################

    def process_message(self, from_uid, info, bufs, waiting=False):
        '''
        Process a remote message.
        <from_uid> identifies the client that sent the message
        <info> is the message tuple
        <bufs> contains extra buffers used to unwrap numpy arrays

        If <waiting> is True it indicates a main loop is waiting for something,
        in which case signals get queued.
        '''

#        logger.debug('Msg from %s: %s', from_uid, info)

        if info[0] == OS_CALL:
            # In a try statement to postpone checks
            try:
                (callid, objid, funcname, args, kwargs) = info[1:6]
                sig = kwargs.get(OS_SIGNAL, False)

                # Store signals if waiting a reply or event
                if waiting and sig:
                    self._signal_queue.append((from_uid, info, bufs))
                    return

                # Unwrap arguments
                args, bufs = _unwrap_ars_sobjs(args, bufs, from_uid)
                kwargs, bufs = _unwrap_ars_sobjs(kwargs, bufs, from_uid)

                logger.debug('  Processing call %s: %s.%s(%s,%s)', callid, objid, funcname, args, kwargs)

                # Call function
                obj = self.get_object(objid)
                func = getattr(obj, funcname, None)
                ret = func(*args, **kwargs)

                # If a signal, no need to return anything to caller
                if sig:
                    return

                # Wrap return value
                ret, bufs = _wrap_ars_sobjs(ret)
                logger.debug('  Returning for call %s: %s', callid, lambda:ellipsize(str(ret)))

            # Handle errors
            except Exception, e:
                if len(info) < 6:
                    logger.error('Invalid call msg: %s', info)
                    ret = RemoteException('Invalid call msg')
                elif 'obj' not in locals() or obj is None:
                    ret = RemoteException('Object %s not available' % objid)
                elif 'func' not in locals() or func is None:
                    ret = RemoteException('Object %s does not have function %s' % (objid, funcname))
                else:
                    tb = traceback.format_exc(15)
                    ret = RemoteException('%s\n%s' % (e, tb))

            # Prepare return packet
            try:
                msg = pickle.dumps((OS_RETURN, callid, ret))
            except:
                ret = RemoteException('Unable to pickle return %s' % str(ret))
                msg = pickle.dumps((OS_RETURN, callid, ret))
                bufs = None
            self.backend.send_to(from_uid, msg, bufs)

        elif info[0] == OS_RETURN:
            if len(info) < 3:
                logger.debug('Invalid call msg')
                return Exception('Invalid return msg')

            # Get call id and unwrap return value
            callid, ret = info[1:3]
            ret, bufs = _unwrap_ars_sobjs(ret, bufs, from_uid)

#            logger.debug('  Processing return for %s', callid)
            if callid in self.reply_objects:
                self.reply_objects[callid].set(ret)
                # We should not keep track of the reply object
                del self.reply_objects[callid]
            else:
                raise Exception('Reply for unkown call %s', callid)

        elif info[0] == 'hello_from':
            logger.debug('Client %s connected from %s', from_uid, info[1])
            self.backend.connect_from(info[1], from_uid)
            if not self.backend.connected_to(from_uid):
                logger.debug('Initiating reverse connection...')
                self.backend.connect_to(info[1])
                self.request_client_proxy(from_uid, async=True)
            return

        if info[0] == 'goodbye_from':
            logger.debug('Goodbye client %s from %s', from_uid, info[1])
            forget_uid = self.backend.get_uid_for_addr(info[1])
            if forget_uid in self.clients:
                del self.clients[forget_uid]
                logger.debug('deleting client %s', forget_uid)
            self.backend.forget_connection(info[1], remote=False)
            if from_uid in self.clients:
                del self.clients[from_uid]
                logger.debug('deleting client %s', from_uid)
            return

        # Ping - pong to check alive
        if info[0] == 'ping':
            logger.debug('PING')
            msg = pickle.pickle(('pong',))
            self.backend.send_to(from_uid, msg)
        elif info[0] == 'pong':
            logger.debug('PONG')

        else:
            logger.debug('Unknown msg: %s', info)

    def flush_queue(self, nmax=5):
        '''
        Process a maximum on <nmax> queued signals.
        Return True if signal queue empty when returning.
        '''
        i = 0
        while i < nmax and len(self._signal_queue) > 0:
            from_uid, info, bufs = self._signal_queue.pop(0)
            self.process_message(from_uid, info, bufs)
            i += 1
        return (len(self._signal_queue) == 0)

class RootObject(object):
    '''
    Every program using shared objects should have an instance of RootObject.
    This object exposes functions of the ObjectSharer instance called helper.
    '''

    def __init__(self):
        pass

    def hello_world(self):
        return 'Hello world!'

    def hello_exception(self):
        return 1 / 0

    def client_announce(self, name):
        helper.add_client(name)

    def list_objects(self):
        return helper.list_objects()

    def get_object_info(self, objname):
        return helper.get_object_info(objname)

    def receive_signal(self, uid, signame, *args, **kwargs):
        helper.receive_signal(uid, signame, *args, **kwargs)

class _FunctionCall():

    def __init__(self, client, objname, funcname, share_options):
        self._client = client
        self._objname = objname
        self._funcname = funcname

        if share_options is None:
            self._share_options = {}
        else:
            self._share_options = share_options

        # Setup doc string, including remote function name and parameters
        doc = funcname
        argspec = self._share_options.get('__argspec__', None)
        if argspec:
            doc += inspect.formatargspec(*argspec) + '\n'
        else:
            doc += '()\n'

        docstr = self._share_options.get('__doc__', '')
        if docstr:
            doc += docstr
        setattr(self, '__doc__', doc)

        self._cached_result = None

    def __call__(self, *args, **kwargs):
        cache = self._share_options.get('cache_result', False)
        if cache and self._cached_result is not None:
            return self._cached_result

        ret = helper.call(self._client, self._objname, self._funcname, *args, **kwargs)
        if cache:
            self._cached_result = ret

        return ret

class ObjectProxy(object):
    '''
    Client side object proxy.

    Based on the info dictionary this object will be populated with functions
    and properties that are available on the remote object.
    '''

    PROXY_CACHE = {}
    def __new__TODO(cls, client, uid, info=None, newinst=False):
        if info is None:
            return None
        if info['uid'] in ObjectProxy.PROXY_CACHE:
            return ObjectProxy.PROXY_CACHE[info['uid']]
        else:
            return super(ObjectProxy, cls).__new__(client, uid, info, newinst)

    def __init__(self, client, info):
        self._OS_UID = info['uid']
        self._OS_SRV_ID = client
        self._OS_SRV_ADDR = helper.backend.get_addr_for_uid(client)
        self.__new_hid = 1
        self._specials = {}
        self.__initialize(info)

    def __getitem__(self, key):
        func = self._specials.get('__getitem__', None)
        if func is None:
            raise Exception('Object does not support indexing')
        return func(key)

    def __setitem__(self, key, val):
        func = self._specials.get('__setitem__', None)
        if func is None:
            raise Exception('Object does not support indexing')
        return func(key, val)

    def __delitem__(self, key): #TODO: make this less copy/paste
        func = self._specials.get('__delitem__', None)
        if func is None:
            raise Exception('Object does not support indexing')
        return func(key)

    def __contains__(self, key):
        func = self._specials.get('__contains__', None)
        if func is None:
            raise Exception('Object does not implement __contains__')
        return func(key)

    def __str__(self):
        s = 'ObjectProxy for %s @ %s (address %s)' % (self._OS_UID, self._OS_SRV_ID, self._OS_SRV_ADDR)
        func = self._specials.get('__str__', None)
        if func:
            s += '\nRemote info:\n%s' % (func(),)
        return s

    def __repr__(self):
        func = self._specials.get('__str__', None)
        s = 'ObjectProxy(%s)' % (self._OS_UID,)
        if func:
            s += ': %s' % (func(), )
        return s

    def __initialize(self, info):
        if info is None:
            return

        for funcname, share_options in info['functions']:
            func = _FunctionCall(self._OS_SRV_ID, self._OS_UID, funcname, share_options)
            if funcname in SPECIAL_FUNCTIONS:
                self._specials[funcname] = func
            else:
                setattr(self, funcname, func)

        for propname in info['properties']:
            setattr(self, propname, 'blaat')

    def connect(self, signame, func):
        return helper.connect_signal(self._OS_UID, signame, func)

    def disconnect(self, hid):
        return helper.disconnect_signal(hid)

    def os_get_client(self):
        return self._OS_SRV_ID

    def os_get_uid(self):
        return self._OS_UID

helper = ObjectSharer()
register = helper.register
find_object = helper.find_object

root = RootObject()
register(root, name='root')

# To integrate with Qt, run this loop in a separate thread.
# However, we might want to call process_message through a Qt.Queue something
# to make sure it is executed in the main thread. That way the objects can
# properly manipulate GUI elements.

import zmq

class ZMQBackend(object):

    def __init__(self):
        self.ctx = zmq.Context()
        self.srv = None
        self.uid = None
        self.addr = None
        self.port = None
        self.timer = None
        self.addr_to_sock_map = {}
        self.addr_to_uid_map = {}
        self.uid_to_sock_map = {}
        helper.set_backend(self)
        self._timeouts = {}
        self._scheduled_timeouts = []
        self._last_timeout_id = 0

    def get_addr(self):
        '''
        Return the ZMQ end point that this instance can be reached on.
        '''

        if self.addr == '*':
            return 'tcp://127.0.0.1:%d' % self.port
        else:
            return 'tcp://%s:%d' % (self.addr, self.port)

    def start_server(self, addr='*', port=None):
        '''
        Start ZMQ server listening on IP address <addr> and <port>.
        '''

        self.addr = addr
        self.port = port

        self.srv = self.ctx.socket(zmq.ROUTER)
        if port is None:
            self.port = self.srv.bind_to_random_port('tcp://%s'%addr, min_port=50000, max_port=60000, max_tries=100)
        else:
            self.srv.bind('tcp://%s:%d' % (addr, port))

        logger.debug('ObjectSharer listening at %s', self.get_addr())

    def connect_from(self, addr, uid):
        '''
        Should be called when a connection is made to associate
        <uid> with <addr>
        '''
        self.addr_to_uid_map[addr] = uid
        if addr in self.addr_to_sock_map:
            self.uid_to_sock_map[uid] = self.addr_to_sock_map[addr]

    def connected_to(self, uid):
        '''
        Return whether we are connected to client identified by <uid>.
        '''
        return uid in self.uid_to_sock_map

    def get_uid_for_addr(self, addr):
        return self.addr_to_uid_map.get(addr, None)

    def get_addr_for_uid(self, uid):
        for k, v in self.addr_to_uid_map.iteritems():
            if v == uid:
                return k
        return None

    def refresh_connection(self, addr):
        self.forget_connection(addr)
        time.sleep(.01)
        self.connect_to(addr)

    def forget_connection(self, addr, remote=True):
        logger.debug('Forgetting connection: %s', addr)
        msg = ('goodbye_from', 'tcp://%s:%d' % (self.addr, self.port))
        if addr not in self.addr_to_sock_map: # Open up socket so we can tell remote to forget it
            if remote:
                sock = self.ctx.socket(zmq.DEALER)
                sock.connect(addr)
                sock.send(pickle.dumps(msg))
                sock.close()
            else:
                return
        else:
            if remote:
                sock = self.addr_to_sock_map[addr]
                sock.send(pickle.dumps(msg))
                sock.close()

            del self.addr_to_sock_map[addr]

            if addr in self.addr_to_uid_map:
                uid = self.addr_to_uid_map.pop(addr)
                if uid in self.uid_to_sock_map:
                    del self.uid_to_sock_map[uid]

    def _generate_id(self):
        '''
        Generate a unique id.
        '''
        chars = 'abcdefghijklmnopqrstuvwxyz'
        chars += chars.upper()
        idstr = 'p%d_' % os.getpid()
        for i in range(6):
            idstr += chars[random.randint(0, len(chars)-1)]
        return idstr

    def connect_to(self, addr, delay=1000, async=False, uid=None):
        '''
        Connect to a remote ObjectSharer at <addr>.
        If <uid> is specified it is associated with the client at <addr>.
        If <async> is False (default), wait for a reply.
        '''
        logger.debug('Connecting to %s', addr)
        if addr in self.addr_to_sock_map:
            logger.warning('Already connected to %s' % addr)
            return
        if uid is not None:
            if uid in self.addr_to_uid_map.values():
                logger.warning('Client %s already present at different address')
                return
            self.addr_to_uid_map[addr] = uid

        sock = self.ctx.socket(zmq.DEALER)
        sock.setsockopt(zmq.IDENTITY, self._generate_id())

        sock.connect(addr)
        self.addr_to_sock_map[addr] = sock
        uid = self.addr_to_uid_map.get(addr, None)
        if uid is not None:
            self.uid_to_sock_map[uid] = sock

        # Identify ourselves
        msg = ('hello_from', 'tcp://%s:%d' % (self.addr, self.port))
        sock.send(pickle.dumps(msg))

        # Wait for the server to reply.
        # On the server, which received the hello_from first, this should
        # never have to wait.
        if addr not in self.addr_to_uid_map:
            logger.debug('Waiting for hello reply from server...')
            hello = AsyncHelloReply(addr)
            self.main_loop(delay=delay, wait_for=hello)
            if not hello.is_valid():
                raise TimeoutError('Connection to %s timed out; no reply received' % addr)

        if addr not in self.addr_to_uid_map:
            raise Exception('UID not resolved!')
        helper.request_client_proxy(self.addr_to_uid_map[addr], async=async)

    def send_to(self, dest, msg, bufs=None):
        '''
        Send <msg> to client <dest> (a uid).
        '''

        logger.debug('Sending %d bytes to %s', len(msg), dest)
        sock = self.uid_to_sock_map.get(dest, None)
        if sock is None:
            raise Exception('Unable to resolve destination %s' % dest)

        if bufs is None:
            sock.send(msg)
        else:
            msg = [msg, ]
            msg.extend(bufs)
            sock.send_multipart(msg)

        # This somehow reduces the latency.
        # My guess is it has to do with thread scheduling and/or the GIL.
        # It probably gets us higher in hierarchy as an I/O bound thread.
        if REDUCE_LATENCY:
            for i in range(10):
                os.getcwd()

    def timeout_add(self, delay, func, *args):
        '''
        Schedule a callback within the ZBE mainloop. If the function returns
        True it will be rescheduled to occur periodically. If it returns False
        it gets called only once.

        <delay> in msec.
        '''
        self._last_timeout_id += 1
        start = time.time()
        self._timeouts[self._last_timeout_id] = dict(
            start=start,
            delay=delay/1000.0,
            func=func,
            args=args
        )
        bisect.insort(self._scheduled_timeouts, (start+delay/1000.0, self._last_timeout_id))
        return self._last_timeout_id

    def timeout_remove(self, t_id):
        if t_id in self._timeouts:
            del self._timeouts[t_id]

    def _run_timeouts(self):
        now = time.time()
        while len(self._scheduled_timeouts) > 0 and self._scheduled_timeouts[0][0] < now:
            t, t_id = self._scheduled_timeouts.pop(0)
            info = self._timeouts.get(t_id, None)
            if info is None:
                continue

            try:
                ret = info['func'](*info['args'])

                # Only reschedule if returning True
                if not ret:
                    self.timeout_remove(t_id)
                    continue

                # Reschedule
                now2 = time.time()
                delta = (now2 - t) % info['delay']
                t_new = now2 + info['delay'] - delta
                logger.debug('Rescheduling timeout %d for %s', t_id, t_new - now2)
                bisect.insort(self._scheduled_timeouts, (t_new, t_id))
            except Exception, e:
                logger.error('Timeout call %d failed: %s', t_id, str(e))
                self.timeout_remove(t_id)

    def main_loop(self, delay=None, wait_for=None):
        '''
        Run the receiving main loop for a maximum of <delay> msec.

        If <wait_for> is specified (a single object or a list), the loop will
        terminate once all objects return True from is_valid().
        '''

        start = time.time()
        if delay is None:
            endtime = None
        else:
            endtime = start + delay / 1000.0

        # Convert wait_for to a list
        if wait_for is not None:
            if type(wait_for) in (types.TupleType, types.ListType):
                wait_for = list(wait_for)
            else:
                wait_for = [wait_for,]

        # If nothing to wait for, flush signal queue
        else:
            helper.flush_queue()

        poller = zmq.Poller()
        poller.register(self.srv, flags=zmq.POLLIN)

        while True:

            # Set curdelay based on end time
            if endtime is not None:
                curdelay = max(endtime - time.time(), 0)
            else:
                curdelay = 10000

            # Adjust curdelay if we have scheduled timeouts
            if len(self._scheduled_timeouts) > 0:
                to_delay = (self._scheduled_timeouts[0][0] - time.time()) * 1000.0
                to_delay = max(0, to_delay)
                curdelay = min(curdelay, to_delay)

            # Poll
            socks = poller.poll(curdelay)
            if len(socks) == 0:
                self._run_timeouts()
                if endtime is not None and time.time() >= endtime:
                    return False
                # Poll again
                continue

            # Receive message
            msgs = self.srv.recv_multipart()
            client = msgs[0]

            # Decode message
            try:
                info = pickle.loads(msgs[1])
            except Exception, e:
                logger.warning('Unable to decode object: %s [%r]', str(e), msgs[1])
                return

            # Process
            try:
                logger.debug('Starting Message processing %s', str(info))
                waiting = (wait_for is not None)
                helper.process_message(client, info, msgs[2:], waiting=waiting)
            except Exception, e:
                logger.warning('Failed to process message: %s\n%s', str(e), traceback.format_exc())
            finally:
                logger.debug('Message processed %s', str(info))

            # If we are waiting for call results and have them, return
            if wait_for is not None:
                i = 0
                while i < len(wait_for):
                    if wait_for[i].is_valid():
                        del wait_for[i]
                    else:
                        i += 1
                if len(wait_for) == 0:
                    return True

            # Check whether we timed out
            if endtime is not None and time.time() >= endtime:
                return False

    def _qt_timer(self):
        self.main_loop(delay=0)
        return True

    def add_qt_timer(self, interval=20):
        '''
        Install a callback timer at <interval> msec to integrate ZMQ message
        processing into the Qt4 main loop.
        '''

        if self.timer is not None:
            logger.warning('Timer already installed')
            return False

        from PyQt4 import QtCore, QtGui
        _app = QtGui.QApplication.instance()
        self.timer = QtCore.QTimer()
        QtCore.QObject.connect(self.timer, QtCore.SIGNAL('timeout()'), self._qt_timer)
        self.timer.start(interval)
        return True

