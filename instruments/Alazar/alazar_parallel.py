from multiprocessing import Process, Lock, Array, Value
import ctypes as C
import numpy as np
Az = C.CDLL(r'C:\Windows\SysWow64\ATSApi.dll')

U8 = C.c_uint8
U16 = C.c_uint16
U32 = C.c_uint32

bytesPerBuffer = None
bytesPerRecord = None
buffersPerMerge = 100
bufferCount = 8
iterations = 100
handle = None
timeout = None

def worker(az_lock, avg_data, acq_count):
    buf_data = C.c_uint8 * bytesPerBuffer
    arr_data = np.ctypeslib.as_array(buf_data)
    sum_data = np.zeros(bytesPerBuffer)
    for _ in range(iterations):
        sum_data.fill(0)
        for i in range(buffersPerMerge):
            az_lock.acquire()
            if i is not 0:
                Az.AlazarPostAsyncBuffer(handle, buf_data, U32(bytesPerBuffer))
            Az.AlazarWaitAsyncBufferComplete(handle, buf_data, U32(timeout))
            az_lock.release()
            sum_data += arr_data
        avg_data.acquire()
        avg_data *= (acq_count - 1.) / acq_count
        avg_data += arr_data * acq_count
        acq_count +=  buffersPerMerge
        avg_data.release()

def acquire_avg_data_parallel():
    az_lock = Lock()
    avg_data = Array(C.c_longdouble, bytesPerBuffer)
    acq_count = Value(U32)

    workers = []
    for _ in range(bufferCount):
        workers.append(Process(target=worker, args=(az_lock, avg_data, acq_count)))
        
    Az.AlazarStartCapture(handle)
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    return np.array(avg_data)

    
