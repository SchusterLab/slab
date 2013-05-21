import numpy as np
import os
import pyqtgraph

def valid_h5file(file_path):
    if not file_path.endswith(".h5"):
        return False
    if os.path.exists(file_path):
        return True
    elif os.access(os.path.dirname(file_path), os.W_OK):
        #the file does not exists but write privileges are given
        return True
    else:
        #can not write there
        return False


def random_color(base=128):
    'A whitish random color. Adjust whiteness up by increasing base'
    return np.random.randint(base, 255, 3)


def method_string(name, args, kwargs):
    argstr = ",".join(map(str, args))
    kwargstr = ",".join(k+'='+str(i) for k, i in kwargs.items())
    if kwargstr != "":
        argstr += ","
    return name + "(" + argstr + kwargstr + ")"


def add_x_data(arr, slice=None):
    if slice is None:
        return np.vstack((np.arange(len(arr)), arr)).T
    a = np.vstack((np.arange(slice.start, slice.stop), arr)).T
    return a


def take_items(d, l):
    res = {}
    for i in l:
        if i in d:
            res[i] = d[i]
    return res


def separate_init_args(initargs):
    data_tree_args = ['rank', 'save', 'plot', 'parametric', 'x0', 'y0',
                      'xscale', 'yscale', 'xlabel', 'ylabel', 'zlabel']
    plot_args = ['title', 'labels', 'name', 'axisItems']
    pen_args = ['color', 'width', 'style', 'cosmetic', 'hsv']
    curve_args = ['pen', 'shadowPen', 'fillLevel', 'fillBrush', 'symbol',
                  'symbolPen', 'symbolBrush', 'symbolSize', 'pxMode', 'antialias', 'decimate', 'name']

    pen_dict = take_items(initargs, pen_args)
    if len(pen_dict) > 0 and 'pen' not in initargs:
        initargs['pen'] = pyqtgraph.mkPen(**pen_dict)

    return (take_items(initargs, data_tree_args),
            take_items(initargs, plot_args),
            take_items(initargs, curve_args))

#def canonicalize_data(data, slice=None):
#    arr = np.array(data)
#    parametric = False
#    if len(arr.shape) == 1:
#        arr = add_x_data(arr, slice)
#        rank = 1
#    elif len(arr.shape) == 2:
#        if arr.shape[0] == 1:
#            arr = add_x_data(arr[0,:], slice)
#            rank = 1
#        elif arr.shape[1] == 1:
#            arr = add_x_data(arr[:,0], slice)
#            rank = 1
#        elif arr.shape[0] == 2:
#            arr = arr.T
#            rank = 1
#            parametric = True
#        elif arr.shape[1] == 2:
#            rank = 1
#        else:
#            rank = 2
#    else:
#        raise NotImplementedError
#    return arr, rank#, parametric


def canonicalize_append_data(data):
    if isinstance(data, (int, float)):
        return np.array((None, data)), 1
    data = np.array(data)
    if len(data) == 2:
        return data, 1
    if len(data.shape) == 1:
        return data, 2
    if len(data.shape) == 2: # Rank 3
        raise NotImplementedError
    raise ValueError


def canonicalize_path(name_or_path):
    if isinstance(name_or_path, str):
        path = (name_or_path,)
    else:
        path = tuple(name_or_path)
    return path

