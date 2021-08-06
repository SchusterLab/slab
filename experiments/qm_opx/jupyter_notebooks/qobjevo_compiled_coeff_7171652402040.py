
# This file is generated automatically by QuTiP.
import numpy as np
import scipy.special as spe
import scipy
from qutip.qobjevo import _UnitedFuncCaller

def proj(x):
    if np.isfinite(x):
        return (x)
    else:
        return np.inf + 0j * np.imag(x)

sin = np.sin
cos = np.cos
tan = np.tan
asin = np.arcsin
acos = np.arccos
atan = np.arctan
pi = np.pi
sinh = np.sinh
cosh = np.cosh
tanh = np.tanh
asinh = np.arcsinh
acosh = np.arccosh
atanh = np.arctanh
exp = np.exp
log = np.log
log10 = np.log10
erf = scipy.special.erf
zerf = scipy.special.erf
sqrt = np.sqrt
real = np.real
imag = np.imag
conj = np.conj
abs = np.abs
norm = lambda x: np.abs(x)**2
arg = np.angle

class _UnitedStrCaller(_UnitedFuncCaller):
    def __init__(self, funclist, args, dynamics_args, cte):
        self.funclist = funclist
        self.args = args
        self.dynamics_args = dynamics_args
        self.dims = cte.dims
        self.shape = cte.shape

    def set_args(self, args, dynamics_args):
        self.args = args
        self.dynamics_args = dynamics_args

    def dyn_args(self, t, state, shape):
        # 1d array are to F ordered
        mat = state.reshape(shape, order="F")
        for name, what, op in self.dynamics_args:
            if what == "vec":
                self.args[name] = state
            elif what == "mat":
                self.args[name] = mat
            elif what == "Qobj":
                if self.shape[1] == shape[1]:  # oper
                    self.args[name] = Qobj(mat, dims=self.dims)
                elif shape[1] == 1:
                    self.args[name] = Qobj(mat, dims=[self.dims[1],[1]])
                else:  # rho
                    self.args[name] = Qobj(mat, dims=self.dims[1])
            elif what == "expect":  # ket
                if shape[1] == op.cte.shape[1]: # same shape as object
                    self.args[name] = op.mul_mat(t, mat).trace()
                else:
                    self.args[name] = op.expect(t, state)

    def __call__(self, t, args={}):
        if args:
            now_args = self.args.copy()
            now_args.update(args)
        else:
            now_args = self.args
        out = []

        w = now_args['w']
        theta = now_args['theta']
        out.append(sin(w*t))
        out.append(sin(w*t+theta))

        return out

    def get_args(self):
        return self.args

