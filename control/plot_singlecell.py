import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import signal

fns = '''1.000_0.100.pkl
1.000_0.200.pkl
1.000_0.300.pkl
1.000_0.400.pkl
1.000_0.500.pkl
2.000_0.300.pkl
3.000_0.300.pkl
4.000_0.300.pkl
5.000_0.300.pkl'''.split()

class Dummy():
    def __init__(self, *a, **k):
        self.a = a
        self.k = k

class A(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'pandas.core.frame' and name == 'DataFrame':
            return Dummy
        if module == 'pandas.core.internals.managers' and name == 'BlockManager':
            return Dummy
        if module == 'pandas.core.internals.blocks' and name == 'new_block':
            return Dummy
        return super().find_class(module, name)


for fn in fns:
    with open(fn, 'rb') as f:
        a = A(f)
        data = a.load()
    f = data['states']['f']
    t = np.arange(len(f))
    f_cutoff = .1
    f_sample = 1000
    sos = signal.butter(N=4, Wn=f_cutoff / (f_sample / 2), btype='low', output='sos')
    f = signal.sosfilt(sos, f)
    f = signal.decimate(f, 10)
    f = signal.decimate(f, 10)
    f = signal.decimate(f, 10)
    f = scipy.signal.savgol_filter(f, 101, 1)
    factor = data["factor"]
    tgt = data["f_tgt"]
    print(len(f))
    if factor == 0.3:
        plt.plot(f, label=f'{factor} {tgt}')
        plt.axhline(data['f_tgt'])
        plt.ylim(0, 6)
        plt.title(fn)
plt.legend()
plt.savefig(f'singlecell.svg')
plt.show()
