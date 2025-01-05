import glob
import numpy as np
import matplotlib.pyplot as plt

fns = glob.glob('out/*.pkl.npz')

for fn in fns:
    f = np.load(fn)
    ks = list(f.keys())
    for i, k in enumerate(ks):
        v = f[k]
        w = np.diff((v > -10).astype(int)) == 1
        w, = np.where(w)
        n = len(v)
        h, e = np.histogram(w, bins=30, range=(0, n))
        ff = h / (e[1] - e[0]) * 1000
        t = (e[1:] + e[:-1]) / 2 * 1e-3
        plt.plot(t, ff)
    plt.title(fn)
    plt.show()

