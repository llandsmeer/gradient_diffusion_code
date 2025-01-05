import pickle
import numpy as np
import glob

for fn in glob.glob('out/*.pkl'):
    print(fn)

    with open(fn, 'rb') as f:
        data = pickle.load(f)

    out = {}
    for i in range(len(data['trace'])):
        out[f'v{i}'] = data['trace'][i]['voltage'][0]
        out[f'f{i}'] = data['trace'][i]['states']['f']

    np.savez(fn + '.npz', **out)
