import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt


i = 0
with open(f'./dsdth.pkl', 'rb') as f:
    data = pickle.load(f)



params =[
'g_K_a',
'g_Na_a',
'g_la',

'g_CaL', 
'g_Kdr_s',
'g_Na_s',
'g_K_s',
'g_ls', 

'g_h',
'g_K_Ca',
'g_CaH',
'g_ld',
         ]
def_states = {
        'axon': ['ax', 'ah'],
        'soma': ['sk', 'sl', 'sn', 'sh', 'sx'],
        'dend': ['dq', 'ds', 'dr', 'dcaconc'],
}

nstates = 0
for comp, states in def_states.items(): 
    for state in states:
        nstates += 1

o = np.empty((nstates, len(params), 1000))

i = 0
names = []
for comp, states in def_states.items(): 
    for state in states:
        names.append(state)
        for j, param in enumerate(params):
            key = f'{comp}_d{state}_d{param}'
            x = data[key]
            x = x[:,1:].mean(1)
            o[i, j] = x
        i += 1

idxs = np.array([680, 695, 713, 728, 792, 837, 843])

o = o / o.std(2)[:,:,None]
lim = np.abs(o).max()
for i in idxs:
    plt.clf()
    plt.imshow(o[:,:,i], cmap='RdBu', vmin=-lim, vmax=lim)
    plt.yticks(np.arange(len(names)), names, rotation=45)
    plt.xticks(np.arange(len(params)), params, rotation=45)
    plt.show()

plt.clf()
v = data['v'][:,1]
plt.plot(v)
plt.scatter(idxs, v[idxs])
for i in idxs:
    plt.text(i, v[i], str(i))
plt.xlim(idxs[0]-100, idxs[-1]+100)
plt.show()

