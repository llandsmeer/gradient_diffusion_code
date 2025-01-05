import sys
import os
import arbor
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

out = {}

print(arbor.config()['version'])
print(arbor.config()['source'])

tfinal = 2000000
print(tfinal)
dt = 0.005

radius = 1
length = 10
ncv = 1

cm = 1

rL = 100
diff = 100 / rL / cm

f_tgt = 1
factor = 1

if len(sys.argv) > 1:
    f_tgt = float(sys.argv[1])
if len(sys.argv) > 2:
    factor = float(sys.argv[2])

out['tfinal'] = tfinal
out['dt'] = dt
out['radius'] = radius
out['ncv'] = ncv
out['cm'] = cm
out['rL'] = rL
out['f_tgt'] = f_tgt
out['factor'] = factor

tag = f'{f_tgt:03.3f}_{factor:03.3f}'

if os.path.exists(f'out/{tag}.pkl'):
    print('exists!')
    exit(0)

out[tag] = tag

args = {
    'f_tgt': f_tgt,
    'tau_f': 10 * 1e3 / 1,
    'C_m': cm,
    'lambda': 1 * 1 / 10e3,
    'alpha': factor * 1 * 0.1 / 10e3,
    'clipbound': 1,
    'gain0': 0.06
}

out['args'] = args

# {'f_tgt': 0.01, 'tau_f': 4000.0, 'C_m': 1, 'lambda': 2.5e-05, 'alpha': 0.1, 'clipbound': 100, 'gain0': 0.03}


tree = arbor.segment_tree()
tree.append(arbor.mnpos, arbor.mpoint(0, 0, 0, radius), arbor.mpoint(0, 0, length, radius), tag=1)
labels = arbor.label_dict()
props = arbor.neuron_cable_properties()
props.catalogue.extend(arbor.load_catalogue('./dual-catalogue.so'), '')
#props.catalogue.extend(arbor.load_catalogue('./sde-catalogue.so'), 'sde_')
name = 'eighh'
mech = props.catalogue[name]
states = list(mech.state.keys())
ions = list(mech.ions.keys())
decor = (arbor.decor()
    .set_property(Vm=-53, cm=cm, rL=rL)
    .discretization(arbor.cv_policy_fixed_per_branch(ncv))
    .paint('(all)', arbor.density(name, args))
    #.paint('(all)', arbor.density('sde_ou', dict(mu=0, theta=.1, sigma=0.7)))
    )
for ion in ions:
    props.set_ion(ion=ion, valence=1, int_con=0, ext_con=0, rev_pot=0, diff=diff)

cell = arbor.cable_cell(tree, decor, labels)

#probes = [
#    arbor.cable_probe_density_state('(root)', name, state)
#    for state in states
#] + [
#    arbor.cable_probe_membrane_voltage_cell(),
#    arbor.cable_probe_total_current_cell(),
#    arbor.cable_probe_ion_diff_concentration_cell('x')
#]

trace_states = 'f Df_g_Na Df_g_K Df_gain m n h'.split()
probes = [
    arbor.cable_probe_membrane_voltage_cell(),
]
for state in trace_states:
    probes.append(arbor.cable_probe_density_state('(root)', name, state))
probes_ion_start = len(probes)
for ion in ions:
    probes.append(
    arbor.cable_probe_ion_diff_concentration_cell(ion)
    )

class Recipe(arbor.recipe):
    def probes(self, _): return probes
    def num_cells(self): return 1
    def cell_kind(self, _): return arbor.cell_kind.cable
    def cell_description(self, _): return cell
    def global_properties(self, _): return props
    def num_params(self): return len(keys)
    def trace(self, tfinal, dt, config=None):
        sim = arbor.simulation(self)
        for i in range(len(probes)): sim.sample((0, i), arbor.regular_schedule(1))
        sim.run(tfinal=tfinal, dt=dt)
        t = np.array(sim.samples(0)[0][0].T[0])
        voltage = np.array(sim.samples(0)[0][0]).T[1:]
        states = {k: np.array(sim.samples(i+1)[0][0].T[1]) for i, k in enumerate(trace_states)}
        #current = np.array(sim.samples(len(states)+1)[0][0]).T[1:]
        xions = {}
        for i, ion in zip(range(probes_ion_start, len(probes)), ions):
            xions[ion] = np.array(sim.samples(i)[0][0]).T[1:]
        return t, voltage, states, xions

recipe = Recipe()
t, V, states, ions = recipe.trace(tfinal, dt)
t /= 1e3

out['t'] = t
out['V'] = V
out['states'] = states
out['ions'] = ions

df = dict(t=t)
df.update(dict(zip([f'v{i}' for i in range(len(V))], V)))
df.update(states)
for k, v in ions.items():
    df.update(dict(zip([f'{k}{i}' for i in range(len(v))], v)))
df = pd.DataFrame(df)

# out['df'] = df

fig, ax = plt.subplots(nrows=5, sharex=True, gridspec_kw=dict(hspace=0))
ax[0].set_ylabel('Voltage (mV)')
ax[1].set_ylabel('Gradients')
ax[2].set_ylabel('Parameters')
ax[3].set_ylabel('Gating')
ax[4].set_ylabel('Frequency')

for i, vi in enumerate(V):
    ax[0].plot(t, vi, '-', label=f'v{i}')
    m = np.isnan(vi)
    ax[0].plot(t[m], np.zeros_like(m[m]), '-', color='red', lw=4)

f = states.pop('f')
ax[4].plot(t, f, color='magenta', label='f')
ax[4].axhline(args['f_tgt'], ls='--', color='magenta', label='f_tgt')

for k, v in states.items():
    ax[1].plot(t, v, '-', label=k)

for k, v in ions.items():
    print(k)
    ls = '-'
    if k == 'gain':
        a = ax[2].twinx()
        ls = '--'
    elif k in ['m', 'n', 'h']:
        a = ax[3]
    elif k in ['g_K', 'g_Na']:
        a = ax[2]
    else:
        a = ax[1]
    for i, xi in enumerate(v):
        a.plot(t, xi, label=k, ls=ls)
    if k == 'gain':
        a.legend()


for idx in range(len(ax)):
    ax[idx].axvline(0, color='black')
    ax[idx].axvline(tfinal*1e-3, color='black')
    ax[idx].legend()

ax[-1].set_xlabel('Time (s)')
states['f'] = f
#plt.show()
with open(f'out/{tag}.pkl', 'wb') as f:
    pickle.dump(out, file=f)
plt.savefig(f'out/{tag}.svg')
plt.savefig(f'out/{tag}.png')
#input()

