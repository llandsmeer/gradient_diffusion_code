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

tfinal = 20000000
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

WEIGHT = 0.01
DELAY = 10
NSRC = 5
NCELLS = 10

if len(sys.argv) > 1:
    f_tgt = float(sys.argv[1])
if len(sys.argv) > 2:
    factor = float(sys.argv[2])
if len(sys.argv) > 3:
    WEIGHT = float(sys.argv[3])
    DELAY = float(sys.argv[4])
    NSRC = int(sys.argv[5])
    NCELLS = int(sys.argv[6])



tag = f'{tfinal}_{f_tgt:03.3f}_{factor:03.3f}_{WEIGHT}_{DELAY}_{NSRC}_{NCELLS}'

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
labels['synapse_site'] = '(location 0 0.5)'
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
    .place('"synapse_site"', arbor.synapse("expsyn"), "syn")
    .place('(root)', arbor.threshold_detector(-10), "detector")
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
    def num_cells(self): return NCELLS
    def cell_kind(self, _): return arbor.cell_kind.cable
    def cell_description(self, _): return cell
    def connections_on(self, gid):
        cand = list([x for x in range(self.num_cells()) if x != gid])
        np.random.seed(gid)
        srcs = np.random.choice(cand, NSRC)
        out = []
        for src in srcs:
            out.append(arbor.connection((src, "detector"), "syn", WEIGHT, DELAY))
        #src = (gid-1)%self.ncells
        return out
    def global_properties(self, _): return props
    def num_params(self): return len(keys)
    def trace(self, tfinal, dt, config=None):
        sim = arbor.simulation(self)
        handles = {}
        for j in range(self.num_cells()):
            for i in range(len(probes)):
                handles[j, i ] = sim.sample((j, i), arbor.regular_schedule(1))
        sim.run(tfinal=tfinal, dt=dt)
        out = []
        for j in range(self.num_cells()):
            t = np.array(sim.samples(handles[j, 0])[0][0].T[0])
            voltage = np.array(sim.samples(handles[j, 0])[0][0]).T[1:]
            states = {k: np.array(sim.samples(handles[j, i+1])[0][0].T[1]) for i, k in enumerate(trace_states)}
            xions = {}
            for i, ion in zip(range(probes_ion_start, len(probes)), ions):
                xions[ion] = np.array(sim.samples(handles[j, i])[0][0]).T[1:]
            out.append(dict(
                t=t,
                voltage=voltage,
                states=states,
                xions=xions))
        return out

recipe = Recipe()
trace = recipe.trace(tfinal, dt)

out['trace'] = trace
out['tfinal'] = tfinal
out['dt'] = dt
out['radius'] = radius
out['ncv'] = ncv
out['cm'] = cm
out['rL'] = rL
out['f_tgt'] = f_tgt
out['factor'] = factor

with open(f'out/{tag}.pkl', 'wb') as f:
    pickle.dump(out, file=f)
