'''
This file exists because I want to know the relation between
beta & Cm. We find beta = 100 / rL / Cm. That makes sense
after looking at the units. rL is in Ohm.cm
'''

import arbor
import numpy as np

# I'm seeking a relation between rL and beta
rL = np.random.uniform(0.1, 1000) # voltage diffusion term!

# given any variation in these parameters:
tfinal = np.random.uniform(10, 10000)
radius = np.random.uniform(0.1, 100)
length = np.random.uniform(1, 2000)
ncv = np.random.randint(2, 50)
cm = np.random.uniform(0.1, 100)
dt = np.random.uniform(0.001, 0.025)
fraction_along = np.random.uniform(0, 1)
perturbation = np.random.uniform(0.1, 10)
zero = np.random.uniform(0.1, 10)

# this relation leads to the same ion/voltage behaviour
diff = 100 / rL / cm # ion diffusion

# test case:
tree = arbor.segment_tree()
tree.append(arbor.mnpos, arbor.mpoint(0, 0, 0, radius), arbor.mpoint(0, 0, length, radius), tag=1)
props = arbor.neuron_cable_properties()
props.set_ion(ion='x', valence=1, int_con=zero, ext_con=0, rev_pot=0, diff=diff)
decor = arbor.decor()
decor.set_property(Vm=zero, cm=cm, rL=rL)
decor.discretization(arbor.cv_policy_fixed_per_branch(ncv))
decor.paint(f'(cable 0 0 {fraction_along})', Vm=perturbation)
decor.paint(f'(cable 0 0 {fraction_along})', ion_name='x', int_con=perturbation)
class Recipe(arbor.recipe):
    def probes(self, _): return [
        arbor.cable_probe_membrane_voltage_cell(),
        arbor.cable_probe_ion_diff_concentration_cell('x')]
    def num_cells(self): return 1
    def cell_kind(self, _): return arbor.cell_kind.cable
    def cell_description(self, _): return arbor.cable_cell(tree, decor, arbor.label_dict())
    def global_properties(self, _): return props

recipe = Recipe()
sim = arbor.simulation(recipe)
sim.sample((0, 0), arbor.regular_schedule(0.1))
sim.sample((0, 1), arbor.regular_schedule(0.1))
sim.run(tfinal=tfinal, dt=dt)
tv = np.array(sim.samples(0)[0][0])
t, v = tv.T[0], tv.T[1:]
c = np.array(sim.samples(1)[0][0]).T[1:]
assert np.allclose(v, c)
print('equation cm*rL*beta=100 holds')

# import matplotlib.pyplot as plt
# for i, vi in enumerate(v): plt.plot(t, vi, '-', color='red', alpha=0.5)
# for i, xi in enumerate(c): plt.plot(t, xi, '-', color='green', alpha=0.5)
# plt.title('red = voltage ; green = ion')
# plt.show()
