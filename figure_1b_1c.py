from typing import NamedTuple
import arbor
import time
import jax
import jax.numpy as jnp
import arbor_pycat
import numpy as np
import matplotlib.pyplot as plt

rL = 100
cm = 0.01
beta = 100 / (cm * rL)

def exprelr(x): return jnp.where(jnp.isclose(x, 0), 1., x / jnp.expm1(x))
def alpha_m(V): return exprelr(-0.1*V - 4.0)
def alpha_h(V): return 0.07*jnp.exp(-0.05*V - 3.25)
def alpha_n(V): return 0.1*exprelr(-0.1*V - 5.5)
def beta_m(V):  return 4.0*jnp.exp(-(V + 65.0)/18.0)
def beta_h(V):  return 1.0/(jnp.exp(-0.1*V - 3.5) + 1.0)
def beta_n(V):  return 0.125*jnp.exp(-0.0125*V - 0.8125)

class State(NamedTuple):
    m: jax.Array
    h: jax.Array
    n: jax.Array

class Params(NamedTuple):
    w: float

class Const(NamedTuple):
    gna: jax.Array
    gk: jax.Array
    gl: jax.Array
    ena: jax.Array
    ek: jax.Array
    el: jax.Array
    iapp: jax.Array

@jax.jit
def cm_dot_v(v: jax.Array, state: State, params: Params, const: Const):
    ina = const.gna*state.m**3*state.h*(v - const.ena)
    ik = const.gk*state.n**4*(v - const.ek)
    il = const.gl*(v - const.el)
    return ina + ik + il - params.w * const.iapp

@jax.jit
def dot_state(v: jax.Array, state: State):
    dot_m = (alpha_m(v)*(1-state.m) - beta_m(v)*state.m)
    dot_h = (alpha_h(v)*(1-state.h) - beta_h(v)*state.h)
    dot_n = (alpha_n(v)*(1-state.n) - beta_n(v)*state.n)
    return dot_m, dot_h, dot_n

calc_didv = jax.jit(jax.vmap(jax.jacrev(cm_dot_v, argnums=0), [0, 0, None, 0]))
calc_dids = jax.jit(jax.vmap(jax.jacrev(cm_dot_v, argnums=1), [0,0,None,0]))
didtheta = jax.jit(jax.jacrev(cm_dot_v, argnums=2))
dfdv = jax.jit(jax.vmap(jax.jacrev(dot_state, argnums=0)))
dfds = jax.jit(jax.vmap(jax.jacrev(dot_state, argnums=1)))
dfdtheta = 0 # it's zeros

nparams = len(Params._fields)

def load_dsdtheta(pp):
    return State(
            m=Params(*[getattr(pp, f'dmd{i}') for i in range(nparams)]), # type: ignore
            h=Params(*[getattr(pp, f'dhd{i}') for i in range(nparams)]), # type: ignore
            n=Params(*[getattr(pp, f'dnd{i}') for i in range(nparams)])) # type: ignore

dmdtheta_matrix = \
            [(f'dmd{i}', '', 0) for i in range(nparams)] + \
            [(f'dhd{i}', '', 0) for i in range(nparams)] + \
            [(f'dnd{i}', '', 0) for i in range(nparams)]

@arbor_pycat.register
class ExampleMech(arbor_pycat.CustomMechanism):
    name = 'custom_hh'
    kind = 'density'
    state_vars = [('m', '', 0), ('h', '', 0), ('n', '', 0), ('t', '', 0)] + dmdtheta_matrix
    parameters = [('gna', '',   0.120), ('gk', '',    0.036), ('gl', '',    0.0003),
                  ('ena', '',  55    ), ('ek', '',  -77    ), ('el', '',  -65), ('iapp', '', 0.)]
    ions = [arbor_pycat.IonInfo(name=f'x{i}', write_int_concentration=True, use_diff_concentration=True) for i in range(nparams)]
    def init_mechanism(self, pp):
        assert all(np.diff(pp.node_index) == 1)
        v = pp.v[pp.node_index]
        pp.m = alpha_m(v) / (alpha_m(v) + beta_m(v))
        pp.h = alpha_h(v) / (alpha_h(v) + beta_h(v))
        pp.n = alpha_n(v) / (alpha_n(v) + beta_n(v))
    def advance_state(self, pp):
        v = pp.v[pp.node_index]
        state = State(pp.m, pp.h, pp.n)
        dot_m, dot_h, dot_n = dot_state(v, state)
        pp.t[:] += pp.dt;         pp.m[:] += pp.dt * dot_m
        pp.h[:] += pp.dt * dot_h; pp.n[:] += pp.dt * dot_n
        dfmdv, dfhdv, dfndv = dfdv(v, state)
        dfmds, dfhds, dfnds = dfds(v, state)
        dsdtheta = load_dsdtheta(pp)
        for j in range(nparams):
            idx = getattr(pp, f'index_x{j}')
            dvdth = getattr(pp, f'x{j}d')[idx]
            getattr(pp, f'dmd{j}')[:] += pp.dt*(
                    dfdtheta
                    + dfmdv * dvdth
                    + dfmds.m * dsdtheta.m[j] + dfmds.h * dsdtheta.h[j] + dfmds.n * dsdtheta.n[j]
                    )
            getattr(pp, f'dhd{j}')[:] += pp.dt*(
                    dfdtheta
                    + dfhdv * dvdth
                    + dfhds.m * dsdtheta.m[j] + dfhds.h * dsdtheta.h[j] + dfhds.n * dsdtheta.n[j]
                    )
            getattr(pp, f'dnd{j}')[:] += pp.dt*(
                    dfdtheta
                    + dfndv * dvdth
                    + dfnds.m * dsdtheta.m[j] + dfnds.h * dsdtheta.h[j] + dfnds.n * dsdtheta.n[j]
                    )
    def compute_currents(self, pp):
        v = pp.v[pp.node_index]
        en = pp.iapp if pp.t[0] % 100 > 95 else 0*pp.iapp
        params = Params(w=1.)
        state = State(pp.m, pp.h, pp.n)
        const = Const(pp.gna, pp.gk, pp.gl, pp.ena, pp.ek, pp.el, en)
        pp.i[pp.node_index] = cm_dot_v(v, state, params, const)
        dids = calc_dids(v, state, params, const)
        didv = calc_didv(v, state, params, const)
        dsdtheta = load_dsdtheta(pp)
        for j, didtheta_j in enumerate(didtheta(v, state, params, const)):
            idx = getattr(pp, f'index_x{j}')
            dvdtheta = getattr(pp, f'x{j}d')
            dvdtheta[idx] += pp.dt*(
                    -didtheta_j/cm
                    -didv * dvdtheta[idx]/ cm
                    -dids.n * dsdtheta.n[j] / cm
                    -dids.m * dsdtheta.m[j] / cm
                    -dids.h * dsdtheta.h[j] / cm)


cat = arbor_pycat.build()

# import matplotlib.pyplot as plt

tree = arbor.segment_tree()
last = tree.append(arbor.mnpos, arbor.mpoint(-10, 0, 0, 10), arbor.mpoint(10, 0, 0, 10), tag=1)
for _ in range(10):
    last = tree.append(last, arbor.mpoint(0, 0, 0, .5), arbor.mpoint(40, 0, 0, .5), tag=2)

# (2) Define the soma and its midpoint
labels = arbor.label_dict({"soma": "(tag 1)", "axon": "(tag 2)", "midpoint": "(location 0 0.5)"})

# (3) Create cell and set properties
decor = (
    arbor.decor()
    .set_property(Vm=-68.5)
    .paint('"soma"', arbor.density("custom_hh", dict(iapp=0.1)))
    .paint('"axon"', arbor.density("custom_hh"))
    .set_property(rL=rL, cm=cm)
    .discretization(arbor.cv_policy_every_segment())
)

class single_recipe(arbor.recipe):
    def __init__(self):
        arbor.recipe.__init__(self)
        self.the_props = arbor.neuron_cable_properties()
        self.the_props.catalogue.extend(cat, '')
        for i in range(nparams):
            self.the_props.set_ion(ion=f'x{i}', int_con=0 , ext_con=0, rev_pot=0, valence=1, diff=beta)
    def num_cells(self): return 1
    def cell_kind(self, _): return arbor.cell_kind.cable
    def cell_description(self, gid):
        cc = arbor.cable_cell(tree, decor, labels)
        arbor.write_component(cc, '/tmp/cc')
        return cc
    def probes(self, _): return [
            arbor.cable_probe_membrane_voltage_cell(),
            arbor.cable_probe_ion_diff_concentration_cell('x0')] + [
                arbor.cable_probe_density_state_cell('custom_hh', name)
                for name, _, _ in dmdtheta_matrix]
    def global_properties(self, kind): return self.the_props
recipe = single_recipe()
sim = arbor.simulation(recipe)
vhandle = sim.sample((0, 0), arbor.regular_schedule(0.1))
x0handle = sim.sample((0, 1), arbor.regular_schedule(0.1))
dmdth_handles = [sim.sample((0, 2+i), arbor.regular_schedule(0.1)) for i in range(len(dmdtheta_matrix))]
sim.run(tfinal=1)
a = time.time()
sim.run(tfinal=1+250)
b = time.time()
data, meta = sim.samples(vhandle)[0]
fig, ax = plt.subplots(nrows=3, sharex=True)
t = data[:, 0]
cmap = plt.get_cmap('bone')
for i in range(1, data.shape[1]):
    v = data[:, i]
    ax[0].plot(t, 0*i+v, color=cmap(i / data.shape[1]))
data, meta = sim.samples(x0handle)[0]
t, x0 = data[:,0], data[:,1:]
for i in range(x0.shape[1]):
    ax[1].plot(t, 0*i+x0[:,i], color=cmap(i / x0.shape[1]))
cmaps = 'Reds Blues Greens Oranges Purples'.split()
# cmaps = ['Grays']*10
for j, (name, _, _) in enumerate(dmdtheta_matrix):
    cmap = plt.get_cmap(cmaps[j])
    data, meta = sim.samples(dmdth_handles[j])[0]
    t, x0 = data[:,0], data[:,1:]
    for i in range(x0.shape[1]):
        ax[2].plot(t, 0*i+x0[:,i], color=cmap(i / x0.shape[1]), label=name if i == 0 else None)
ax[0].text(98, 0.4, 'Vm', color='black')
ax[1].text(98.5, 40, 'dV/dw', color='black')
ax[2].text(98, 0.4, 'dm/dw', color='red')
ax[2].text(100, -0.3, 'dh/dw', color='blue')
ax[2].text(101, 0.2, 'dn/dw', color='green')
ax[2].set_xlim(92, 108)
ax[0].set_ylabel('Vm (mV)')
ax[1].set_ylabel('dV/dw (mV)')
ax[2].set_ylabel('dstate/dw')
ax[2].set_xlabel('Time (ms)')
plt.tight_layout()
plt.show()
