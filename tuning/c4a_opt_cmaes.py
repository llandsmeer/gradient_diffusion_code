import os

os.environ['JAX_PLATFORM'] = 'cpu'

import cma
import numpy as np
import sys
import pickle
from scipy.signal import find_peaks
from jax.scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
import jax
import functools
from tqdm import tqdm
import jax.numpy as jnp
import arbor
import arbor_pycat._core as acm
import optax

NCELLS = 20

opt_id = int(sys.argv[1])
tag = sys.argv[2]
fix = False
if '_' in tag:
    tag = tag.split('_')
    tfinal = int(float(tag[1]))
    if 'fix' in tag:
        fix = True
    if '50' in tag:
        print('50!')
        NCELLS = 50
else:
    tfinal = 1000

if 'delta' in tag:
    print('!'*100)
    tfinal = 100 + opt_id * 10
    if opt_id > 100:
        tfinal = tfinal + (opt_id-100)*50
    if tfinal > 3000:
        tfinal = 3000


if os.path.exists(f'./OPT_adam_{tag}_opt_{opt_id:03d}.pkl'):
    print('ALREADY EXIST')
    exit(1)

dt = 0.025

lam = 0

##################################################################################

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_debug_nans", True)

rL = 100
cm = 0.01
beta = 100 / (cm * rL)

class Params:
    params = 'g_CaL', 'g_h', 'g_K_Ca', 'g_ld', \
             'g_la',  'g_ls',  'g_Na_s', 'g_Kdr_s', \
             'g_K_s', 'g_CaH', 'g_Na_a', 'g_K_a', \

    default = dict(
        g_CaL   =   0.015,  g_h     =   0.025,  g_K_Ca  =   0.300,  g_ld    =   1.3e-5,
        g_la    =   1.3e-5, g_ls    =   1.3e-5, g_Na_s  =   0.030,  g_Kdr_s =   0.030,
        g_K_s   =   0.015,  g_CaH   =   0.090,  g_Na_a  =   0.200,  g_K_a   =   0.200,
        V_Na    =  55.0,    V_K     = -75.0,    V_Ca    = 120.0,    V_h     = -43.0,
        V_l     =  10.0
        )
    @staticmethod
    def make(d):
        p = jnp.array([ d['g_CaL'] / Params.default['g_CaL'], d['g_h'] / Params.default['g_h'],   d['g_K_Ca'] / Params.default['g_K_Ca'], d['g_ld'] / Params.default['g_ld'],
                        d['g_la'] / Params.default['g_la'],  d['g_ls'] / Params.default['g_ls'],  d['g_Na_s'] / Params.default['g_Na_s'], d['g_Kdr_s'] / Params.default['g_Kdr_s'],
                        d['g_K_s'] / Params.default['g_K_s'], d['g_CaH'] / Params.default['g_CaH'], d['g_Na_a'] / Params.default['g_Na_a'], d['g_K_a'] / Params.default['g_K_a'],
                        d['V_Na'] / Params.default['V_Na'],  d['V_K'] / Params.default['V_K'],   d['V_Ca'] / Params.default['V_Ca'],   d['V_h'] / Params.default['V_h'],
                        d['V_l'] / Params.default['V_l'] ])
        return p
    @staticmethod
    def makedefault(): return jnp.ones(len(Params.params))
    g_CaL   = staticmethod(lambda x: x[ 0] * Params.default['g_CaL'])
    g_h     = staticmethod(lambda x: x[ 1] * Params.default['g_h'])
    g_K_Ca  = staticmethod(lambda x: x[ 2] * Params.default['g_K_Ca'])
    g_ld    = staticmethod(lambda x: x[ 3] * Params.default['g_ld'])
    g_la    = staticmethod(lambda x: x[ 4] * Params.default['g_la'])
    g_ls    = staticmethod(lambda x: x[ 5] * Params.default['g_ls'])
    g_Na_s  = staticmethod(lambda x: x[ 6] * Params.default['g_Na_s'])
    g_Kdr_s = staticmethod(lambda x: x[ 7] * Params.default['g_Kdr_s'])
    g_K_s   = staticmethod(lambda x: x[ 8] * Params.default['g_K_s'])
    g_CaH   = staticmethod(lambda x: x[ 9] * Params.default['g_CaH'])
    g_Na_a  = staticmethod(lambda x: x[10] * Params.default['g_Na_a'])
    g_K_a   = staticmethod(lambda x: x[11] * Params.default['g_K_a'])

    V_Na    = staticmethod(lambda _: Params.default['V_Na'])
    V_K     = staticmethod(lambda _: Params.default['V_K'])
    V_Ca    = staticmethod(lambda _: Params.default['V_Ca'])
    V_h     = staticmethod(lambda _: Params.default['V_h'])
    V_l     = staticmethod(lambda _: Params.default['V_l'])

if opt_id == 0:
    es = cma.CMAEvolutionStrategy(Params.makedefault(), .1, {'popsize': NCELLS})
elif opt_id >= 1:
    prev = pickle.load(open(f'OPT_adam_{tag}_opt_{opt_id-1:03d}.pkl', 'rb'))
    es = prev['opt_state']
else:
    exit(1)

guess = es.ask()

def xoverexpm1(x):
    'x / (exp(x) - 1)'
    return jnp.where(jnp.isclose(x, 0), 1.0-x/2, x / jnp.expm1(x))

class Soma:
    state = 'sk', 'sl', 'sh', 'sn', 'sx'
    @staticmethod
    def init(v):
        k           = 1 / (1 + jnp.exp(-(v + 61)/4.2))
        l           = 1 / (1 + jnp.exp( (v + 85)/8.5))
        h           = 1 / (1 + jnp.exp( (v + 70)/5.8))
        n           = 1 / ( 1 + jnp.exp(-(v +  3)/10))
        alpha_x     = 1.3 * xoverexpm1(-(v + 25)/10)
        beta_x      = 1.69 * jnp.exp(-(v + 35)/80)
        x           = alpha_x / (alpha_x + beta_x)
        return jnp.stack([k, l, h, n, x])
    @staticmethod
    def compute_current(v, state, params):
        k, l, h, n, x = state
        I_leak = Params.g_ls(params) * (v - Params.V_l(params))
        Ical   = Params.g_CaL(params) * k * k * k * l * (v - Params.V_Ca(params))
        m_inf  = 1 / (1 + jnp.exp(-(v + 30)/5.5))
        Ina    = Params.g_Na_s(params) * m_inf**3 * h * (v - Params.V_Na(params))
        Ikdr   = Params.g_Kdr_s(params) * n**4 * (v - Params.V_K(params))
        Ik     = Params.g_K_s(params) * x**4 * (v - Params.V_K(params))
        return I_leak + Ik + Ikdr + Ina + Ical
    @staticmethod
    def state_gradient(v, state, params):
        del params
        k, l, h, n, x = state
        k_inf       = 1 / (1 + jnp.exp(-(v + 61)/4.2))
        l_inf       = 1 / (1 + jnp.exp( (v + 85)/8.5))
        tau_l       = (20 * jnp.exp((v + 160)/30) / (1 + jnp.exp((v + 84) / 7.3))) + 35
        dk_dt       = k_inf - k
        dl_dt       = (l_inf - l) / tau_l
        h_inf       = 1 / (1 + jnp.exp( (v + 70)/5.8))
        tau_h       = 3 * jnp.exp(-(v + 40)/33)
        dh_dt       = (h_inf - h) / tau_h
        n_inf       = 1 / ( 1 + jnp.exp(-(v +  3)/10))
        tau_n       = 5 + (47 * jnp.exp( (v + 50)/900))
        dn_dt       = (n_inf - n) / tau_n
        alpha_x     = 1.3 * xoverexpm1(-(v + 25)/10)
        beta_x      = 1.69 * jnp.exp(-(v + 35)/80)
        tau_x_inv   = alpha_x + beta_x
        x_inf       = alpha_x / tau_x_inv
        dx_dt       = (x_inf - x) * tau_x_inv
        return jnp.stack([dk_dt, dl_dt, dh_dt, dn_dt, dx_dt])

class Dend:
    state = 'dcaconc', 'dr', 'ds', 'dq'
    @staticmethod
    def init(v):
        caconc      =  jnp.full_like(v, 3.715)
        alpha_r     =  1.7 / (1 + jnp.exp(-(v - 5)/13.9))
        beta_r      =  0.1*xoverexpm1((v + 8.5)/5)
        r           =  alpha_r / (alpha_r + beta_r)
        alpha_s     =  jnp.where(0.00002 * caconc < 0.01, 0.00002 * caconc, 0.01)
        s           =  alpha_s / (alpha_s + 0.015)
        q           =  1 / (1 + jnp.exp((v + 80)/4))
        return jnp.stack([caconc, r, s, q])
    @staticmethod
    def compute_current(v, state, params):
        _, r, s, q  = state
        I_leak      =  Params.g_ld(params) * (v - Params.V_l(params))
        Icah        =  Params.g_CaH(params) * r * r * (v - Params.V_Ca(params)) * 0
        Ikca        =  Params.g_K_Ca(params) * s * (v - Params.V_K(params))
        Ih          =  Params.g_h(params) * q * (v - Params.V_h(params))
        return I_leak + Icah + Ikca + Ih
    @staticmethod
    def state_gradient(v, state, params):
        caconc, r, s, q = state
        Icah        =  Params.g_CaH(params) * r * r * (v - Params.V_Ca(params))
        alpha_r     =  1.7 / (1 + jnp.exp(-(v - 5)/13.9))
        beta_r      = 0.1*xoverexpm1((v + 8.5)/5)
        tau_r_inv5  =  (alpha_r + beta_r)
        r_inf       =  alpha_r / tau_r_inv5
        dr_dt       =  (r_inf - r) * tau_r_inv5 * 0.2
        alpha_s     =  jnp.where(
                0.00002 * caconc < 0.01,
                0.00002 * caconc,
                0.01)
        tau_s_inv   =  alpha_s + 0.015
        s_inf       =  alpha_s / tau_s_inv
        ds_dt       =  (s_inf - s) * tau_s_inv
        q_inf       =  1 / (1 + jnp.exp((v + 80)/4))
        tau_q_inv   =  jnp.exp(-0.086*v - 14.6) + jnp.exp(0.070*v - 1.87)
        dq_dt       =  (q_inf - q) * tau_q_inv
        dCa_dt      =  -3 * Icah - 0.075 * caconc
        return jnp.stack([dCa_dt, dr_dt, ds_dt, dq_dt])

class Axon:
    state = 'ah', 'ax'
    @staticmethod
    def init(v):
        h     =  1 / (1 + jnp.exp( (v+60)/5.8))
        alpha_x     = 1.3 * xoverexpm1(-(v + 25)/10)
        beta_x    =  1.69 * jnp.exp(-(v + 35)/80)
        tau_x_inv =  alpha_x + beta_x
        x     =  alpha_x / tau_x_inv
        return jnp.stack([h, x])
    @staticmethod
    def compute_current(v, state, params):
        h, x = state
        m_inf     =  1 / (1 + jnp.exp(-(v+30)/5.5))
        I_leak    =  Params.g_la(params) * (v - Params.V_l(params))
        Ina       =  Params.g_Na_a(params) * m_inf**3 * h * (v - Params.V_Na(params))
        Ik        =  Params.g_K_a(params) * x**4 * (v - Params.V_K(params))
        return I_leak + Ina + Ik
    @staticmethod
    def state_gradient(v, state, params):
        del params
        h, x = state
        h_inf     =  1 / (1 + jnp.exp( (v+60)/5.8))
        tau_h     =  1.5 * jnp.exp(-(v+40)/33)
        dh_dt     =  (h_inf - h) / tau_h
        alpha_x     = 1.3 * xoverexpm1(-(v + 25)/10)
        beta_x    =  1.69 * jnp.exp(-(v + 35)/80)
        tau_x_inv =  alpha_x + beta_x
        x_inf     =  alpha_x / tau_x_inv
        dx_dt     =  (x_inf - x) * tau_x_inv
        return jnp.stack([dh_dt, dx_dt])

def forward(decl, mech_name, params):
    jit_state_gradient = jax.jit(decl.state_gradient)
    jit_compute_current = jax.jit(decl.compute_current)
    jit_write_ions = jax.jit(decl.write_diff) if hasattr(decl, 'write_diff') else None
    arb_mech = acm.ArbMech()
    assert 0 == arb_mech.add_parameter('cellid', '', 0.)
    for i, name in enumerate(decl.state):
        assert i == arb_mech.add_state(name, '', 0.)
    for i, name in enumerate(getattr(decl, 'ions', [])):
        assert i == arb_mech.add_ion(
            name = name,
            write_int_concentration = True,
            write_ext_concentration = False,
            use_diff_concentration = True,
            write_rev_potential = False,
            read_rev_potential = True,
            read_valence = False,
            verify_valence = False,
            expected_valence = 1)
    def init(pp):
        v = pp.v[pp.node_index]
        val = decl.init(v)
        pp.set_state(val)
    if not hasattr(decl, 'ions'):
        def advance_state(pp):
            v = pp.v[pp.node_index]
            x = pp.get_state()
            p = params[int(pp.param(0)[0])]
            x = x + pp.dt * jit_state_gradient(v, x, p)
            pp.set_state(x)
    else:
        def advance_state(pp):
            v = pp.v[pp.node_index]
            x = pp.get_state()
            I = pp.get_diff_ions()
            p = params[int(pp.param(0)[0])]
            x = x + pp.dt * jit_state_gradient(v, x, I, p)
            pp.set_state(x)
    if jit_write_ions is not None:
        def compute_currents(pp):
            v = pp.v[pp.node_index]
            x = pp.get_state()
            p = params[int(pp.param(0)[0])]
            i = jit_compute_current(v, x, p)
            pp.i[pp.node_index] = i
            I = pp.get_diff_ions()
            d = jit_write_ions(v, x, I, p)
            pp.set_diff_ions(I + pp.dt * d)
    else:
        def compute_currents(pp):
            v = pp.v[pp.node_index]
            x = pp.get_state()
            i = jit_compute_current(v, x, params[int(pp.param(0)[0])])
            pp.i[pp.node_index] = i
    def write_diff(_): pass
    arb_mech.set_init(init)
    arb_mech.set_advance_state(advance_state)
    arb_mech.set_compute_currents(compute_currents)
    arb_mech.set_write_ions(write_diff)
    arb_mech.set_name(mech_name)
    acm.register(arb_mech)

def register():
    forward(Soma, 'io_soma_fwd', guess)
    forward(Dend, 'io_dend_fwd', guess)
    forward(Axon, 'io_axon_fwd', guess)

def build():
    import arbor
    so_name = acm.get_so_name()
    cat = arbor.load_catalogue(so_name)
    return cat

class Recipe(arbor.recipe):
    def __init__(self, ncells):
        super().__init__()
        self.ncells = ncells
        self.the_props = arbor.neuron_cable_properties()
        self.the_props.catalogue.extend(CAT, '')
    def num_cells(self): return self.ncells
    def cell_kind(self, _): return arbor.cell_kind.cable
    def cell_description(self, gid):
        morph = arbor.load_swc_arbor('./C4A.swc')
        label = arbor.label_dict()
        label['soma_group'] = '(tag 1)'
        label['dendrite_group'] = '(tag 3)'
        label['axon_group'] = '(tag 2)'
        decor = (
            arbor.decor()
            .paint('"soma_group"',      arbor.density('io_soma_fwd', dict(cellid=gid)))
            .paint('"dendrite_group"',  arbor.density('io_dend_fwd', dict(cellid=gid)))
            .paint('"axon_group"',      arbor.density('io_axon_fwd', dict(cellid=gid)))
            .set_property(cm=cm, rL=rL, Vm=-60)
        )
        return arbor.cable_cell(morph, decor, label)
    def probes(self, _): return [
            arbor.cable_probe_membrane_voltage_cell(),
            ]
    def global_properties(self, kind): return self.the_props

##################################################################################

register()
CAT = build()
recipe = Recipe(NCELLS)

sim = arbor.simulation(recipe)
handles = []
for i in range(NCELLS):
    vhandle = sim.sample((i, 0), arbor.regular_schedule(1.))
    handles.append(vhandle)
a = time.time()
sim.run(tfinal=tfinal, dt=dt) # 350 seconds
b = time.time()
print(b - a)
out = {}

out['ts'] = b - a
out['tfinal'] = tfinal
out['dt'] = dt
out['ncells'] = NCELLS

with open('./tgtv2.pkl', 'rb') as f:
    tgtdata = pickle.load(f)
vtgt = tgtdata['v'][:tfinal,1]
print(vtgt)

out['vtgt'] = vtgt

L = []

for i in range(NCELLS):
    data, meta = sim.samples(handles[i])[0]
    out[f'v{i}'] = data
    out[f'v{i}_meta'] = str(meta)
    v = data[:,1]
    e = vtgt - v
    loss = (e**2).mean() / 2
    L.append(loss)

print(L)
es.tell(guess, L)

out['L'] = L
out['opt_state'] = es

with open(f'./OPT_adam_{tag}_opt_{opt_id:03d}.pkl', 'wb') as f:
    pickle.dump(out, f)
