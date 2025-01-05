import os

os.environ['JAX_PLATFORM'] = 'cpu'

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

opt_id = int(sys.argv[1])
tag = sys.argv[2]
fix = False
if '_' in tag:
    tag = tag.split('_')
    lr = float(tag[0])
    tfinal = int(float(tag[1]))
    if 'fix' in tag:
        fix = True
else:
    lr = float(tag)
    tfinal = 3000

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
# jax.config.update('jax_default_device', jax.devices('cpu')[0])
jax.config.update("jax_debug_nans", True)

rL = 100
cm = 0.01
beta = 100 / (cm * rL)

class Params:
    params = 'g_CaL', 'g_h', 'g_K_Ca', 'g_ld', \
             'g_la',  'g_ls',  'g_Na_s', 'g_Kdr_s', \
             'g_K_s', 'g_CaH', 'g_Na_a', 'g_K_a', \
             #'V_Na',  'V_K',   'V_Ca',   'V_h', \
             #'V_l'

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

    # V_Na    = staticmethod(lambda x: x[12] * Params.default['V_Na'])
    # V_K     = staticmethod(lambda x: x[13] * Params.default['V_K'])
    # V_Ca    = staticmethod(lambda x: x[14] * Params.default['V_Ca'])
    # V_h     = staticmethod(lambda x: x[15] * Params.default['V_h'])
    # V_l     = staticmethod(lambda x: x[16] * Params.default['V_l'])

if opt_id == 0:
    guess = Params.makedefault()
elif opt_id >= 1:
    prev = pickle.load(open(f'OPT_adam_{tag}_opt_{opt_id-1:03d}.pkl', 'rb'))
    guess = jnp.array(prev['update'])
else:
    exit(1)

print('guess')
print(guess)

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
        # alpha_x     = 0.13 * (v + 25) / (1 - jnp.exp(-(v + 25)/10))
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
        # alpha_x     = 0.13 * (v + 25) / (1 - jnp.exp(-(v + 25)/10))
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
        # beta_r      =  0.02*(v + 8.5) / (jnp.exp((v + 8.5)/5) - 1.0)
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
        # beta_r      =  0.02*(v + 8.5) / (jnp.exp((v + 8.5)/5) - 1.0)
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
        # alpha_x   =  0.13*(v + 25) / (1 - jnp.exp(-(v + 25)/10))
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
        # alpha_x   =  0.13*(v + 25) / (1 - jnp.exp(-(v + 25)/10))
        alpha_x     = 1.3 * xoverexpm1(-(v + 25)/10)
        beta_x    =  1.69 * jnp.exp(-(v + 35)/80)
        tau_x_inv =  alpha_x + beta_x
        x_inf     =  alpha_x / tau_x_inv
        dx_dt     =  (x_inf - x) * tau_x_inv
        return jnp.stack([dh_dt, dx_dt])

names_dV_dth = tuple([f'dV_d{param}' for param in Params.params])
def MAKEGRAD(Mech): # turn a mechanism into its gradient mechanism
    names_dS_dth = tuple([f'd{state}_d{param}' for state in Mech.state for param in Params.params])
    class Grad:
        state = names_dS_dth + Mech.state
        ions = names_dV_dth
        @staticmethod
        def init(v):
            return jnp.concatenate([
                jnp.zeros((len(names_dS_dth), v.shape[0])),
                Mech.init(v)
                ])
        @staticmethod
        def compute_current(v, state, params):
            S = state[len(names_dS_dth):]
            return Mech.compute_current(v, S, params)
        @staticmethod
        def state_gradient(v, state, ions, params):
            S = state[len(names_dS_dth):]
            dS_dt = Mech.state_gradient(v, S, params)
            dV_dth = ions.reshape((len(names_dV_dth), len(v)))
            dS_dth = state[:len(names_dS_dth)].reshape((len(Mech.state), len(Params.params), len(v)))
            @functools.partial(jax.vmap, in_axes=-1, out_axes=-1)
            def calc_dS_dth_dt(v, S, dV_dth, dS_dth):
                dS_dt_dth = jax.jacrev(Mech.state_gradient, 2)(v, S, params)
                dS_dt_dV = jax.jacrev(Mech.state_gradient, 0)(v, S, params)
                dS_dt_dS = jax.jacrev(Mech.state_gradient, 1)(v, S, params)
                return (dS_dt_dth + dS_dt_dV[:,None] @ dV_dth[None,:] + dS_dt_dS @ dS_dth - lam * dS_dth).flatten()
            return jnp.concatenate([calc_dS_dth_dt(v, S, dV_dth, dS_dth), dS_dt])
        @staticmethod
        def write_diff(v, state, ions, params):
            S = state[len(names_dS_dth):]
            dV_dth = ions.reshape((len(names_dV_dth), len(v)))
            dS_dth = state[:len(names_dS_dth)].reshape((len(Mech.state), len(Params.params), len(v)))
            @functools.partial(jax.vmap, in_axes=-1, out_axes=-1)
            def calc_dV_dth_dt(dV_dth, v, S, dS_dth):
                dV_dt_dth = jax.jacrev(Mech.compute_current, 2)(v, S, params) / -cm
                dV_dt_dV = jax.jacrev(Mech.compute_current, 0)(v, S, params) / -cm
                dV_dt_ds = jax.jacrev(Mech.compute_current, 1)(v, S, params) / -cm
                return dV_dt_dth + dV_dt_dV * dV_dth + dV_dt_ds @ dS_dth
            return calc_dV_dth_dt(dV_dth, v, S, dS_dth) - lam * dV_dth
    return Grad

def forward(decl, mech_name, params):
    if 1:
        jit_state_gradient = jax.jit(decl.state_gradient)
        jit_compute_current = jax.jit(decl.compute_current)
        jit_write_ions = jax.jit(decl.write_diff) if hasattr(decl, 'write_diff') else None
    else:
        jit_state_gradient = decl.state_gradient
        jit_compute_current = decl.compute_current
        jit_write_ions = decl.write_diff if hasattr(decl, 'write_diff') else None
    n = len(decl.state)
    arb_mech = acm.ArbMech()
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
            x = x + pp.dt * jit_state_gradient(v, x, params)
            pp.set_state(x)
    else:
        def advance_state(pp):
            v = pp.v[pp.node_index]
            x = pp.get_state()
            I = pp.get_diff_ions()
            x = x + pp.dt * jit_state_gradient(v, x, I, params)
            pp.set_state(x)
    if jit_write_ions is not None:
        def compute_currents(pp):
            v = pp.v[pp.node_index]
            x = pp.get_state()
            i = jit_compute_current(v, x, params)
            pp.i[pp.node_index] = i
            I = pp.get_diff_ions()
            d = jit_write_ions(v, x, I, params)
            pp.set_diff_ions(I + pp.dt * d)
    else:
        def compute_currents(pp):
            v = pp.v[pp.node_index]
            x = pp.get_state()
            i = jit_compute_current(v, x, params)
            pp.i[pp.node_index] = i
    def write_diff(_): pass
    #if jit_write_ions is not None:
    #    def write_diff(pp):
    #        v = pp.v[pp.node_index]
    #        x = pp.get_state()
    #        I = pp.get_diff_ions()
    #        d = jit_write_ions(v, x, I, params)
    #        set_diff_ions(pp, I + pp.dt * d)
    #else:
    #    def write_diff(_): pass
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
    forward(MAKEGRAD(Soma), 'io_soma_bwd', guess)
    forward(MAKEGRAD(Dend), 'io_dend_bwd', guess)
    forward(MAKEGRAD(Axon), 'io_axon_bwd', guess)

def build():
    import arbor
    so_name = acm.get_so_name()
    cat = arbor.load_catalogue(so_name)
    return cat

class Recipe(arbor.recipe):
    def __init__(self):
        super().__init__()
        self.the_props = arbor.neuron_cable_properties()
        self.the_props.catalogue.extend(CAT, '')
        for name in names_dV_dth:
            self.the_props.set_ion(ion=name, int_con=0 , ext_con=0, rev_pot=0, valence=1, diff=beta)
    def num_cells(self): return 1
    def cell_kind(self, _): return arbor.cell_kind.cable
    def cell_description(self, gid):
        morph = arbor.load_swc_arbor('./C4A.swc')
        label = arbor.label_dict()
        label['soma_group'] = '(tag 1)'
        label['dendrite_group'] = '(tag 3)'
        label['axon_group'] = '(tag 2)'
        decor = (
            arbor.decor()
            .paint('"soma_group"',      arbor.density('io_soma_bwd'))
            .paint('"dendrite_group"',  arbor.density('io_dend_bwd'))
            .paint('"axon_group"',      arbor.density('io_axon_bwd'))
            .set_property(cm=cm, rL=rL, Vm=-60)
        )
        return arbor.cable_cell(morph, decor, label)
    def probes(self, _): return [
            arbor.cable_probe_membrane_voltage_cell(),
            ] + [arbor.cable_probe_ion_diff_concentration_cell(name) for name in names_dV_dth]
    def global_properties(self, kind): return self.the_props

##################################################################################

register()
CAT = build()
recipe = Recipe()

sim = arbor.simulation(recipe)
vhandle = sim.sample((0, 0), arbor.regular_schedule(1.))
dvdthhandles = [sim.sample((0, 1+i), arbor.regular_schedule(1.))
                for i in range(len(names_dV_dth))]
a = time.time()
sim.run(tfinal=tfinal, dt=dt) # 350 seconds
b = time.time()
print(b - a)
out = {}
data, meta = sim.samples(vhandle)[0]
out['ts'] = b - a
out['tfinal'] = tfinal
out['dt'] = dt
out['v'] = data
out['v_meta'] = str(meta)
for handle, name in zip(dvdthhandles, names_dV_dth):
    data, meta = sim.samples(handle)[0]
    print(name, round(abs(data[:,1:]).max(), 3), abs(data[:,1:]).max())
    out[f'{name}'] = data
    out[f'{name}_meta'] = str(meta)

v = out['v'][:,1]

def fit_sine(v):
    #fhz = 5.6
    peaks = find_peaks(v, prominence=.1)[0]
    if len(peaks < 3):
        fhz = 5.8
        start = 0
        end = -1
    else:
        fhz = (1000/jnp.diff(peaks))[-3:].mean()
        idxs = peaks[-3:]
        start = idxs[0]
        end = idxs[-1]
    if fix:
        fhz = 5.8
    # amp = 48.82 / 2
    # amp = 25 / 2

    amp = v[start:end].ptp()/2
    phase = (v.argmax()*1e-3 * fhz % 1) * 2*jnp.pi
    offset = (v[start:end].min() + v[start:end].max())/2
    t = jnp.arange(0, len(v)) * 1e-3
    ttune = t[start:end]
    def f(t, offset, amp, fhz, phase):
        return offset + amp * jnp.sin(2*jnp.pi * t * fhz + phase)
    def lsq(n):
        phase = n
        est = f(ttune, offset, amp, fhz, phase)
        print(est.shape)
        print(v[start:end].shape)
        return (jnp.abs(est - v[start:end])).mean()
    res = minimize(lsq, jnp.array([phase]), method='BFGS')
    phase = res.x.item()
    if amp < 15:
        amp = 15
    if amp > 30:
        amp = 30
    est = f(t, offset, amp, fhz, phase)
    return est


#v = v[len(v)//2:]
with open('./tgtv2.pkl', 'rb') as f:
    tgtdata = pickle.load(f)
vtgt = tgtdata['v'][:tfinal,1]
e = vtgt - v

# e = e / jnp.abs(e).max()
update = []

idxs = find_peaks(v, prominence=.1)[0][-3:]

for param in Params.params:
    g = out[f'dV_d{param}'][:,1]
    d = (g * e)
    d2 = (g * (1-e)**2)
    update.append(-d.mean())
    out[f'g_{param}'] = g
    out[f'd_{param}'] = d
grads = jnp.array(update)


optimizer = optax.adam(lr)
if opt_id == 0:
    opt_state = optimizer.init(guess)
elif opt_id >= 1:
    # prev = pickle.load(open(f'OPT_{tag}_opt_{opt_id-1:03d}.pkl', 'rb'))
    # optimizer = prev['optimizer']
    opt_state = prev['opt_state']
else:
    exit(1)

updates, opt_state = optimizer.update(grads, opt_state)
params_next = optax.apply_updates(guess, updates)


# out['outgrad'] = update
# update = update * lr
# update = update.at[update< -0.1].set(-.1)
# update = update.at[update> +0.1].set(+.1)
# print(update)
out['update'] = params_next
# out['optimizer'] = optimizer
out['opt_state'] = opt_state
out['vtgt'] = vtgt
plt.plot(v)
plt.plot(vtgt)
plt.savefig(f'OPT_adam_{tag}_opt_{opt_id:03d}.png')

with open(f'./OPT_adam_{tag}_opt_{opt_id:03d}.pkl', 'wb') as f:
    pickle.dump(out, f)
