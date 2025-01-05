import os
import numpy as np
from sympy import Symbol, solve, Eq, S, symbols, exp, Function, Matrix
from sympy.printing.precedence import precedence
from sympy.printing.python import PythonPrinter
import textwrap

def get_grads(SYM=False, current=False):
    Iapp, C_m, V_m, g_Na, E_Na, g_K, E_K, g_L, E_L = symbols('Iapp C_m v g_Na E_Na g_K E_K g_L E_L')
    m, h, n, f = symbols('m h n f')
    gain, tau_f, f_tgt, lam, e, alpha = symbols('gain tau_f f_tgt lambda e alpha')
    alpha_m = (0.1*(25-V_m))/(exp((25-V_m)/10)-1) # type: ignore
    beta_m = 4*exp(-V_m/18) # type: ignore
    alpha_h = 0.07*exp(-V_m/20) # type: ignore
    beta_h = 1/(exp((30-V_m)/10)+1) # type: ignore
    alpha_n = (0.01*(10-V_m))/(exp((10-V_m)/10)-1) # type: ignore
    beta_n = 0.125*exp(-V_m/80) # type: ignore
    if SYM:
        alpha_m = Function('alpha_m')(V_m) # type: ignore
        beta_m = Function('beta_m')(V_m) # type: ignore
        alpha_h = Function('alpha_h')(V_m) # type: ignore
        beta_h = Function('beta_h')(V_m) # type: ignore
        alpha_n = Function('alpha_n')(V_m) # type: ignore
        beta_n = Function('beta_n')(V_m) # type: ignore
    else:
        alpha_m = (0.1*(25-V_m))/(exp((25-V_m)/10)-1) # type: ignore
        beta_m = 4*exp(-V_m/18) # type: ignore
        alpha_h = 0.07*exp(-V_m/20) # type: ignore
        beta_h = 1/(exp((30-V_m)/10)+1) # type: ignore
        alpha_n = (0.01*(10-V_m))/(exp((10-V_m)/10)-1) # type: ignore
        beta_n = 0.125*exp(-V_m/80) # type: ignore
    dm_dt = alpha_m*(1-m) - beta_m*m
    dh_dt = alpha_h*(1-h) - beta_h*h
    dn_dt = alpha_n*(1-n) - beta_n*n
    df_dt = 1000/(2*np.pi) * ((h-.5)*dm_dt - (m-.5)*dh_dt) / ((h-.5)**2 + (m-.5)**2) / tau_f - f / tau_f
    de_dt = (f - f_tgt)**2/2
    I_Na = g_Na*m**3*h*(V_m - E_Na)
    I_K = g_K*n**4*(V_m - E_K)
    I_L = g_L*(V_m - E_L)
    I_total =  I_Na + I_K + I_L - gain * Iapp
    dvdt = (1/C_m)*(-I_total)
    theta  = g_Na, g_K, gain
    s      = m, h, n, f, e
    ds     = dm_dt, dh_dt, dn_dt, df_dt, de_dt
    DVsym = np.array(symbols([f'DV_{t}' for t in theta])).reshape(1, -1)
    DSsym = np.array([symbols([f'D{si}_{t}' for t in theta]) for si in s])
    DV = (np.array([dvdt.diff(theta_i) for theta_i in theta]) +
          dvdt.diff(V_m) * DVsym +
          np.array([dvdt.diff(s_i) for s_i in s]) @ DSsym).reshape(1, -1)
    DS = (np.array([[si.diff(t) for t in theta] for si in ds]) +
          np.array([si.diff(V_m) for si in ds]).reshape(-1, 1)@DV +
          np.array([[si.diff(s_i) for s_i in s] for si in ds])@DSsym) #  THIS LAST ONE LOOKS WRONG!
    DV = np.array([e.simplify() for e in DV.flatten()]).reshape(DV.shape)
    if SYM:
        DS = np.array([e.simplify() for e in DS.flatten()]).reshape(DS.shape)
    DV = DV - lam * DVsym
    DS = DS - lam * DSsym
    # dv = Matrix(DV).transpose() ; ds = Matrix(DS)
    dthetadt = -alpha * DSsym[s.index(e)] * np.array([
        Symbol(f'{x}0') for x in theta]) # XXX: SHOULD BE |DS| OR SOMETHING!
    grads = []
    if current:
        grads.append((None, I_total))
    else:
        grads.append((V_m, dvdt))
    grads.extend(zip(s, ds))
    grads.extend(zip(DVsym.flatten(), DV.flatten()))
    grads.extend(zip(DSsym.flatten(), DS.flatten()))
    grads.extend(zip(theta, dthetadt))
    return grads

def grads_to_jacobian(grads, simplify=False):
    names, derivs = zip(*grads)
    Jac = []
    for deriv in derivs:
        Jac.append([deriv.diff(nm) for nm in names])
    Jac = Matrix(Jac)
    if simplify:
        Jac = np.array([e.simplify() for e in np.array(Jac).flatten()]).reshape(Jac.shape)
    return Jac

class NMODLPrinter(PythonPrinter):
    def _print_Pow(self, expr):
        PREC = precedence(expr)
        if expr.exp is S.Half:
            return "sqrt(%s)" % self._print(expr.base)
        if expr.is_commutative:
            if -expr.exp is S.Half:
                return "%s/sqrt(%s)" % tuple(map(lambda arg: self._print(arg), (S.One, expr.base)))
            if expr.exp is -S.One:
                return '%s/%s' % (self._print(S.One),
                                  self.parenthesize(expr.base, PREC, strict=False))
        e = self.parenthesize(expr.exp, PREC, strict=False)
        return '%s^%s' % (self.parenthesize(expr.base, PREC, strict=False), e)

    def _print_Float(self, expr):
        return f'{expr:f}'.rstrip('0')

    def _print_Rational(self, expr):
        return f'({expr})'

    def nope__getattribute__(self, k):
        'used to find new _print_X functions'
        print(k)
        f = super().__getattribute__(k)
        def wrapper(*a, **kw):
            out =  f(*a, **kw)
            print(f'# {k}({a}, {kw}) => {out}')
            return out
        if callable(f):
            return wrapper
        else:
            print(f'# {k} = {f}')
            return f

def make_nmodl(filename):
    p = NMODLPrinter()

    out = []

    def emit(x):
        n = len(x) - len(x.lstrip(' '))
        l, *ll = textwrap.wrap(x, 80)
        print(l)
        for l in ll:
            print(' '*n + ' '*4 + l)
        out.append(x)

    grads = get_grads(current=True)
    theta = ['g_Na', 'g_K', 'gain']
    ions = ['DV_g_Na', 'DV_g_K', 'DV_gain'] + theta
    param_defaults = {
            'C_m': 1,
            'alpha': 1,
            'lambda': 1,
            'E_K': -77,
            'E_L': -53,
            'E_Na': 50,
            'g_Na0': 120,
            'g_K0': 36,
            'gain0': 1,
            'g_L': 0.3,
            'clipbound': 100,
            'f_tgt': 1.0, 'tau_f': 10/1,
            'mu': 0,
            'sigma': .7,
            'theta': .1
            }
    #params = ['C_m', 'alpha', 'lambda', 'E_K', 'E_L', 'E_Na', 'g_L', 'clipbound']
    #params.extend(f'{th}0' for th in theta)
    params = list(param_defaults.keys())
    solve0 = ['m', 'h', 'n']
    state = ['ou']
    i = None
    for k, v in grads:
        if k is None:
            i = v
        elif str(k) not in params:
            state.append(str(k))
    emit('NEURON {')
    emit('    SUFFIX eighh')
    emit('    NONSPECIFIC_CURRENT i')
    for ion in  ions:
        emit(f'    USEION {ion} WRITE {ion}d')
    names = list(set(params) | set(state))
    emit(f'    RANGE {", ".join(names)}')
    emit('}')
    emit('FUNCTION abs(x) {')
    emit('    if (x < 0) { abs = -x }')
    emit('    else { abs = x} ')
    emit('}')
    emit('FUNCTION mmax(a, b) {')
    emit('    if (a < b) { mmax = b }')
    emit('    else { mmax = a} ')
    emit('}')
    emit('FUNCTION cliprange(x) {')
    emit('    if (x != x) { cliprange = 0 }')
    emit('    else if (x > +clipbound) { cliprange = +clipbound }')
    emit('    else if (x < -clipbound) { cliprange = -clipbound }')
    emit('    else { cliprange = x} ')
    emit('}')
    emit('PARAMETER {')
    for param in params:
        if param in param_defaults:
            emit(f'    {param} = {param_defaults.pop(param)}')
        else:
            assert False
            emit(f'    {param}')
    assert not param_defaults
    emit('}')
    emit('STATE {')
    emit(f'    {" ".join(state)}')
    emit('}')
    emit('INITIAL {')
    for k, v in grads:
        k = str(k)
        if k in solve0:
            res, = solve(Eq(v, 0), Symbol(k))
            emit(f'    {k} = {p.doprint(res)}')
    for name in state:
        if name in theta:
            emit(f'    {name} = {name}0')
            if name in ions:
                emit(f'    {name}d = {name}0')
        elif name not in solve0:
            emit(f'    {name} = 0')
    emit('}')
    emit('WHITE_NOISE { W }')
    emit('DERIVATIVE d {')
    emit('    LOCAL Iapp')
    emit('    Iapp = ou*ou*ou*ou')
    emit("    ou' = (mu - ou)*theta + sigma*W")
    for ion in  ions:
        emit(f'    {ion} = {ion}d')
    for k, v in grads:
        if k is not None:
            emit(f"    {k}' = {p.doprint(v)}")
    emit('}')
    emit('BREAKPOINT {')
    emit(f"    SOLVE d METHOD stochastic")
    emit(f"    LOCAL Iapp, gradientnorm")
    emit(f'    Iapp = ou*ou*ou*ou')
    emit(f"    i = {p.doprint(i)}")
    # clip by DX_*
    for group in set(str(k[0]).split('_')[0] for k in grads if k[0] is not None and str(k[0]).startswith('D')):
        emit(f"    ? START {group}")
        emit(f"    gradientnorm = 0")
        for k, v in grads:
            if not str(k).startswith(group):
                continue
            if k is not None and str(k).startswith('D'):
                emit(f"    gradientnorm = gradientnorm + {k}*{k}")
            #emit(f"    {k} = cliprange({k})")
        emit("    if (gradientnorm > clipbound*clipbound) {")
        emit("        gradientnorm = clipbound / sqrt(gradientnorm)")
        emit("    } else {")
        emit("        gradientnorm = 1")
        emit("    }")
        for k, v in grads:
            if not str(k).startswith(group):
                continue
            if k is not None and str(k).startswith('D'):
                emit(f"    {k} = {k}*gradientnorm")
        emit(f"    ? END {group}")
    for ion in  ions:
        emit(f'    {ion}d = {ion}')
    for name in theta:
        emit(f'    if ({name} < 0) {{ {name} = 0 }}')
    emit('}')

    with open(filename, 'w') as f:
        print('\n'.join(out), file=f)

os.makedirs('dual', exist_ok=True)
fn = 'dual/eighh.mod'
make_nmodl(fn)
os.system(f'modcc -A {fn}')
os.system('arbor-build-catalogue dual dual')
