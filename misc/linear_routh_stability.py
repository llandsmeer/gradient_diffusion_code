import sympy as sm
from tbcontrol.symbolic import routh

gl, cm, lam, alpha = sm.symbols(r'g_l C_m lambda alpha', real=True, positive=True)
R, Rl, omega = sm.symbols(r'R, R_l, omega', real=True, positive=True)
omega_th = sm.symbols(r'omega_theta', real=True, positive=True)
beta_th = sm.symbols(r'beta_theta', real=True, positive=True)

omega_th = 0 # XXX it's just line noise, doesn't add more information (adding it makes it more stable)

# 
# m = sm.Matrix([
#         [-gl/cm, 0, gl/cm],
#         [gl/(lam*cm), -lam, 0],
#         [0, -alpha, 0]
#     ])
# 
# p = m.charpoly()
# A = routh(p)
# 
# eqs = [e > 0 for e in A[:, 0] if (e >0) != True]
# assert len(eqs) == 1
# 
# eq, = eqs
# print(sm.latex(sm.simplify(eq)))

print('\n'*10)

m = sm.Matrix([
        [-gl/cm - (omega**2/2 * sm.pi * R * cm * Rl), 0, gl/cm],
        [1, -lam, 0],
        [0, -alpha, -omega_th*beta_th] # XXX ZERO OUT omega_th*beta_th
    ])

p = m.charpoly()
A = routh(p)

eqs = [e > 0 for e in A[:, 0] if (e >0) != True]
assert len(eqs) == 1

eq, = eqs
eq = sm.simplify(eq)
assert type(eq).__name__ == 'StrictGreaterThan'
acon = sm.solve(eq, alpha)
print(sm.latex(acon))

om0 = acon.subs(omega, 0).subs(omega_th, 0).simplify().expand()
print(sm.latex(om0))
input()
