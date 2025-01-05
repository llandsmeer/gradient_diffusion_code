import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy import exp

tstop = 1000
dt = 0.005
trecord = 0.1
v = -60

C_m   =   1.0 ; ou   =  0.0
E_K   = -77.0 ; E_L   = -53.0 ; E_Na = 50.0
g_Na  = 120.0 ; g_K   =  36.0 ; g_L  =  0.3
theta = 0.1   ; sigma = 0.7

alpha_m = (0.1*(25-v))/(exp((25-v)/10)-1)
beta_m = 4*exp(-v/18)
alpha_h = 0.07*exp(-v/20)
beta_h = 1/(exp((30-v)/10)+1)
alpha_n = (0.01*(10-v))/(exp((10-v)/10)-1)
beta_n = 0.125*exp(-v/80)
m0 = alpha_m / (alpha_m + beta_m)
h0 = alpha_h / (alpha_h + beta_h)
n0 = alpha_n / (alpha_n + beta_n)
m, h, n = m0, h0, n0
record_every = int(round(trecord / dt))

step = 0
trace = []
for step in tqdm(range(int(round(tstop / dt)))):
    alpha_m = (0.1*(25-v))/(exp((25-v)/10)-1)
    beta_m = 4*exp(-v/18)
    alpha_h = 0.07*exp(-v/20)
    beta_h = 1/(exp((30-v)/10)+1)
    alpha_n = (0.01*(10-v))/(exp((10-v)/10)-1)
    beta_n = 0.125*exp(-v/80)
    dm_dt = alpha_m*(1-m) - beta_m*m
    dh_dt = alpha_h*(1-h) - beta_h*h
    dn_dt = alpha_n*(1-n) - beta_n*n
    de_dt = 0
    I_Na = g_Na*m**3*h*(v - E_Na)
    I_K = g_K*n**4*(v - E_K)
    I_L = g_L*(v - E_L)
    Iapp = ou**4
    I_total =  I_Na + I_K + I_L - Iapp
    dv_dt = (1/C_m)*(-I_total)
    if step % record_every == 0:
        trace.append(dict(step=step, t=step*dt, v=v, n=n, m=m, h=h, i=I_total,
            dm=dm_dt))
    if np.isnan(dv_dt) or np.isnan(dm_dt) or np.isnan(dn_dt) or np.isnan(dh_dt):
        breakpoint()
    v += dt * dv_dt
    m += dt * dm_dt
    h += dt * dh_dt
    n += dt * dn_dt
    ou += np.random.normal() * sigma * np.sqrt(dt) - ou * theta * dt
    step = step + 1
    # print(f'{dv_dt=:.2f} {dm_dt=:.2f} {dh_dt=:.2f} {dn_dt=:.2f} {v=:.2f} {m=:.2f} {h=:.2f} {n=:.2f}')

trace = pd.DataFrame(trace)
plt.plot(trace.m-.5, trace.dm)
plt.plot(trace.m-.5, trace.h-.5)
plt.show()
fig, ax = plt.subplots(nrows=3, sharex=True)
ax[0].plot(trace.t, trace.v)
ax[0].axhline(int(E_L), ls='--', color='black')
ax[0].axhline(20, ls='--', color='black')
ax[1].plot(trace.t, trace.m, label='m') # m is the slow closer, let's use that
ax[1].plot(trace.t, trace.n, label='n')
ax[1].plot(trace.t, trace.h, label='h')
ax[1].plot(trace.t, trace.dm, label='dm_dt')
phi = np.arctan2(trace.m - 0.5, trace.dm)
phi = np.unwrap(phi) / (2*np.pi)
ax[2].plot(trace.t, phi)

phi = np.arctan2(trace.m - 0.5, trace.h- 0.5)
phi = np.unwrap(phi) / (2*np.pi)
ax[2].plot(trace.t, phi)

#ax[1].plot(trace.t, trace.n, '--', label='n')
#ax[1].plot(trace.t, trace.h, label='h')
#ax[1].plot(trace.t, trace.m**3*trace.h, label='Na = $m^3h$')
#ax[1].plot(trace.t, trace.n**4, '--', label='K = $n^4$')
ax[1].legend()
plt.show()
