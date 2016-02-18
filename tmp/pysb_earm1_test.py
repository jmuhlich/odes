#!/usr/bin/env python
"""Reproduce figures 4A and 4B from the EARM 1.0 publication (Albeck et
al. 2008)."""

from __future__ import print_function
from pysb.integrate import Solver
from numpy import array, linspace, logspace, floor, empty_like, argmax
import matplotlib.pyplot as plt

from pysb.examples.earm_1_0 import model

import pysb.integrate
pysb.integrate.Solver._use_inline = False

# saturating level of ligand (corresponding to ~1000 ng/ml SuperKiller TRAIL)
Lsat = 6E4; 

# relationship of ligand concentration in the model (in # molecules/cell) to actual TRAIL concentration (in ng/ml)
Lfactor = model.parameters['L_0'].value / 50;

L_0_baseline = model.parameters['L_0'].value


def scan():
    t = linspace(0, 20*3600, 20*60+1)  # 20 hours, in seconds, 1 min sampling
    dt = t[1] - t[0]

    Ls_exp = Lsat / array([1, 4, 20, 100, 500])
    Td_exp = [144.2, 178.7, 236,   362.5, 656.5]
    Td_std = [32.5,   32.2,  36.4,  78.6, 171.6]
    Ts_exp = [21.6,   23.8,  27.2,  22.0,  19.0]
    Ts_std = [9.5,     9.5,  12.9,   7.7,  10.5]

    CVenv = 0.2
    # num steps was originally 40, but 15 is plenty smooth enough for screen display
    Ls = floor(logspace(1,5,15)) 

    fs = empty_like(Ls)
    Ts = empty_like(Ls)
    Td = empty_like(Ls)
    s = Solver(model, t, integrator='cvode')
    print("Scanning over %d values of L_0" % len(Ls))
    for i in range(len(Ls)):
        model.parameters['L_0'].value = Ls[i]

        print("  L_0 = %g" % Ls[i])
        s.run()
        x = s.yobs

        fs[i] = (x['PARP_unbound'][0] - x['PARP_unbound'][-1]) / x['PARP_unbound'][0]
        dP = 60 * (x['PARP_unbound'][:-1] - x['PARP_unbound'][1:]) / (dt * x['PARP_unbound'][0])  # in minutes
        ttn = argmax(dP)
        dPmax = dP[ttn]
        Ts[i] = 1 / dPmax  # minutes
        Td[i] = t[ttn] / 60  # minutes

    plt.figure("Figure 4A")
    plt.plot(Ls/Lfactor, Td, 'g-', Ls/Lfactor, (1-CVenv)*Td, 'g--', Ls/Lfactor, (1+CVenv)*Td, 'g--')
    plt.errorbar(Ls_exp/Lfactor, Td_exp, Td_std, None, 'go', capsize=0),
    plt.ylabel('Td (min)'),
    plt.xlabel('TRAIL (ng/ml)'),
    a = plt.gca()
    a.set_xscale('log')
    a.set_xlim((min(Ls) / Lfactor, max(Ls) / Lfactor))
    a.set_ylim((0, 1000))
    plt.show()
    
    model.parameters['L_0'].value = L_0_baseline

if __name__ == '__main__':
    scan()
