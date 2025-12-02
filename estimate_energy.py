import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import simpson
from matplotlib import pyplot as plt

from NumericalMethods import numerov, numerov_rev

def f(r, E, l):
    if r == 0:
        if l == 0:
            return -np.inf
        else:
            return np.inf
    else:
        return -2 / r - 1 * E + l * (l + 1) / r ** 2

def optimizer(E, l, x0, xend, K):
    fopt = lambda r: f(r, E, l)

    y0 = np.exp(np.sqrt(-E) * xend)
    y1 = np.exp(np.sqrt(-E) * (xend - dx))
    
    _, yarr_backward = numerov_rev(fopt, x0, xend, y0, y1, K)

    _, yarr_forward = numerov(fopt, x0, xend, 0, yarr_backward[1], K)

    return yarr_forward[-1] - y0

x0 = 0
xend = 50

K = 1000
dx = (xend - x0) / (K - 1)

plot_idx = 0

fix_joint, ax_joint = plt.subplots(3, 2)
ax_joint_flat = ax_joint.flatten()

fix_joint.suptitle('Hydrogen Wavefunction Probability Densities', fontsize=22)

nvec = np.arange(1, 4) # Principal quantum numbers

for n in nvec:
    lvec = np.arange(n) # Angular momentum quantum numbers
    for l in lvec:
        E0 = -3 / (2 * n ** 2) # Initial eigenvalue guess
        E = fsolve(optimizer, x0=E0, args=(l, 0, xend, K))[0] # Numerical eigenvalue
        Ea = -1 / n ** 2 # Analytical eigenvalue

        print(E0)
        print(E)
        print(Ea) # hihihihihihihihihihi
        print(f'{np.abs((Ea - E) / Ea) * 100:.2e}\n')

        y0 = np.exp(np.sqrt(-E) * xend)
        y1 = np.exp(np.sqrt(-E) * (xend - dx))

        ff = lambda r: f(r, E, l)

        r, y = numerov_rev(ff, 0, xend, y0, y1, K)

        P = y ** 2

        norm = simpson(P, r)
        P_norm = P / norm
        fig, ax = plt.subplots()

        ax.plot(r, P_norm)

        ax.grid(which='both')
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlabel('$\\frac{r}{a_0}$', fontsize=16)
        ax.set_ylabel('$P$', fontsize=16)
        ax.set_title(f'Numerical Radial Probability for $n = {n}$, $l = {l}$', fontsize=18)

        ax_joint_flat[plot_idx].plot(r, P_norm)

        ax_joint_flat[plot_idx].grid(which='both')
        ax_joint_flat[plot_idx].tick_params(axis='both', labelsize=12)
        ax_joint_flat[plot_idx].set_xlabel('$\\frac{r}{a_0}$', fontsize=16)
        ax_joint_flat[plot_idx].set_ylabel('$P$', fontsize=16)
        ax_joint_flat[plot_idx].set_title(f'Numerical Radial Probability for $n = {n}$, $l = {l}$', fontsize=18)

        plot_idx += 1

plt.show()
