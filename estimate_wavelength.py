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

l = 0
R_H = 1.097373E7

K = 1000
dx = (xend - x0) / (K - 1)

plot_idx = 0

fix_joint, ax_joint = plt.subplots(3, 2)
ax_joint_flat = ax_joint.flatten()

fix_joint.suptitle('Hydrogen Wavefunction Probability Densities', fontsize=22)

nvec = np.arange(1, 7) # Principal quantum numbers
Evec = np.full_like(nvec, np.nan, dtype=float)

for i, n in enumerate(nvec):
    E0 = -3 / (2 * n ** 2) # Initial eigenvalue guess
    Evec[i] = fsolve(optimizer, x0=E0, args=(l, 0, xend, K))[0] # Numerical eigenvalue

lambdavec = 1 / (R_H * (Evec[2:] - Evec[1])) * 1E9
print(lambdavec)
