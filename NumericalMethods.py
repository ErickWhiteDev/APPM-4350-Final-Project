import numpy as np

def numerov(f, x0, xend, y0, y1, K):
    xarr = np.linspace(x0, xend, K) # Discretization grid
    dx = (xend - x0) / (K - 1)
    dx2 = dx ** 2

    yarr = np.full_like(xarr, np.nan) # y evaluations at gridpoints
    phiarr = np.full_like(xarr, np.nan) # phi evaluations at gridpoints
    farr = [f(x) for x in xarr]

    yarr[0] = y0
    yarr[1] = y1    

    phi = lambda k: 0 if np.isinf(farr[k]) else yarr[k] * (1 - dx2 * farr[k] / 12)
    phiarr[0] = phi(0)
    phiarr[1] = phi(1)

    for k in range(2, K):
        phiarr[k] = 2 * phiarr[k - 1] - phiarr[k - 2] + dx2 * farr[k - 1] * yarr[k - 1]
        yarr[k] = phiarr[k] / (1 - dx2 * farr[k] / 12)

    return xarr, yarr

def numerov_rev(f, x0, xend, y0, y1, K):
    xarr = np.linspace(x0, xend, K) # Discretization grid
    dx = (xend - x0) / (K - 1)
    dx2 = dx ** 2

    yarr = np.full_like(xarr, np.nan) # y evaluations at gridpoints
    phiarr = np.full_like(xarr, np.nan) # phi evaluations at gridpoints
    farr = [f(x) for x in xarr][::-1]

    yarr[0] = y0
    yarr[1] = y1    

    phi = lambda k: 0 if np.isinf(farr[k]) else yarr[k] * (1 - dx2 * farr[k] / 12)
    phiarr[0] = phi(0)
    phiarr[1] = phi(1)

    for k in range(2, K):
        phiarr[k] = 2 * phiarr[k - 1] - phiarr[k - 2] + dx2 * farr[k - 1] * yarr[k - 1]
        yarr[k] = phiarr[k] / (1 - dx2 * farr[k] / 12)

    return xarr, yarr[::-1]
