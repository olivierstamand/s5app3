import numpy as np


def passebas_a_coupebande(h, n, omega_0):
    delta = np.zeros_like(h)
    delta[n == 0] = 1
    h_bandcut = delta - 2 * h * np.cos(omega_0 * n)

    return h_bandcut


def passe_bas(n, N, K):
    hn = np.sin(np.pi * n * K / N) / (N * np.sin(np.pi * n / N))
    return hn


def coupe_bande(n, N, K, w0):
    cb = 1 - 2 * K / N
    if n != 0:
        cb = -2 * passe_bas(n, N, K) * np.cos(w0 * n)
    return cb
