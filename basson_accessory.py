import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


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


def plot_waveform_avec_unites(data, fe):
    duration = len(data) / fe
    time = np.linspace(0, duration, len(data))
    # Création du graphique
    plt.figure(figsize=(10, 4))
    plt.plot(time, data)
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.title("Signal Audio")
    plt.grid(True)
    plt.show()


def plot_spectrum(signal, fe, title):
    N = len(signal)
    spectrum = np.fft.fft(signal)
    amplitude = np.abs(spectrum)[
        : N // 2
    ]  # Prendre la moitié du spectre (valeurs réelles)
    freqs = np.fft.fftfreq(N, d=1 / fe)[: N // 2]  # Fréquences correspondantes

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, amplitude)
    plt.title(title)
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
