import numpy as np
import matplotlib.pyplot as plt
import math
import soundfile as sf
import wave
import struct
from accessory import *
from basson_accessory import *


data, fe = sf.read("note_basson_plus_sinus_1000_hz.wav")
tracer_forme_onde(y=data, titre="Forme du fichier audio original")


f_0 = 1000
f_1 = 40
w_0 = 2 * np.pi * f_0 / fe
w_1 = 2 * np.pi * f_1 / fe
N = 6000
n = np.linspace(-(N / 2) + 1, N / 2, N)
m = math.ceil(f_1 * N / fe)
K = 2 * m + 1

index = np.linspace(-(N / 2) + 1, (N / 2), N)
filtre_coupe_bande = [coupe_bande(n, N, K, w_0) for n in index]
plt.figure(figsize=(10, 6))
plt.plot(index, filtre_coupe_bande)
plt.title("Filtre Coupe-Bande", fontsize=14)
plt.xlabel("Indice", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.grid(True)
plt.show()

coupe_bande_temporel = np.fft.ifft(filtre_coupe_bande)
# (iii) Réponse à une sinusoïde de 1000 Hz
t = np.arange(0, 1, 1 / fe)  # durée de 1 seconde
sinusoide_1000Hz = np.sin(2 * np.pi * 1000 * t)
reponse_sinus_1000Hz = np.convolve(sinusoide_1000Hz, coupe_bande_temporel, mode="same")

plt.figure(figsize=(10, 6))
plt.plot(t, reponse_sinus_1000Hz)
plt.title("Réponse du filtre à une sinusoïde de 1000 Hz", fontsize=14)
plt.xlabel("Temps (s)", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.grid(True)
plt.show()

# (iv) Réponse en fréquence : amplitude et phase
freqs = np.fft.fftfreq(N, d=1 / fe)
reponse_freq = np.fft.fft(filtre_coupe_bande)

# Amplitude
# plt.figure(figsize=(10, 6))
# plt.plot(freqs[: N // 2], np.abs(reponse_freq[: N // 2]))
# plt.title("Amplitude de la réponse en fréquence", fontsize=14)
# plt.xlabel("Fréquence (Hz)", fontsize=12)
# plt.ylabel("Amplitude", fontsize=12)
# plt.grid(True)
# plt.show()

# # Phase
# plt.figure(figsize=(10, 6))
# plt.stem(freqs[: N // 2], np.angle(reponse_freq[: N // 2]))
# plt.title("Phase de la réponse en fréquence", fontsize=14)
# plt.xlabel("Fréquence (Hz)", fontsize=12)
# plt.ylabel("Phase (radians)", fontsize=12)
# plt.grid(True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# (iv) Réponse en fréquence : amplitude (en dB) et phase
freqs = np.fft.fftfreq(N, d=1 / fe)
reponse_freq = np.fft.fft(filtre_coupe_bande)

# Amplitude en dB
amplitude_db = 20 * np.log10(np.abs(reponse_freq[: N // 2]))

plt.figure(figsize=(10, 6))
plt.plot(freqs[: N // 2], amplitude_db)
plt.title("Amplitude de la réponse en fréquence (en dB)", fontsize=14)
plt.xlabel("Fréquence (Hz)", fontsize=12)
plt.ylabel("Amplitude (dB)", fontsize=12)
plt.grid(True)
plt.show()

# Phase
plt.figure(figsize=(10, 6))
plt.stem(freqs[: N // 2], np.angle(reponse_freq[: N // 2]))
plt.title("Phase de la réponse en fréquence", fontsize=14)
plt.xlabel("Fréquence (Hz)", fontsize=12)
plt.ylabel("Phase (radians)", fontsize=12)
plt.grid(True)
plt.show()
