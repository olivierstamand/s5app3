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


window_filtre_coupe_bande = np.hamming(N) * filtre_coupe_bande
plt.figure(figsize=(10, 6))
plt.plot(index, filtre_coupe_bande)
plt.title("Filtre Coupe-Bande adouci", fontsize=14)
plt.xlabel("Indice", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.grid(True)
plt.show()


# Générer une sinusoïde de 1000 Hz
t = np.arange(0, N) / fe
sinus_1000_hz = np.sin(2 * np.pi * f_0 * t)
filtered_signal = np.convolve(sinus_1000_hz, window_filtre_coupe_bande, mode="same")
plt.figure()
plt.plot(t, filtered_signal)
plt.title("Réponse à une sinusoïde de 1000 Hz après filtrage")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()


# Calcul de la réponse en fréquence
H_f = np.fft.fft(filtre_coupe_bande, N)
H_f_shifted = np.fft.fftshift(H_f)
frequencies = np.fft.fftfreq(N, 1 / fe)
frequencies_shifted = np.fft.fftshift(frequencies)

# Tracer la magnitude
plt.figure()
plt.plot(frequencies_shifted, 20 * np.log10(np.abs(H_f_shifted)))
plt.title("Réponse en amplitude du filtre")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude (dB)")
plt.grid()
plt.show()

# Tracer la phase
plt.figure()
plt.plot(frequencies_shifted, np.angle(H_f_shifted))
plt.title("Réponse en phase du filtre")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Phase (rad)")
plt.grid()
plt.show()
