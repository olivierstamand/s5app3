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
X = np.fft.fft(data)
frequences = np.fft.fftfreq(len(data),d=1/fe)

plt.plot(frequences, 20 * np.log10(np.abs(X)))
plt.xlim(0,1500)
plt.title("Spectres de fourier avant filtrage")
plt.xlabel("fréquence (Hz)")
plt.ylabel("Amplitude (dB)")
plt.show()

f_0 = 1000
f_1 = 40
w_0 = 2 * np.pi * f_0 / fe
w_1 = 2 * np.pi * f_1 / fe
N = 6000
n = np.linspace(-(N / 2) + 1, N / 2, N)
m = math.ceil(f_1 * N / fe)
K = 2 * m + 1

index = np.linspace(-(N / 2) + 1, (N / 2), N)
# filtre_coupe_bande = passebas_a_coupebande(
#     h=passe_bas(n=index, N=N, K=K), n=index, omega_0=w_0
# )
filtre_coupe_bande = [coupe_bande(n, N, K, w_0) for n in index]
tracer_forme_onde(y=filtre_coupe_bande, titre="Filtre coupe-bande)")
bidon = np.fft.fft(filtre_coupe_bande)
bidonfreq = np.fft.fftfreq(len(filtre_coupe_bande),d=1/fe) 

plt.plot(bidonfreq, 20 * np.log10(np.abs(bidon))) 
plt.show()

window = np.hanning(N)
filtre_coupe_bande_fenetre = filtre_coupe_bande * window
tracer_forme_onde(y=filtre_coupe_bande_fenetre, titre="Filtre coupe-bande fenêtré")

audio = np.convolve(filtre_coupe_bande_fenetre, data)
tracer_forme_onde(y=audio, titre="Forme du fichier audio filtré")
X = np.fft.fft(audio)
frequences = np.fft.fftfreq(len(audio),d=1/fe)

plt.plot(frequences, 20 * np.log10(np.abs(X)))
plt.xlim(0,1500)
plt.title("Spectres de fourier avant filtrage")
plt.xlabel("fréquence (Hz)")
plt.ylabel("Amplitude (dB)")
plt.show()

creer_wav_audio(audio, fe, "basson_filtre.wav")
