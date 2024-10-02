import numpy as np
import matplotlib.pyplot as plt
import math
import soundfile as sf
import wave
import struct
from accessory import *
from basson_accessory import *


data, fe = sf.read("note_basson_plus_sinus_1000_hz.wav")
plot_waveform_avec_unites(data=data, fe=fe)
# plot_waveform(y=data, title="Forme du fichier audio original")


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
plot_waveform(y=filtre_coupe_bande, title="Filtre coupe-bande)")


window = np.hanning(N)
filtre_coupe_bande_fenetre = filtre_coupe_bande * window
plot_waveform(y=filtre_coupe_bande_fenetre, title="Filtre coupe-bande fenêtré")

audio = np.convolve(filtre_coupe_bande_fenetre, data)
plot_waveform(y=audio, title="Forme du fichier audio filtré")


# Pour le rapport:
plot_spectrum(data, fe, title="Spectre d'amplitude - Basson avant filtrage")
create_wav_file(audio, fe, "basson_filtre.wav")


# Signal avant filtrage (basson original)

# Signal après filtrage
plot_spectrum(audio, fe, title="Spectre d'amplitude - Basson après filtrage")
