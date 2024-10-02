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
plt.title("Spectres de fourier après filtrage")
plt.xlabel("fréquence (Hz)")
plt.ylabel("Amplitude (dB)")
plt.show()

creer_wav_audio(audio, fe, "basson_filtre.wav")

data = audio
# Signal temporel fenêtré
fenetre = np.hanning(len(data))
data_fenetre = np.multiply(data, fenetre)
data_fenetre = np.divide(data_fenetre, np.max(data_fenetre))  # Normalisation
tracer_forme_onde(y=data_fenetre, titre="Forme du fichier audio fenêtré")

# Signal fréquentiel (FFT)
# Passer dans le domaine fréquentiel permet de trouver l'ordre du filtre passe-bas
w = np.pi / 1000
Nb_sinusoides = 32

# Trouver l'ordre du filtre passe-bas
N_passe_bas = trouver_ordre_filtre_passe_bas(w=w)

# Enveloppe du signal initial
coeff = np.ones(N_passe_bas) / N_passe_bas

enveloppe = np.convolve(coeff, np.abs(data), mode="same")
enveloppe = np.divide(enveloppe, np.max(enveloppe))
tracer_forme_onde(y=enveloppe, titre="Enveloppe du signal initial")
X = np.fft.fft(data_fenetre)
frequences = np.fft.fftfreq(len(data_fenetre),d=1/fe)

index_lad = np.argmax(abs(X))
fondamentale = frequences[index_lad]
index_harmoniques = [index_lad * i for i in range(0, Nb_sinusoides + 1)]
freq_harmoniques = [frequences[i] for i in index_harmoniques]
harmoniques = [np.abs(X[i]) for i in index_harmoniques]
phases = [np.angle(X[i]) for i in index_harmoniques]

# Création du fichier audio à partir des harmoniques
note_audio = creer_audio(harmoniques, phases, fondamentale, fe, enveloppe, 2)
tracer_forme_onde(y=note_audio, titre="Forme du fichier audio synthétisé")
X = np.fft.fft(note_audio)
frequences = np.fft.fftfreq(len(note_audio),d=1/fe)

tracer_forme_onde(x= frequences,y=20 * np.log10(np.abs(X)),  titre="Spectre de fréquence du basson synthétisé",db=True)

creer_wav_audio(note_audio, fe, "basson_synth.wav")
fig, ax = plt.subplots()
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.axis("tight")
cellText = []
for i in range(len(harmoniques)):
    cellText.append([freq_harmoniques[i], 20*np.log10(harmoniques[i]), phases[i]])
table = ax.table(cellText = cellText, colLabels = ["Fréquence (Hz)", "Amplitude", "Phase"], loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 0.5)

plt.show()

# Pour le rapport:
plot_spectrum(data, fe, title="Spectre d'amplitude - Basson avant filtrage")
creer_wav_audio(audio, fe, "basson_filtre.wav")


# Signal avant filtrage (basson original)

# Signal après filtrage
plot_spectrum(audio, fe, title="Spectre d'amplitude - Basson après filtrage")
