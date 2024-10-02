import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from accessory import *
import pandas as pd


# Signal initial temporel
data, fe = sf.read("note_guitare_LAd.wav")
print(f"Taux d'échantillonnage : {fe}")
N = len(data)
print(f"Nombre d'échantillons : {N}")
tracer_forme_onde(y=data, titre="Forme du fichier audio original")

# Signal temporel fenêtré
fenetre = np.hanning(N)
data_fenetre = np.multiply(data, fenetre)
data_fenetre = np.divide(data_fenetre, np.max(data_fenetre))  # Normalisation
tracer_forme_onde(y=data_fenetre, titre="Forme du fichier audio fenêtré")

# Signal fréquentiel (FFT)
# Passer dans le domaine fréquentiel permet de trouver l'ordre du filtre passe-bas
w = np.pi / 1000
Nb_sinusoides = 32
X = np.fft.fft(data_fenetre)
frequences = np.fft.fftfreq(len(data_fenetre),d=1/fe)
tracer_forme_onde(x=frequences, y=20 * np.log10(np.abs(X)), titre="Spectre de fréquence du la#",limit=8000,db =True)

# Trouver l'ordre du filtre passe-bas
N_passe_bas = trouver_ordre_filtre_passe_bas(w=w)

# Enveloppe du signal initial
coeff = np.ones(N_passe_bas) / N_passe_bas

response = np.fft.fft(coeff)
frequences = np.fft.fftfreq(len(coeff),d=1/fe)
tracer_forme_onde(x=frequences, y=np.abs((response)), titre="Réponse en fréquence du filtre passe-bas",db=True,limit=100)

enveloppe = np.convolve(coeff, np.abs(data), mode="same")
enveloppe = np.divide(enveloppe, np.max(enveloppe))
tracer_forme_onde(y=enveloppe, titre="Enveloppe du signal initial")

# Reconstruire le signal à partir des 32 plus grandes harmoniques
index_lad = np.argmax(abs(X))
fondamentale = frequences[index_lad]
index_harmoniques = [index_lad * i for i in range(0, Nb_sinusoides + 1)]
freq_harmoniques = [frequences[i] for i in index_harmoniques]
harmoniques = [np.abs(X[i]) for i in index_harmoniques]
phases = [np.angle(X[i]) for i in index_harmoniques]

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
tracer_forme_onde_frequentielle(freq_harmoniques, harmoniques, phases)

# Création du fichier audio à partir des harmoniques
note_audio = creer_audio(harmoniques, phases, fondamentale, fe, enveloppe, 2)
tracer_forme_onde(y=note_audio, titre="Forme du fichier audio synthétisé")
X = np.fft.fft(note_audio)
frequences = np.fft.fftfreq(len(note_audio),d=1/fe)

la_freq = fondamentale
tracer_forme_onde(x= frequences,y=20 * np.log10(np.abs(X)),  titre="Spectre de fréquence du la# synthétisé",db=True)

creer_wav_audio(audio=note_audio, taux_echantillonnage=fe, nom_fichier="la#_synthetise.wav")

# Dictionnaire des fréquences pour les autres notes
frequences_notes = {
    "do": la_freq * 0.595,
    "do#": la_freq * 0.630,
    "ré": la_freq * 0.667,
    "ré#": la_freq * 0.707,
    "mi": la_freq * 0.749,
    "fa": la_freq * 0.794,
    "fa#": la_freq * 0.841,
    "sol": la_freq * 0.891,
    "sol#": la_freq * 0.944,
    "la": la_freq,
    "la#": la_freq,
    "si": la_freq * 1.123,
}

# Création des différentes notes et silences
sol_audio = creer_audio(harmoniques, phases, frequences_notes["sol"], fe, enveloppe, 0.4)
mib_audio = creer_audio(harmoniques, phases, frequences_notes["ré#"], fe, enveloppe, 1.5)
fa_audio = creer_audio(harmoniques, phases, frequences_notes["fa"], fe, enveloppe, 0.4)
re_audio = creer_audio(harmoniques, phases, frequences_notes["ré"], fe, enveloppe, 1.5)

silence_1 = creer_silence(fe, 0.2)
silence_2 = creer_silence(fe, 1.5)

# Construction de la mélodie
beethoven = (
    sol_audio
    + silence_1
    + sol_audio
    + silence_1
    + sol_audio
    + silence_1
    + mib_audio
    + silence_2
    + fa_audio
    + silence_1
    + fa_audio
    + silence_1
    + fa_audio
    + silence_1
    + re_audio
)
X = np.fft.fft(beethoven)
frequences = np.fft.fftfreq(len(beethoven),d=1/fe)

la_freq = fondamentale
tracer_forme_onde(x= frequences,y=20 * np.log10(np.abs(X)),  titre="Spectre de fréquence de symphonie Beethoven",db=True)
# Création du fichier audio final
creer_wav_audio(beethoven, fe, "beethoven.wav")

