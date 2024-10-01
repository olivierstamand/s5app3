import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from accessory import *


# Signal initial temporel
data, fe = sf.read("note_guitare_LAd.wav")
print(f"Sample rate: {fe}")
N = len(data)
print(N)
plot_waveform(y=data, title="Forme du fichier audio original")


# Signal temporel fenêtré
window = np.hanning(N)
data_fenetre = np.multiply(data, window)
data_fenetre = np.divide(
    data_fenetre, np.max(data_fenetre)
)  # Normalisation ca a l'air mais je comprends pas pourquoi
plot_waveform(y=data_fenetre, title="Forme du fichier audio fenêtré")

# Signal fréquentiel (FFT)
# NOTE: passer dans le domaine fréquentiel permet de trouver l'ordre du filtre passe-bas
w = np.pi / 1000
Nb_sinusoids = 32
X = np.fft.fft(data_fenetre)
freqs = np.fft.fftfreq(N) * fe
plot_waveform(y=X, title="Spectre de fréquence du signal fenêtré")

# # Find the 32 largest sinusoids (by magnitude)
# X_magnitudes = np.abs(X)
# indices_largest = np.argpartition(X_magnitudes, -Nb_sinusoids)[-Nb_sinusoids:]

# # Create a new frequency domain representation with only the 32 largest sinusoids
# X_reconstructed = np.zeros_like(X, dtype=complex)
# X_reconstructed[indices_largest] = X[indices_largest]
# plt.plot(freqs, np.abs(X_reconstructed))
# plt.title("Frequency Domain Signal (32 Largest Sinusoids)")
# plt.show()
# A essayer ?

N_passe_bas = trouver_ordre_filtre_passe_bas(w=w)

# Enveloppe du signal initial
coeff = np.ones(N_passe_bas) / N_passe_bas
enveloppe = np.convolve(coeff, np.abs(data), mode="same")
enveloppe = np.divide(enveloppe, np.max(enveloppe))
plot_waveform(y=enveloppe, title="Enveloppe du signal initial")

# Reconstruire le signal à partir des 32 plus grandes harmoniques
index_lad = np.argmax(abs(X))
fundamental = freqs[index_lad]
index_harms = [index_lad * i for i in range(0, Nb_sinusoids + 1)]
harm_freqs = [freqs[i] for i in index_harms]
harmonics = [np.abs(X[i]) for i in index_harms]
phases = [np.angle(X[i]) for i in index_harms]
note_audio = create_audio(harmonics, phases, fundamental, fe, enveloppe, 2)
plot_waveform(y=note_audio, title="Forme du fichier audio synthétisé")

create_wav_from_audio(audio=note_audio, sampleRate=fe, filename="la#_synthetise.wav")
