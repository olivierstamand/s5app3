import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# 1. Charger le fichier .wav
samplerate, data = wavfile.read('/home/olivier/Downloads/note_guitare_lad.wav')

# Si le fichier audio est stéréo, prendre une seule des deux pistes
if len(data.shape) > 1:
    data = data[:, 0]  # Prendre uniquement la première piste (mono)

# 2. Appliquer la fenêtre de Hanning
N = len(data)  # Nombre d'échantillons
window = np.hanning(N)  # Créer la fenêtre de Hanning
data_windowed = data * window  # Appliquer la fenêtre sur le signal

# 3. Appliquer la FFT avec NumPy
T = 1.0 / samplerate  # Intervalle de temps entre les échantillons
frequencies = np.fft.fftfreq(N, T)  # Fréquences correspondant à la FFT
fft_values = np.fft.fft(data_windowed)  # Calcul de la FFT après application de la fenêtre





# # 4. Calculer l'amplitude en dB
# amplitude_spectrum = np.abs(fft_values[:N // 2])  # Amplitude linéaire
# amplitude_spectrum_db = 20 * np.log10(amplitude_spectrum + 1e-6)  # Conversion en dB (petite constante pour éviter log(0))

# # 5. Afficher le spectre en dB
# plt.figure(figsize=(10, 6))
# plt.plot(frequencies[:N // 2], amplitude_spectrum_db)
# plt.title('Spectre de fréquence avec fenêtre de Hanning (en dB)')
# plt.xlabel('Fréquence (Hz)')
# plt.ylabel('Amplitude (dB)')
# plt.grid()
# plt.show()
