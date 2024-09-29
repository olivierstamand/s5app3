import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import math



import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

data, fe = sf.read('./note_guitare_LAd.wav')
N = len(data)
window = np.hanning(N)
data_windowed = window * data
Nb_sinusoids = 32

X = np.fft.fft(data_windowed)
freqs = np.fft.fftfreq(N) * fe  #

index_lad = np.argmax(abs(X))
fundamental = freqs[index_lad]

# Get amplitudes at harmonics
index_harms = [index_lad * i for i in range(0, Nb_sinusoids + 1)]
harm_freqs = [freqs[i] for i in index_harms]
harmonics = [np.abs(X[i]) for i in index_harms]
phases = [np.angle(X[i]) for i in index_harms]

fig, (harm, phas) = plt.subplots(2)


harm.stem(harm_freqs, harmonics)
harm.set_yscale("log")
harm.set_title("Amplitude des harmoniques")
harm.set_xlabel("Fréquence (Hz)")
harm.set_ylabel("Amplitude")
phas.stem(harm_freqs, phases)
phas.set_title("Phase des harmoniques")
phas.set_xlabel("Fréquence (Hz)")
phas.set_ylabel("Amplitude")

plt.show()


amplitude= np.sum(harmonics)
phase = np.sum(phases)

signal = np.sin(2*np.pi*amplitude+phase)


w = np.pi/1000
for n in range (0,2000): 
    sum = np.sum(np.exp(-1j*w*np.arange(n)))
    gain = np.abs(sum) * 1/n 
    if gain <= 10 ** (-3 / 20):  
        N=n 
        break 

coeff= np.ones(N)/N
enveloppe=np.convolve(coeff,np.abs(data))
plt.plot(enveloppe)
plt.show()

synth = enveloppe * signal

plt.plot(synth)
plt.show()


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
