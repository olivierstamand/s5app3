import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import wave
import struct


def read_file(filename):
    with wave.open(filename, "rb") as wav:
        # Extract info from wave file
        sample_rate = wav.getframerate()
        frames = wav.readframes(-1)
        frames = np.frombuffer(frames, dtype=np.int16)

        # Normalize at 1
        max_amp = np.amax(frames)
        frames = np.divide(frames, max_amp)

        return sample_rate, frames


def pad_thai(array, length):
    return np.pad(array, (0, length - len(array)))


def unpad_thai(array, length):
    return array[0:length]


def create_wav_from_audio(audio, sampleRate, filename):
    with wave.open(filename, "w") as wav:
        nchannels = 1
        sampwidth = 2
        nframes = len(audio)
        wav.setparams(
            (nchannels, sampwidth, sampleRate, nframes, "NONE", "not compressed")
        )

        for sample in audio:
            wav.writeframes(struct.pack("h", int(sample)))


def create_audio(harmonics, phases, fundamental, sampleRate, enveloppe, duration_s=2):
    audio = []
    ts = np.linspace(0, duration_s, int(sampleRate * duration_s))

    audio = []
    for t in ts:
        total = 0
        for i in range(len(harmonics)):
            total += harmonics[i] * np.sin(2 * np.pi * fundamental * i * t + phases[i])

        audio.append(total)
    new_env = unpad_thai(enveloppe, len(audio))
    new_audio = pad_thai(audio, len(new_env))
    audio = np.multiply(new_audio, new_env)
    return audio.tolist()


# Signal initial temporel
fe, data = read_file("./note_guitare_LAd.wav")
# fe, data = read_file("note_guitare_LAd.wav")
# fe = 160000
print(f"Sample rate: {fe}")
N = len(data)
print(N)
plt.plot(np.arange(N), data)
plt.title("Original Time Domain Signal")
plt.show()


# Signal temporel fenêtré
window = np.hanning(N)
plt.plot(np.arange(N), window)
plt.show()
data_windowed = window * data
plt.plot(np.arange(N), data_windowed)
plt.title("Windowed Time Domain Signal")
plt.show()

Nb_sinusoids = 32
# Signal fréquentiel (FFT)
X = np.fft.fft(data_windowed)
freqs = np.fft.fftfreq(N) * fe
plt.plot(freqs, np.abs(X))
plt.title("Original Frequency Domain Signal")
plt.show()

# # Find the 32 largest sinusoids (by magnitude)
# X_magnitudes = np.abs(X)
# indices_largest = np.argpartition(X_magnitudes, -Nb_sinusoids)[-Nb_sinusoids:]

# # Create a new frequency domain representation with only the 32 largest sinusoids
# X_reconstructed = np.zeros_like(X, dtype=complex)
# X_reconstructed[indices_largest] = X[indices_largest]
# plt.plot(freqs, np.abs(X_reconstructed))
# plt.title("Frequency Domain Signal (32 Largest Sinusoids)")
# plt.show()


w = np.pi / 1000
for n in range(1, 2000):
    sum = np.sum(np.exp(-1j * w * np.arange(n)))
    gain = np.abs(sum) * 1 / n
    if gain <= 10 ** (-3 / 20):
        N = n
        break

# Enveloppe du signal initial
coeff = np.ones(N) / N
enveloppe = np.convolve(coeff, np.abs(data), mode="same")
plt.plot(np.arange(len(data)), enveloppe)
plt.show()


index_lad = np.argmax(abs(X))
fundamental = freqs[index_lad]
# print(index_lad)
print("La# fundamental frequency: " + str(fundamental))

# Get amplitudes at harmonics
index_harms = [index_lad * i for i in range(0, Nb_sinusoids + 1)]
harm_freqs = [freqs[i] for i in index_harms]
harmonics = [np.abs(data[i]) for i in index_harms]
phases = [np.angle(data[i]) for i in index_harms]
note_audio = create_audio(harmonics, phases, fundamental, fe, enveloppe, 2)
plt.plot(note_audio)
plt.show()
# create_wav_from_audio(note_audio, fe, "bidon.wav")

sf.write("bidon.wav", note_audio, fe)

# X_reconstructed_time = np.fft.ifft(X_reconstructed)
# synth = enveloppe * X_reconstructed_time
# plt.plot(synth)
# plt.show()
# synth = np.real(synth)
# print("Bidon")
# sf.write("synth.wav", synth, fe)


# # # # 4. Calculer l'amplitude en dB
# # # amplitude_spectrum = np.abs(fft_values[:N // 2])  # Amplitude linéaire
# # # amplitude_spectrum_db = 20 * np.log10(amplitude_spectrum + 1e-6)  # Conversion en dB (petite constante pour éviter log(0))

# # # # 5. Afficher le spectre en dB
# # # plt.figure(figsize=(10, 6))
# # # plt.plot(frequencies[:N // 2], amplitude_spectrum_db)
# # # plt.title('Spectre de fréquence avec fenêtre de Hanning (en dB)')
# # # plt.xlabel('Fréquence (Hz)')
# # # plt.ylabel('Amplitude (dB)')
# # # plt.grid()
# # # plt.show()


# with wave.open("testla#.wav", "w") as wav:
#     nchannels = 1
#     sampwidth = 2
#     nframes = len(X_reconstructed)
#     # nframes = len(audio)
#     wav.setparams((nchannels, sampwidth, fe, nframes, "NONE", "not compressed"))

#     for sample in X_reconstructed:
#         wav.writeframes(struct.pack("h", np.int16(sample)))
