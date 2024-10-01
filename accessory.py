import soundfile as sf
import matplotlib.pyplot as plt
import wave
import struct
import numpy as np


def read_wav_file():
    data, fe = sf.read("./note_guitare_LAd.wav")
    return data, fe


def plot_waveform(y, title, x=None):
    if x is None:
        plt.plot(y)
    else:
        plt.plot(x, y)
    plt.title(title)
    plt.grid()
    plt.show()


def create_wav_file(audio, sampleRate, filename):
    audio = np.int16(audio / np.max(np.abs(audio)) * 32767)  # normalize to 16-bit PCM

    with wave.open(filename, "w") as wav:
        nchannels = 1
        sampwidth = 2
        nframes = len(audio)
        wav.setparams(
            (nchannels, sampwidth, sampleRate, nframes, "NONE", "not compressed")
        )

        for sample in audio:
            wav.writeframes(struct.pack("h", sample))


def plot_frequential_waveform(harm_freqs, harmonics, phases):

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


def trouver_ordre_filtre_passe_bas(w):
    for n in range(1, 2000):
        sum = np.sum(np.exp(-1j * w * np.arange(n)))
        gain = np.abs(sum) * 1 / n
        if gain <= 10 ** (-3 / 20):
            N = n
            break
    return N
