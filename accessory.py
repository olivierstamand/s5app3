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
