import soundfile as sf
import matplotlib.pyplot as plt
import wave
import struct
import numpy as np

def lire_fichier_wav():
    data, fe = sf.read("./note_guitare_LAd.wav")
    return data, fe

def tracer_forme_onde(y, titre, x=None,limit=160000,db=False):
    if x is None:
        plt.plot(y)
    else:
        plt.plot(x, y)
    if db:
        plt.ylabel("Amplitude (Db)")
        plt.xlabel("Fréquence (Hz)")
    else:
        plt.ylabel("Amplitude normalisé")
        plt.xlabel("Echantillons/rad")
    plt.xlim(0, limit)
    plt.title(titre)
    plt.grid()
    plt.show()


def tracer_forme_onde_frequentielle(freq_harmoniques, harmoniques, phases):
    fig, (harm, phas) = plt.subplots(2)
    harm.stem(freq_harmoniques, harmoniques)
    harm.set_yscale("log")
    harm.set_title("Amplitude des harmoniques")
    harm.set_xlabel("Fréquence (Hz)")
    harm.set_ylabel("Amplitude")
    phas.stem(freq_harmoniques, phases)
    phas.set_title("Phase des harmoniques")
    phas.set_xlabel("Fréquence (Hz)")
    phas.set_ylabel("Amplitude")
    plt.show()

def lire_fichier(nom_fichier):
    with wave.open(nom_fichier, "rb") as wav:
        taux_echantillonnage = wav.getframerate()
        frames = wav.readframes(-1)
        frames = np.frombuffer(frames, dtype=np.int16)

        # Normaliser à 1
        amplitude_max = np.amax(frames)
        frames = np.divide(frames, amplitude_max)

        return taux_echantillonnage, frames

def pad_thai(array, longueur):
    return np.pad(array, (0, longueur - len(array)))

def unpad_thai(array, longueur):
    return array[0:longueur]

def creer_wav_audio(audio, taux_echantillonnage, nom_fichier):
    with wave.open(nom_fichier, "w") as wav:
        nchannels = 1
        sampwidth = 2
        nframes = len(audio)
        wav.setparams(
            (nchannels, sampwidth, taux_echantillonnage, nframes, "NONE", "not compressed")
        )

        for sample in audio:
            wav.writeframes(struct.pack("h", int(sample)))

def creer_audio(harmoniques, phases, fondamentale, taux_echantillonnage, enveloppe, duree_s=2):
    audio = []
    ts = np.linspace(0, duree_s, int(taux_echantillonnage * duree_s))

    audio = []
    for t in ts:
        total = 0
        for i in range(len(harmoniques)):
            total += harmoniques[i] * np.sin(2 * np.pi * fondamentale * i * t + phases[i])

        audio.append(total)
    nouvelle_enveloppe = unpad_thai(enveloppe, len(audio))
    nouvel_audio = pad_thai(audio, len(nouvelle_enveloppe))
    audio = np.multiply(nouvel_audio, nouvelle_enveloppe)
    return audio.tolist()

def trouver_ordre_filtre_passe_bas(w):
    for n in range(1, 2000):
        
        somme = np.sum(np.exp(-1j * w * np.arange(n)))
        gain = np.abs(somme) * 1 / n
        if gain <= 10 ** (-3 / 20):
            N = n
            break
    return N

def creer_silence(taux_echantillonnage, duree_s = 1):
    return [0 for t in np.linspace(0, duree_s , int(taux_echantillonnage * duree_s))]
