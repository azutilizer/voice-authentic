from os import listdir
from os.path import exists
import pickle
from scipy.io import wavfile
import audioPlayer as player
import numpy as np
from utilities import get_one_data, convert

POWER_SPECTRUM_FLOOR = 1e-100


def play_wave(wave_file='', status=True):
    if not exists(wave_file):
        return
    player.audio_play(wave_file, status)


def read_audio(audio_path):
    if not exists(audio_path):
        return -1
    fs, data = wavfile.read(audio_path)  # load the data
    # plt.plot(data, 'r')
    # plt.show()
    dims = data.shape
    if len(dims) > 1 or fs != 16000:
        _, data = convert(audio_path)
    return data


def save_speaker_model(model_path, speaker_model):
    with open(model_path, "wb") as f:
        pickle.dump(speaker_model, f)


def load_speaker_model(model_path):
    if not exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def recognize(model, voice_file_path):

    feat = get_one_data(voice_file_path)
    feat = np.array([feat])
    voice = model.predict(feat)
    score = model.predict_proba(feat)
    score = max(score[0])
    """
    if score < 0.9:
        voice = ["NOT AUTHENTIC"]
    else:
        voice = ["AUTHENTIC"]
    """
    return voice[0], score


def recognize_DNN(model, voice_file_path):
    feat = get_one_data(voice_file_path)
    feat = np.array([feat])
    scores = model.predict(feat)
    score = max(scores[0])
    max_ind = list(scores[0]).index(score)

    class_labels = []
    for sub_folder in listdir("voice"):
        class_labels.append(sub_folder)
    voice = class_labels[max_ind]
    return voice, score
