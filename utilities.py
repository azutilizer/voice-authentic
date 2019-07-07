import numpy as np
import scipy.io.wavfile as wav
import os
import librosa
import speechpy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

mslen = 80000  # at least, 5 seconds


def read_wav(filename):
    audio_data = read_audio_librosa(filename)
    wav_data = audio_data[0]
    sample_rate = audio_data[1]

    frame_len = int(0.25 * sample_rate)
    frame_step = int(0.01 * sample_rate)
    vad_interval = librosa.effects.split(y=wav_data, top_db=30, frame_length=frame_len, hop_length=frame_step)

    if len(vad_interval) == 0:
        print("can't extract speech from {}".format(filename))
        return -1, -1
    flatten_vad_index = []
    for x in vad_interval:
        flatten_vad_index.extend(range(x[0], x[1]))
    flatten_vad_index = np.array(flatten_vad_index)
    wav_data = np.take(wav_data, flatten_vad_index)

    return sample_rate, wav_data


def get_data(flatten=True, mfcc_len=39):
    data = []
    labels = []
    max_fs = 0
    min_sample = int('9' * 10)
    s = 0
    cnt = 0

    dataset_folder = "voice/"
    class_labels = []
    for class_dir in os.listdir(dataset_folder):
        if os.path.isdir(os.path.join(dataset_folder, class_dir)):
            class_labels.append(str(class_dir))

    cur_dir = os.getcwd()
    # os.chdir('..')
    os.chdir(dataset_folder)
    for i, directory in enumerate(class_labels):
        print("started reading folder", directory)
        os.chdir(directory)
        for filename in os.listdir('.'):
            fs, signal = read_wav(filename)
            if fs == -1:
                continue
            dims = signal.shape
            if len(dims) > 1 or fs != 16000:
                fs, signal = convert(filename)
            max_fs = max(max_fs, fs)
            s_len = len(signal)
            # pad the signals to have same size if lesser than required
            # else slice them
            if s_len < mslen:
                pad_len = mslen - s_len
                pad_rem = pad_len % 2
                pad_len //= 2
                signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
            else:
                pad_len = s_len - mslen
                pad_rem = pad_len % 2
                pad_len //= 2
                signal = signal[pad_len:pad_len + mslen]
            min_sample = min(len(signal), min_sample)
            mfcc = speechpy.feature.mfcc(signal, fs, num_cepstral=mfcc_len)

            if flatten:
                # Flatten the data
                mfcc = mfcc.flatten()
            data.append(mfcc)
            labels.append(directory)
            cnt += 1
        print("ended reading folder", directory)
        os.chdir('..')
    os.chdir(cur_dir)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


def display_metrics(y_pred, y_true):
    print(accuracy_score(y_pred=y_pred, y_true=y_true))
    print(confusion_matrix(y_pred=y_pred, y_true=y_true))


def get_one_data(filename):
    max_fs = 0
    mfcc_len = 39
    min_sample = int('9' * 10)
    fs, signal = read_wav(filename)
    if fs == -1:
        return np.array([])
    dims = signal.shape
    if len(dims) > 1 or fs != 16000:
        fs, signal = convert(filename)
    max_fs = max(max_fs, fs)
    s_len = len(signal)
    # pad the signals to have same size if lesser than required
    # else slice them
    if s_len < mslen:
        pad_len = mslen - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
    else:
        pad_len = s_len - mslen
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = signal[pad_len:pad_len + mslen]
    min_sample = min(len(signal), min_sample)
    mfcc = speechpy.feature.mfcc(signal, fs, num_cepstral=mfcc_len)

    mfcc = mfcc.flatten()

    return np.array(mfcc)


def convert(file_name):
    tmp_file = 'tmp.wav'
    conv_command = "ffmpeg -i \"{}\" -acodec pcm_s16le -ar 16000 -ac 1 -y -loglevel panic {}".format(file_name, tmp_file)
    os.system(conv_command)

    if not os.path.exists(tmp_file):
        print("convert failed!")
        return -1, -1
    fs, signal = read_wav(tmp_file)
    os.remove(tmp_file)
    return 16000, signal


def read_audio_librosa(wav_path):
    return librosa.load(wav_path, sr=16000, )
