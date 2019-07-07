import os
import numpy as np
import librosa
import audioPlayer as player
import tempfile
import queue
import sys
import array
import struct
import sounddevice as sd
import soundfile as sf
import wave
from pysndfx import AudioEffectsChain


SOX_BIN = os.path.join('sox', 'sox')
FFMPEG_BIN = 'ffmpeg'
rec_status = False
rec_filename = ''
name_id = ''


def get_platform():
    if 'win32' in sys.platform:
        return 'win32'
    return 'linux'


def convert2wav(audio_file, wav_file):
    if get_platform() == 'win32':
        commands = '{} -i \"{}\" -acodec pcm_s16le -ac 1 -ar 16000 \"{}\" -y -loglevel panic'.format(
            FFMPEG_BIN, audio_file, wav_file)
    else:
        commands = 'ffmpeg -i \"{}\" -acodec pcm_s16le -ac 1 -ar 16000 \"{}\" -y -loglevel panic'.format(
            audio_file, wav_file)
    # commands = '{} {} {}'.format(SOX_BIN, audio_file, wav_file)
    os.system(commands)


def play_wave(wave_file='', status=True):
    if not os.path.exists(wave_file):
        return False
    ext = os.path.basename(wave_file)[-4:]
    if ext != '.wav':
        convert2wav(wave_file, 'tmp.wav')
        player.audio_play('tmp.wav', status)
    else:
        player.audio_play(wave_file, status)
    return True


def read_voice_file(voice_file):
    if not os.path.exists(voice_file):
        return []
    y, sr = librosa.load(voice_file, sr=16000)

    return np.asarray(y, dtype=np.float)


def record_from_mic():
    global rec_status, name_id, rec_filename
    rec_status = True
    rec_filename = os.path.join('Records',
                                tempfile.mktemp(prefix='{}_'.format(name_id), suffix='.wav', dir='')
                                )
    q = queue.Queue()

    def callback(indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    # Make sure the file is opened before recording anything:
    rec_buffer = []
    with sd.InputStream(samplerate=16000, device=0,
                        channels=1, callback=callback):
        print('#' * 80)
        print('press Ctrl+C to stop the recording')
        print('#' * 80)
        while rec_status:
            rec_buffer.extend(q.get())

        print("* done recording")

        raw_floats = [x for x in rec_buffer]
        floats = array.array('f', raw_floats)
        samples = [int(sample * 32767)
                   for sample in floats]
        raw_ints = struct.pack("<%dh" % len(samples), *samples)

        wf = wave.open(rec_filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(raw_ints)
        wf.close()


def record_stop():
    global rec_status
    rec_status = False


def get_duration(wave_file):
    y, fs = librosa.load(wave_file, sr=16000)
    nFrames = len(y)
    audio_length = nFrames * (1 / fs)
    return audio_length


def split_audio_ffmpeg(src_file, start_tm, end_tm, dst_file):
    if not os.path.exists(src_file):
        return ''
    if os.path.exists(dst_file):
        os.remove(dst_file)
    commands = 'ffmpeg  -i \"{}\" -ss {:.2f} -to {:.2f} \"{}\" -y -loglevel panic'.format(
            src_file, start_tm, end_tm, dst_file)
    os.system(commands)
    return dst_file


def split_sample(audio_file, voice_id, voice_name):
    seg_duration = 6
    audio_length = get_duration(audio_file)
    seg_nums = int(audio_length // seg_duration)
    if seg_nums < 1:
        return False
    for i in range(seg_nums):
        start_tm = float(i * seg_duration)
        end_tm = float((i+1) * seg_duration)
        dst_file_path = os.path.join("Database",
                                     "{}_{}_{}.wav".format(
                                         voice_name, voice_id, i+1
                                     ))
        _ = split_audio_ffmpeg(audio_file, start_tm, end_tm, dst_file_path)
    return True


def reduce_noise_power(audio_file):
    if not os.path.exists(audio_file):
        print('Input file does not exist: {}'.format(audio_file))
        return
    y, sr = librosa.load(audio_file, sr=16000)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    tmp_val = round(np.median(cent))
    threshold_h = tmp_val*1.5
    threshold_l = tmp_val*0.1

    less_noise = AudioEffectsChain().lowshelf(gain=-30.0, frequency=threshold_l, slope=0.8).highshelf(
        gain=-10.0, frequency=threshold_h, slope=0.5)
    y_clean = less_noise(y)
    # librosa.output.write_wav(audio_file, y_clean, sr)
    sf.write(audio_file, y_clean, sr, subtype='PCM_16')
