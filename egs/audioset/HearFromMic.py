import os
import sys
import csv
import numpy as np
import torch
import torchaudio
import pyaudio
import wave
import io
import time

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
target_length = 1024

torchaudio.set_audio_backend("soundfile")  # switch backend
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
from src.models import ASTModel

# download pretrained model in this directory
os.environ['TORCH_HOME'] = '../pretrained_models'


def load_label(label_csv):
    with open(label_csv, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        lines = list(reader)
    labels = []
    ids = []  # Each label has a unique id such as "/m/068hy"
    for i1 in range(1, len(lines)):
        id = lines[i1][1]
        label = lines[i1][2]
        ids.append(id)
        labels.append(label)
    return labels


if __name__ == '__main__':

    label_csv = './data/class_labels_indices.csv'  # label and indices for audioset data

    pya = pyaudio.PyAudio()
    checkpoint_path = '../../pretrained_models/audioset_10_10_0.4593.pth'
    ast_mdl = ASTModel(label_dim=527, input_tdim=target_length, imagenet_pretrain=False, audioset_pretrain=False)
    print(f'[*INFO] load checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
    audio_model.load_state_dict(checkpoint)
    audio_model = audio_model.to(torch.device("cuda:0"))
    for i in range(pya.get_device_count()):
        info = pya.get_device_info_by_index(i)
        if info['maxInputChannels'] < 1:
            continue
        print(f"{info['index']}: {info['name']}")
    stream = pya.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")
    wind=0
    frames = []
    PerSec=0
    while True:
        wind = wind + 1
        data = stream.read(CHUNK)
        frames.append(data)
        if wind > int(RATE / CHUNK * RECORD_SECONDS) :
            wind = int(RATE / CHUNK * RECORD_SECONDS)
            frames.pop(0)
        else:
            continue
        t = time.localtime(time.time())
        if t.tm_sec % RECORD_SECONDS == 0 :
           if PerSec == t.tm_sec:
               continue
           else:
               PerSec = t.tm_sec
        else:
            continue



        container = io.BytesIO()
        waveFile = wave.open(container, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(pya.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        container.seek(0)
        waveform, sr = torchaudio.load(container)
        container.close()

        #print(f'Wave Form Dimensions {waveform.size()}')
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=RATE, use_energy=False,
            window_type='hanning', num_mel_bins=128, dither=0.0,
            frame_shift=10)
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
        fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
        input_tdim = fbank.shape[0]
        feats_data = fbank.expand(1, input_tdim, 128)  # reshape the feature
        #print(f' feats_data:{feats_data.size()}')
        audio_model.eval()  # set the eval model
        with torch.no_grad():
            output = audio_model.forward(feats_data)
            output = torch.sigmoid(output)
        result_output = output.data.cpu().numpy()[0]
        labels = load_label(label_csv)
        sorted_indexes = np.argsort(result_output)[::-1]
        print('[*INFO] predict results:')
        for k in range(10):
            print('{}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]],
                                    result_output[sorted_indexes[k]]))

    stream.stop_stream()
    stream.close()
    pya.terminate()
