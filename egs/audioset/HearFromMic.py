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

labels_csv_file = './data/class_labels_indices.csv'
model_path = '../../pretrained_models/audioset_10_10_0.4593.pth'
torchaudio.set_audio_backend("soundfile")  # switch backend
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
os.environ['TORCH_HOME'] = '../pretrained_models'

from src.models import ASTModel

class AudioSpectrum:
    def __init__(self):
        self.__int__(chunk = 1024, rate= 44100, buffer_len=5)

    CHUNK = 1024
    RATE = 44100
    RECORD_SECONDS = 5
    ast_mdl = None
    checkpoint = None
    audio_model=None
    labels= []
    ids =[]
    def __int__(self, chunk = 1024, rate= 44100, buffer_len=5):
        self.CHUNK = chunk
        self.RATE = rate
        self.RECORD_SECONDS = buffer_len
        self.ast_mdl = ASTModel(label_dim=527, input_tdim=self.CHUNK, imagenet_pretrain=False, audioset_pretrain=False)
        self.checkpoint = torch.load(model_path, map_location='cuda')
        self.audio_model = torch.nn.DataParallel(self.ast_mdl, device_ids=[0])
        self.audio_model.load_state_dict(self.checkpoint)
        self.audio_model = self.audio_model.to(torch.device("cuda:0"))
        with open(labels_csv_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            lines = list(reader)
        for i1 in range(1, len(lines)):
            id = lines[i1][1]
            label = lines[i1][2]
            self.labels.append(label)
            self.ids.append(id)

    def make_features(self, wave_form):
        f_bank = torchaudio.compliance.kaldi.fbank(
            wave_form, htk_compat=True, sample_frequency=self.RATE, use_energy=False,
            window_type='hanning', num_mel_bins=128, dither=0.0,
            frame_shift=10)
        n_frames = f_bank.shape[0]
        p = self.CHUNK - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            f_bank = m(f_bank)
        elif p < 0:
            f_bank = f_bank[0:self.CHUNK, :]

        f_bank = (f_bank - (-4.2677393)) / (4.5689974 * 2)
        input_t_dim = f_bank.shape[0]
        return f_bank.expand(1, input_t_dim, 128)

    def convert_frames_to_tensor(self, Frames):
        container = io.BytesIO()
        wave_file = wave.open(container, 'wb')
        wave_file.setnchannels(1)
        wave_file.setsampwidth(pya.get_sample_size(pyaudio.paInt16))
        wave_file.setframerate(self.RATE)
        wave_file.writeframes(b''.join(frames))
        wave_file.close()
        container.seek(0)
        wave_form, sr = torchaudio.load(container)
        container.close()
        return wave_form
    def Evaluate_Frames(self,waveform):
        result = {}
        label_results={}
        result["is_humwn"] = False
        feats_data = self.make_features(waveform)
        self.audio_model.eval()  # set the eval model
        with torch.no_grad():
            output = self.audio_model.forward(feats_data)
            output = torch.sigmoid(output)
        result_output = output.data.cpu().numpy()[0]
        sorted_indexes = np.argsort(result_output)[::-1]
        for k in range(5):
            r_label = self.labels[sorted_indexes[k]]
            percent=int(result_output[sorted_indexes[k]] * 100.)
            label_results[r_label] = percent
            if r_label=='Speech' and percent >= 25 :
                result["is_human"] = True
        result["sound_labels"] =label_results
        return result






if __name__ == '__main__':
    MyAudioSpectrum = AudioSpectrum()
    pya = pyaudio.PyAudio()
    stream = pya.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=AudioSpectrum.RATE,
                    input=True,
                    frames_per_buffer=AudioSpectrum.CHUNK)

    print("* recording")
    wind=0
    frames = []
    PerSec=0
    while True:
        wind = wind + 1
        data = stream.read(AudioSpectrum.CHUNK)
        frames.append(data)
        if wind > int(AudioSpectrum.RATE / AudioSpectrum.CHUNK * AudioSpectrum.RECORD_SECONDS) :
            wind = int(AudioSpectrum.RATE / AudioSpectrum.CHUNK * AudioSpectrum.RECORD_SECONDS)
            frames.pop(0)
        else:
            continue
        t = time.localtime(time.time())
        if t.tm_sec % AudioSpectrum.RECORD_SECONDS == 0 :
           if PerSec == t.tm_sec:
               continue
           else:
               PerSec = t.tm_sec
        else:
            continue
        waveform = MyAudioSpectrum.convert_frames_to_tensor(frames)
        result=MyAudioSpectrum.Evaluate_Frames(waveform)
        label_results=result["sound_labels"]
        print('[*INFO] predict results,is human:{}'.format(result["is_human"]))
        for label in label_results:
            print('{}: %{}'.format(label,label_results[label]))

    stream.stop_stream()
    stream.close()
    pya.terminate()
