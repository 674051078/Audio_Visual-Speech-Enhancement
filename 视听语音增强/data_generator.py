import torch
from torch.utils.data import Dataset
import glob
from tqdm import tqdm
import random
import os
import torchaudio
from utils import stft
from torchvision.io import video
from build_model import *
random.seed('LAVSE')
n_fft=511
win_length=511
def data_nor(data, channel):
    mean = torch.mean(data, channel, keepdim=True)
    std = torch.std(data, channel, keepdim=True) + 1e-12
    nor_data = (data - mean) / std

    return nor_data, mean, std

def stft2spec(stft, normalized, save_phase, save_mean_std):
    magnitude = torch.norm(stft, 2, -1)

    if save_phase:
        # (1, 257, frames, 2) -> (257, frames, 2) -> (2, 257, frames)
        stft = stft.squeeze(0)
        stft = stft.permute(2, 0, 1)

        phase = stft / (magnitude + 1e-12)

        specgram = torch.log10(magnitude + 1) # log1p magnitude

        # normalize along frame
        if normalized:
            specgram, mean, std = data_nor(specgram, channel=-1)

        if save_mean_std:
            return (specgram, mean, std), phase
        else:
            return (specgram, None, None), phase

    else:
        specgram = torch.log10(magnitude + 1) # log1p magnitude

        # normalize along frame
        if normalized:
            specgram, mean, std = data_nor(specgram, channel=-1)

        if save_mean_std:
            return (specgram, mean, std), None
        else:
            return (specgram, None, None), None

class AV_Dataset(Dataset):
    def __init__(self,device, name, data_path_list=None, mode='no_model', av=False):
        self.name = name # name: 'train', 'val', 'test', 'clean'
        self.mode = mode # mode: 'training', 'validation', 'testing', 'no_model'
        self.av = av
        self.samples = []
        for dir_path in data_path_list:
            clean_path, noisy_path, lip_path = dir_path#分别取出路径--干净，噪声，嘴唇
            audio_paths = sorted(glob.glob(os.path.join(noisy_path ,'*.wav')))#.pt
            for audio_path in audio_paths:
                file_name = audio_path.rsplit('.', 1)[0]
                file_name = file_name.rsplit('/', 2)[-1]
                #file_name_lip = file_name.rsplit('.', 1)[0]
                file_name_lip = file_name.rsplit('_')
                file_name_lip =  'mouth_' + file_name_lip[0] + '_' + file_name_lip[1] #+'_grayscale'

                clean_audio_path = audio_path.replace(noisy_path, clean_path)
#假设 stftpt_path 是 '/path/to/noisy_files/file1.pt'，而 noisy_path 是 '/path/to/noisy_files/'，clean_path 是 '/path/to/clean_files/'。
# 通过上述替换，我们可以得到 clean_stftpt_path 为 '/path/to/clean_files/file1.pt'，从而将文件从一个目录路径映射到另一个目录路径。                
                noisy_audio_path = audio_path
                lippt_path = lip_path + file_name_lip + '.pt'##.pt

                self.samples.append((clean_audio_path, noisy_audio_path, lippt_path))

        if self.mode == 'training':
            random.shuffle(self.samples)
            self.samples = self.samples[:]
        elif self.mode == 'validation':
            random.shuffle(self.samples)
            self.samples = self.samples[:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clean_audio_path, noisy_audio_path, lippt_path = self.samples[idx]
        file_name = noisy_audio_path.rsplit('.', 1)[0]
        file_name = file_name.rsplit('/', 2)#rsplit: 是字符串的右侧分割方法（right split）。与 split 方法类似，但是从右侧开始分割字符串。
        noisy_type = file_name[-2]
        file_name = file_name[-1]
        file_name = file_name

        if self.mode == 'training' or self.mode == 'validation':

            clean_stft, sample_rate = torchaudio.load(clean_audio_path)
            noisy_stft, sample_rate = torchaudio.load(noisy_audio_path)
            stft_clean = torch.stft(clean_stft,n_fft=n_fft, hop_length=320,win_length=win_length, window=torch.hann_window(win_length), return_complex=False)
            stft_noisy = torch.stft(noisy_stft,n_fft=n_fft, hop_length=320,win_length=win_length, window=torch.hann_window(win_length),return_complex=False)#.to(device)
            #stft_clean = torch.load(clean_stftpt_path)
            #stft_noisy = torch.load(noisy_stftpt_path)

            (spec_clean, _, _), _ = stft2spec(stft_clean, normalized=False, save_phase=False, save_mean_std=False) #返回对数谱
            (spec_noisy, _, _), _ = stft2spec(stft_noisy, normalized=True, save_phase=False, save_mean_std=False)#噪声归一化

            if self.av:
                lippt = torch.load(lippt_path)
                #lippt, _, _ = video.read_video(lippt_path, pts_unit='sec')
                #print(lippt.shape)
                #lippt = lippt.permute(1,2,3,0) #3,16,24,frame_num
                #lippt = Image_process(lippt)
                #lippt = lippt.permute(1,2,3,0)
                frame_num = min(spec_noisy.shape[-1], lippt.shape[-1])

                # data structure: [0] file_name
                #                 [1] frame_num
                #                 [2] spec_clean or phase_noisy
                #                 [3] nor_spec_noisy
                #                 [4] spec_noisy_mean
                #                 [5] spec_noisy_std
                #                 [6] lippt

                data = file_name, frame_num, spec_clean[..., :frame_num], spec_noisy[..., :frame_num], None, None, lippt[..., :frame_num]

            else:
                frame_num = spec_clean.shape[-1]

                # data structure: [0] file_name
                #                 [1] frame_num
                #                 [2] spec_clean or phase_noisy
                #                 [3] nor_spec_noisy
                #                 [4] spec_noisy_mean
                #                 [5] spec_noisy_std
                #                 [6] lippt

                data = file_name, frame_num, spec_clean[..., :frame_num], spec_noisy[..., :frame_num], None, None, None

        elif self.mode == 'testing' or self.mode == 'no_model':

            # file_name = noisy_audio_path.rsplit('.', 1)[0]
            # file_name = file_name.rsplit('/', 2)
            # noisy_type = file_name[-2]
            # file_name = file_name[-1]
            # file_name = file_name + '__' + noisy_type
            if self.mode == 'testing':
                            #stft_noisy = torch.load(noisy_stftpt_path)
                noisy_stft, sample_rate = torchaudio.load(noisy_audio_path)
                stft_noisy = torch.stft(noisy_stft,n_fft=n_fft, hop_length=320,win_length=win_length, window=torch.hann_window(win_length), return_complex=False)
                (spec_noisy, _, _), phase = stft2spec(stft_noisy, normalized=True, save_phase=True, save_mean_std=False)

            elif self.mode == 'no_model':
                #stft_noisy = torch.load(noisy_stftpt_path)
                noisy_stft, sample_rate = torchaudio.load(noisy_audio_path)
                stft_noisy = torch.stft(noisy_stft,n_fft=n_fft, hop_length=320,win_length=win_length, window=torch.hann_window(win_length), return_complex=False)
                (spec_noisy, spec_noisy_mean, spec_noisy_std), phase = stft2spec(stft_noisy, normalized=True, save_phase=True, save_mean_std=True)

            if self.av:
                lippt = torch.load(lippt_path)
                #lippt, _, _ = video.read_video(lippt_path, pts_unit='sec')
                #lippt = lippt.permute(0,3,1,2)
                #lippt = Image_process(lippt)
                #lippt = lippt.permute(1,2,3,0)
                frame_num = min(spec_noisy.shape[-1], lippt.shape[-1])

                # data structure: [0] file_name
                #                 [1] frame_num
                #                 [2] spec_clean or phase_noisy
                #                 [3] nor_spec_noisy
                #                 [4] spec_noisy_mean
                #                 [5] spec_noisy_std
                #                 [6] lippt

                if self.mode == 'no_model':
                    data = file_name, frame_num, phase[..., :frame_num], spec_noisy[..., :frame_num], spec_noisy_mean, spec_noisy_std, lippt[..., :frame_num]
                else:
                    data = file_name, frame_num, phase[..., :frame_num], spec_noisy[..., :frame_num], None, None, lippt[..., :frame_num]

            else:
                frame_num = spec_noisy.shape[-1]

                # data structure: [0] file_name
                #                 [1] frame_num
                #                 [2] spec_clean or phase_noisy
                #                 [3] nor_spec_noisy
                #                 [4] spec_noisy_mean
                #                 [5] spec_noisy_std
                #                 [6] lippt

                if self.mode == 'no_model':
                    data = file_name, frame_num, phase[..., :frame_num], spec_noisy[..., :frame_num], spec_noisy_mean, spec_noisy_std, None
                else:
                    data = file_name, frame_num, phase[..., :frame_num], spec_noisy[..., :frame_num], None, None, None

        return data
