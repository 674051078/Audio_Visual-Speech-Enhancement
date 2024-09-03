import torch
from scipy import linalg
import numpy as np
import scipy
import torch.nn.functional as F
import torch.nn as nn
class TF_loss(object):  #(output_com, label)时域
    def __call__(self, outputs, labels):
        pred_stft = outputs
        true_stft = labels

        #pred_stft = torch.stft(y_pred, n_fft, hop_length, win_length=n_fft, window=WINDOW, center=True)
        #true_stft = torch.stft(y_true, n_fft, hop_length, win_length=n_fft, window=WINDOW, center=True)
        pred_stft_real, pred_stft_imag = pred_stft[:, 0,:,  :], pred_stft[:, 1,:, :]
        true_stft_real, true_stft_imag = true_stft[:, 0,:,  :], true_stft[:, 1,:, :]
        pred_mag = torch.sqrt(pred_stft_real ** 2 + pred_stft_imag ** 2 + 1e-12)
        true_mag = torch.sqrt(true_stft_real ** 2 + true_stft_imag ** 2 + 1e-12)
        pred_real_c = pred_stft_real / (pred_mag ** (2 / 3))
        pred_imag_c = pred_stft_imag / (pred_mag ** (2 / 3))
        true_real_c = true_stft_real / (true_mag ** (2 / 3))
        true_imag_c = true_stft_imag / (true_mag ** (2 / 3))
        real_loss = torch.mean((pred_real_c - true_real_c) ** 2)
        imag_loss = torch.mean((pred_imag_c - true_imag_c) ** 2)
        
        mag_loss = torch.mean((pred_mag ** (1 / 3) - true_mag ** (1 / 3)) ** 2)

        return (real_loss + imag_loss)*0.1 + mag_loss
    
class time_loss(object):  #(output_com, label)时域
    def __call__(self, outputs, labels):
        n_fft = 1023
        hop_length = 320
        WINDOW = torch.hann_window(n_fft).cuda()
        pred_stft = outputs.permute(0,3,2,1)
        true_stft = labels.permute(0,3,2,1)
        pred_wav = torch.istft(pred_stft, n_fft, hop_length, win_length=n_fft, window=WINDOW)
        true_wav = torch.istft(true_stft, n_fft, hop_length, win_length=n_fft, window=WINDOW)

        loss_MAE = nn.L1Loss()
        loss = loss_MAE(pred_wav,true_wav)#torch.abs(pred_wav-true_wav)) /
        return loss* 0.2