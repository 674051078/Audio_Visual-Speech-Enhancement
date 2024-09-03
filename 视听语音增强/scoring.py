# PESQ
import subprocess
# STOI
from scipy.linalg import toeplitz
import soundfile as sf
from pystoi.stoi import stoi
from pesq import pesq
from prepare_path_list import  snr_test_list
import glob
from tqdm import tqdm
import csv
import os
import numpy as np
from utils import cal_time
import time

def eval_composite(ref_wav, deg_wav, sr=16000):
    ref_wav = ref_wav.reshape(-1)
    deg_wav = deg_wav.reshape(-1)

    alpha = 0.95
    len_ = min(ref_wav.shape[0], deg_wav.shape[0])
    ref_wav = ref_wav[:len_]
    ref_len = ref_wav.shape[0]
    deg_wav = deg_wav[:len_]

    # Compute WSS measure
    wss_dist_vec = wss(ref_wav, deg_wav, 16000)
    wss_dist_vec = sorted(wss_dist_vec, reverse=False)
    wss_dist = np.mean(wss_dist_vec[:int(round(len(wss_dist_vec) * alpha))])

    # Compute LLR measure
    LLR_dist = llr(ref_wav, deg_wav, 16000)
    LLR_dist = sorted(LLR_dist, reverse=False)
    LLRs = LLR_dist
    LLR_len = round(len(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[:LLR_len])

    # Compute the SSNR
    _, segsnr_mean = SSNR(ref_wav, deg_wav, 16000)
    segSNR = np.mean(segsnr_mean)

    #lsd = LSD(ref_wav, deg_wav)
    # Compute the PESQ
    #pesq_raw = PESQ(ref_wav, deg_wav)
    PESQ = pesq(sr, ref_wav, deg_wav, 'wb')
    STOI = stoi(ref_wav, deg_wav, sr, extended=False)

    Csig = 3.093 - 1.029 * llr_mean + 0.603 * PESQ - 0.009 * wss_dist
    Csig = trim_mos(Csig)
    Cbak = 1.634 + 0.478 * PESQ - 0.007 * wss_dist + 0.063 * segSNR
    Cbak = trim_mos(Cbak)
    Covl = 1.594 + 0.805 * PESQ - 0.512 * llr_mean - 0.007 * wss_dist
    Covl = trim_mos(Covl)

    return  PESQ,  STOI,  Csig,  Cbak,  Covl #, lsd

def scoring_PESQ(clean, noisy, enhan, sr):
    noisy_pesq = pesq(sr, clean, noisy, 'wb')
    enhan_pesq = pesq(sr, clean, enhan, 'wb')
    return noisy_pesq, enhan_pesq

def scoring_STOI(clean, noisy, enhan, sr):
    noisy_stoi = stoi(clean, noisy, sr, extended=False)
    enhan_stoi = stoi(clean, enhan, sr, extended=False)
    return noisy_stoi, enhan_stoi

def scoring_file(clean_wav_path, noisy_wav_path, enhan_wav_path):

    file_name = noisy_wav_path.rsplit('.', 1)[0]
    file_name = file_name.rsplit('/', 2)
    noise_type = file_name[-2]
    file_name = file_name[-1]

    # print('noise_type =', noise_type)
    # print('file_name =', file_name)

    file_name =  noise_type + '_' + file_name
    # print('file_name =', file_name)

    clean, sr = sf.read(clean_wav_path)
    noisy, sr = sf.read(noisy_wav_path)
    enhan, sr = sf.read(enhan_wav_path)

    noisy_pesq, noisy_stoi, noisy_Csig, noisy_Cbak, noisy_Covl = eval_composite(clean, noisy, sr)#, noisy_lsd
    enhan_pesq, enhan_stoi, enhan_Csig, enhan_Cbak, enhan_Covl= eval_composite(clean, enhan, sr)#, enhan_lsd 

    # noisy_pesq, enhan_pesq = scoring_PESQ(clean, noisy, enhan, sr)
    # noisy_stoi, enhan_stoi = scoring_STOI(clean, noisy, enhan, sr)

    return file_name, noisy_pesq, enhan_pesq, noisy_stoi, enhan_stoi, noisy_Csig, enhan_Csig, noisy_Cbak, enhan_Cbak, noisy_Covl, enhan_Covl #, noisy_lsd,enhan_lsd

def scoring_dir(scoring_path_list):
    scores = []

    for dir_path in scoring_path_list:
        clean_path, noisy_path, enhan_path = dir_path

        print('Scoring PESQ and STOI of wav in \'' + noisy_path + '\'\n' + 'and  ' + enhan_path + '...')#计算有增强后语音与纯净语音的分数
        for wav_path in tqdm(sorted(glob.glob(noisy_path + '*.wav'))): #sorted()
            clean_wav_path = wav_path.replace(noisy_path, clean_path)
            noisy_wav_path = wav_path
            enhan_wav_path = wav_path.replace(noisy_path, enhan_path)

            # print('clean_wav_path =', clean_wav_path)
            # print('noisy_wav_path =', noisy_wav_path)
            # print('enhan_wav_path =', enhan_wav_path)
            scores.append(scoring_file(clean_wav_path, noisy_wav_path, enhan_wav_path))

    return scores # list of (file_name, noisy_pesq, enhan_pesq, noisy_stoi, enhan_stoi)

def prepare_scoring_list(test_spk_list, snr_test_list, result_audio_path, result_nmaudio_path, dataset):#, noise_test_list
    test_spk_list = [str(i).zfill(2) for i in test_spk_list]

    path_list = []

    if dataset == 'AV_enh':
        for spk_num in test_spk_list:
            #for noise in noise_test_list:
            for snr in snr_test_list:
                snrdb = str(snr) + 'dB'#noise + '_' + str(snr).replace('-', 'n')  + 'db'#
                # (clean_path, noisy_path, enhan_path)
                path_list.extend([
                (result_nmaudio_path + 'SP' + spk_num + '/clean/', result_nmaudio_path + 'SP' + spk_num + '/' + snrdb + '/', result_audio_path + 'SP' + spk_num + '/' + snrdb + '/')
                ])

    # print('path_list =', path_list)

    return path_list


def calculate_average_scores(scores, target_db):

    # filtered_scores = []
    # for score in scores:
    #     if target_db in score[0].split('_'):
    #         filtered_scores.append(score)
    filtered_scores = [score for score in scores if target_db in score[0].split('_')]  # 仅选择目标 db 的分数项

    if not filtered_scores:
        print(f'No scores found for {target_db}.')
        return None

    # 提取目标项的分数并计算平均值
    noisy_pesq_values = [score[1] for score in filtered_scores]
    enhan_pesq_values = [score[2] for score in filtered_scores]
    noisy_stoi_values = [score[3] for score in filtered_scores]
    enhan_stoi_values = [score[4] for score in filtered_scores]
    noisy_Csig_values = [score[5] for score in filtered_scores]
    enhan_Csig_values = [score[6] for score in filtered_scores]
    noisy_Cbak_values = [score[7] for score in filtered_scores]
    enhan_Cbak_values = [score[8] for score in filtered_scores]
    noisy_Covl_values = [score[9] for score in filtered_scores]
    enhan_Covl_values = [score[10] for score in filtered_scores]
    #noisy_lsd_values = [score[11] for score in filtered_scores]
    #enhan_lsd_values = [score[12] for score in filtered_scores]

    mean_noisy_pesq = sum(noisy_pesq_values) / len(noisy_pesq_values)
    mean_enhan_pesq = sum(enhan_pesq_values) / len(enhan_pesq_values)
    mean_noisy_stoi = sum(noisy_stoi_values) / len(noisy_stoi_values)
    mean_enhan_stoi = sum(enhan_stoi_values) / len(enhan_stoi_values)
    mean_noisy_Csig = sum(noisy_Csig_values) / len(noisy_Csig_values)
    mean_enhan_Csig = sum(enhan_Csig_values) / len(enhan_Csig_values)
    mean_noisy_Cbak = sum(noisy_Cbak_values) / len(noisy_Cbak_values)
    mean_enhan_Cbak = sum(enhan_Cbak_values) / len(enhan_Cbak_values)
    mean_noisy_Covl = sum(noisy_Covl_values) / len(noisy_Covl_values)
    mean_enhan_Covl = sum(enhan_Covl_values) / len(enhan_Covl_values)
    #mean_noisy_lsd = sum(noisy_lsd_values) / len(noisy_lsd_values)
    #mean_enhan_lsd = sum(enhan_lsd_values) / len(enhan_lsd_values)

    return mean_noisy_pesq, mean_enhan_pesq, mean_noisy_stoi, mean_enhan_stoi, mean_noisy_Csig, mean_enhan_Csig, mean_noisy_Cbak, mean_enhan_Cbak, mean_noisy_Covl, mean_enhan_Covl#,mean_noisy_lsd,mean_enhan_lsd

def write_score(path_list, result_model_path):
    start_time = time.time()
    scores = scoring_dir(path_list)

    # 创建目录用于存储每个 db 的结果
    db_results_path = os.path.join(result_model_path, 'db_results')
    os.makedirs(db_results_path, exist_ok=True)

    target_dbs = ['-5dB', '-2dB', '1dB', '4dB']#,'7dB'
    for target_db in target_dbs:
        # 计算每个目标 db 下的平均值
        average_scores = calculate_average_scores(scores, target_db)
        if average_scores is not None:
            print(f'Average scores for {target_db}: Noisy_PESQ={average_scores[0]},  Enhan_PESQ={average_scores[1]},  Noisy_STOI={average_scores[2]},  Enhan_STOI={average_scores[3]}\n')
            print(f'Average scores for {target_db}: Noisy_CSIG={average_scores[4]},  Enhan_CSIG={average_scores[5]},  Noisy_CBAK={average_scores[6]},  Enhan_CBAK={average_scores[7]}\n')
            print(f'Average scores for {target_db}: Noisy_COVL={average_scores[8]},  Enhan_COVL={average_scores[9]}\n')#, Noisy_LSD={average_scores[10]}, Enhan_LSD={average_scores[11]}

            # 写入结果到目标 db 文件夹
            db_result_file_path = os.path.join(db_results_path, f'Results_Report_{target_db}.csv')
            with open(db_result_file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(('File_Name', 'Noisy_PESQ', 'Enhan_PESQ', 'Noisy_STOI', 'Enhan_STOI','Noisy_CSIG', 'Enhan_CSIG', 'Noisy_CBAK', 'Enhan_CBAK','Noisy_COVL', 'Enhan_COVL'))
                for score in scores:
                    if target_db in score[0]:
                        csv_writer.writerow(score)

                # 写入平均值
                csv_writer.writerow(())
                csv_writer.writerow(('total mean', average_scores[0], average_scores[1], average_scores[2], average_scores[3],average_scores[4],average_scores[5], average_scores[6], average_scores[7],average_scores[8], average_scores[9]))

    print('Results written to:', db_results_path)
    end_time = time.time()
    score_time = cal_time(start_time, end_time)
    print('Scoring complete.\n')

    return score_time

# ----------------------------- HELPERS ------------------------------------ #
def trim_mos(val):
    return min(max(val, 1), 5)


def lpcoeff(speech_frame, model_order):
    # (1) Compute Autocor lags
    winlength = speech_frame.shape[0]
    R = []
    for k in range(model_order + 1):
        first = speech_frame[:(winlength - k)]
        second = speech_frame[k:winlength]
        R.append(np.sum(first * second))

    # (2) Lev-Durbin
    a = np.ones((model_order,))
    E = np.zeros((model_order + 1,))
    rcoeff = np.zeros((model_order,))
    E[0] = R[0]
    for i in range(model_order):
        if i == 0:
            sum_term = 0
        else:
            a_past = a[:i]
            sum_term = np.sum(a_past * np.array(R[i:0:-1]))
        rcoeff[i] = (R[i + 1] - sum_term) / E[i]
        a[i] = rcoeff[i]
        if i > 0:
            a[:i] = a_past[:i] - rcoeff[i] * a_past[::-1]
        E[i + 1] = (1 - rcoeff[i] * rcoeff[i]) * E[i]
    acorr = np.array(R, dtype=np.float32)
    refcoeff = np.array(rcoeff, dtype=np.float32)
    a = a * -1
    lpparams = np.array([1] + list(a), dtype=np.float32)
    acorr = np.array(acorr, dtype=np.float32)
    refcoeff = np.array(refcoeff, dtype=np.float32)
    lpparams = np.array(lpparams, dtype=np.float32)

    return acorr, refcoeff, lpparams


# -------------------------------------------------------------------------- #

# ---------------------- Speech Quality Metric ----------------------------- #

def LSD(audio_buffer_1, audio_buffer_2):
    # Compute FFT
    fft_1 = np.fft.fft(audio_buffer_1)
    fft_2 = np.fft.fft(audio_buffer_2)
    # Compute power spectra
    power_spectrum_1 = np.abs(fft_1) ** 2
    power_spectrum_2 = np.abs(fft_2) ** 2
    # Compute LSD
    log_spectral_distance = np.sqrt(np.mean(np.power(10 * np.log10(power_spectrum_1 / power_spectrum_2+10E-8), 2)))
    return log_spectral_distance

def SSNR(ref_wav, deg_wav, srate=16000, eps=1e-10):
    """ Segmental Signal-to-Noise Ratio Objective Speech Quality Measure
        This function implements the segmental signal-to-noise ratio
        as defined in [1, p. 45] (see Equation 2.12).
    """
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]

    # scale both to have same dynamic range. Remove DC too.
    clean_speech -= clean_speech.mean()
    processed_speech -= processed_speech.mean()
    processed_speech *= (np.max(np.abs(clean_speech)) / np.max(np.abs(processed_speech)))

    # Signal-to-Noise Ratio
    dif = ref_wav - deg_wav
    overall_snr = 10 * np.log10(np.sum(ref_wav ** 2) / (np.sum(dif ** 2) +
                                                        10e-20))
    # global variables
    winlength = int(np.round(30 * srate / 1000))  # 30 msecs
    skiprate = winlength // 4
    MIN_SNR = -10
    MAX_SNR = 35

    # For each frame, calculate SSNR
    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    segmental_snr = []

    for frame_count in range(int(num_frames)):
        # (1) get the frames for the test and ref speech.
        # Apply Hanning Window
        clean_frame = clean_speech[start:start + winlength]
        processed_frame = processed_speech[start:start + winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compute Segmental SNR
        signal_energy = np.sum(clean_frame ** 2)
        noise_energy = np.sum((clean_frame - processed_frame) ** 2)
        segmental_snr.append(10 * np.log10(signal_energy / (noise_energy + eps) + eps))
        segmental_snr[-1] = max(segmental_snr[-1], MIN_SNR)
        segmental_snr[-1] = min(segmental_snr[-1], MAX_SNR)
        start += int(skiprate)
    return overall_snr, segmental_snr


def wss(ref_wav, deg_wav, srate):
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]

    assert clean_length == processed_length, clean_length

    winlength = round(30 * srate / 1000.)  # 240 wlen in samples
    skiprate = np.floor(winlength / 4)
    max_freq = srate / 2
    num_crit = 25  # num of critical bands

    USE_FFT_SPECTRUM = 1
    n_fft = int(2 ** np.ceil(np.log(2 * winlength) / np.log(2)))
    n_fftby2 = int(n_fft / 2)
    Kmax = 20
    Klocmax = 1

    # Critical band filter definitions (Center frequency and BW in Hz)
    cent_freq = [50., 120, 190, 260, 330, 400, 470, 540, 617.372,
                 703.378, 798.717, 904.128, 1020.38, 1148.30,
                 1288.72, 1442.54, 1610.70, 1794.16, 1993.93,
                 2211.08, 2446.71, 2701.97, 2978.04, 3276.17,
                 3597.63]
    bandwidth = [70., 70, 70, 70, 70, 70, 70, 77.3724, 86.0056,
                 95.3398, 105.411, 116.256, 127.914, 140.423,
                 153.823, 168.154, 183.457, 199.776, 217.153,
                 235.631, 255.255, 276.072, 298.126, 321.465,
                 346.136]

    bw_min = bandwidth[0]  # min critical bandwidth

    # set up critical band filters. Note here that Gaussianly shaped filters
    # are used. Also, the sum of the filter weights are equivalent for each
    # critical band filter. Filter less than -30 dB and set to zero.
    min_factor = np.exp(-30. / (2 * 2.303))  # -30 dB point of filter

    crit_filter = np.zeros((num_crit, n_fftby2))
    all_f0 = []
    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * (n_fftby2)
        all_f0.append(np.floor(f0))
        bw = (bandwidth[i] / max_freq) * (n_fftby2)
        norm_factor = np.log(bw_min) - np.log(bandwidth[i])
        j = list(range(n_fftby2))
        crit_filter[i, :] = np.exp(-11 * (((j - np.floor(f0)) / bw) ** 2) + \
                                   norm_factor)
        crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > \
                                                 min_factor)

    # For each frame of input speech, compute Weighted Spectral Slope Measure
    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0  # starting sample
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    distortion = []

    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speeech.
        # Multiply by Hanning window.
        clean_frame = clean_speech[start:start + winlength]
        processed_frame = processed_speech[start:start + winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Compuet Power Spectrum of clean and processed
        clean_spec = (np.abs(np.fft.fft(clean_frame, n_fft)) ** 2)
        processed_spec = (np.abs(np.fft.fft(processed_frame, n_fft)) ** 2)
        clean_energy = [None] * num_crit
        processed_energy = [None] * num_crit

        # (3) Compute Filterbank output energies (in dB)
        for i in range(num_crit):
            clean_energy[i] = np.sum(clean_spec[:n_fftby2] * \
                                     crit_filter[i, :])
            processed_energy[i] = np.sum(processed_spec[:n_fftby2] * \
                                         crit_filter[i, :])
        clean_energy = np.array(clean_energy).reshape(-1, 1)
        eps = np.ones((clean_energy.shape[0], 1)) * 1e-10
        clean_energy = np.concatenate((clean_energy, eps), axis=1)
        clean_energy = 10 * np.log10(np.max(clean_energy, axis=1))
        processed_energy = np.array(processed_energy).reshape(-1, 1)
        processed_energy = np.concatenate((processed_energy, eps), axis=1)
        processed_energy = 10 * np.log10(np.max(processed_energy, axis=1))

        # (4) Compute Spectral Shape (dB[i+1] - dB[i])
        clean_slope = clean_energy[1:num_crit] - clean_energy[:num_crit - 1]
        processed_slope = processed_energy[1:num_crit] - \
                          processed_energy[:num_crit - 1]

        # (5) Find the nearest peak locations in the spectra to each
        # critical band. If the slope is negative, we search
        # to the left. If positive, we search to the right.
        clean_loc_peak = []
        processed_loc_peak = []
        for i in range(num_crit - 1):
            if clean_slope[i] > 0:
                # search to the right
                n = i
                while n < num_crit - 1 and clean_slope[n] > 0:
                    n += 1
                clean_loc_peak.append(clean_energy[n - 1])
            else:
                # search to the left
                n = i
                while n >= 0 and clean_slope[n] <= 0:
                    n -= 1
                clean_loc_peak.append(clean_energy[n + 1])
            # find the peaks in the processed speech signal
            if processed_slope[i] > 0:
                n = i
                while n < num_crit - 1 and processed_slope[n] > 0:
                    n += 1
                processed_loc_peak.append(processed_energy[n - 1])
            else:
                n = i
                while n >= 0 and processed_slope[n] <= 0:
                    n -= 1
                processed_loc_peak.append(processed_energy[n + 1])

        # (6) Compuet the WSS Measure for this frame. This includes
        # determination of the weighting functino
        dBMax_clean = max(clean_energy)
        dBMax_processed = max(processed_energy)

        # The weights are calculated by averaging individual
        # weighting factors from the clean and processed frame.
        # These weights W_clean and W_processed should range
        # from 0 to 1 and place more emphasis on spectral
        # peaks and less emphasis on slope differences in spectral
        # valleys.  This procedure is described on page 1280 of
        # Klatt's 1982 ICASSP paper.
        clean_loc_peak = np.array(clean_loc_peak)
        processed_loc_peak = np.array(processed_loc_peak)
        Wmax_clean = Kmax / (Kmax + dBMax_clean - clean_energy[:num_crit - 1])
        Wlocmax_clean = Klocmax / (Klocmax + clean_loc_peak - clean_energy[:num_crit - 1])
        W_clean = Wmax_clean * Wlocmax_clean
        Wmax_processed = Kmax / (Kmax + dBMax_processed - processed_energy[:num_crit - 1])
        Wlocmax_processed = Klocmax / (Klocmax + processed_loc_peak - processed_energy[:num_crit - 1])
        W_processed = Wmax_processed * Wlocmax_processed
        W = (W_clean + W_processed) / 2
        distortion.append(np.sum(W * (clean_slope[:num_crit - 1] - processed_slope[:num_crit - 1]) ** 2))

        # this normalization is not part of Klatt's paper, but helps
        # to normalize the meaasure. Here we scale the measure by the sum of the
        # weights
        distortion[frame_count] = distortion[frame_count] / np.sum(W)
        start += int(skiprate)
    return distortion


def llr(ref_wav, deg_wav, srate):
    clean_speech = ref_wav
    processed_speech = deg_wav
    clean_length = ref_wav.shape[0]
    processed_length = deg_wav.shape[0]
    assert clean_length == processed_length, clean_length

    winlength = round(30 * srate / 1000.)  # 240 wlen in samples
    skiprate = np.floor(winlength / 4)
    if srate < 10000:
        # LPC analysis order
        P = 10
    else:
        P = 16

    # For each frame of input speech, calculate the Log Likelihood Ratio
    num_frames = int(clean_length / skiprate - (winlength / skiprate))
    start = 0
    time = np.linspace(1, winlength, winlength) / (winlength + 1)
    window = 0.5 * (1 - np.cos(2 * np.pi * time))
    distortion = []

    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speeech.
        # Multiply by Hanning window.
        clean_frame = clean_speech[start:start + winlength]
        processed_frame = processed_speech[start:start + winlength]
        clean_frame = clean_frame * window
        processed_frame = processed_frame * window

        # (2) Get the autocorrelation logs and LPC params used
        # to compute the LLR measure
        R_clean, Ref_clean, A_clean = lpcoeff(clean_frame, P)
        R_processed, Ref_processed, A_processed = lpcoeff(processed_frame, P)
        A_clean = A_clean[None, :]
        A_processed = A_processed[None, :]

        # (3) Compute the LLR measure
        numerator = A_processed.dot(toeplitz(R_clean)).dot(A_processed.T)
        denominator = A_clean.dot(toeplitz(R_clean)).dot(A_clean.T)

        if (numerator / denominator) <= 0:
            print(f'Numerator: {numerator}')
            print(f'Denominator: {denominator}')

        log_ = np.log(abs(numerator / denominator))
        distortion.append(np.squeeze(log_))
        start += int(skiprate)
    return np.nan_to_num(np.array(distortion))
# def write_score(path_list, result_model_path):
#     start_time = time.time()

#     # print('result_model_path =', result_model_path)
#     model_detail = result_model_path.rsplit('/', 2)[-2]
# #split()分割包括 ' '   #路径为'/data/result/LAVSE_epoch(1)lr(1e-05)bs(16)lc(1e-03)/'   #rsplit('/', 2)[-2]后为['/data/result', 'LAVSE_epoch(1)lr(1e-05)bs(16)lc(1e-03)', '']
#     # print('model_detail =', model_detail)

#     scores = scoring_dir(path_list)

#     count = len(scores)#/len(snr_test_list)
#     sum_noisy_pesq = 0.0
#     sum_noisy_stoi = 0.0
#     sum_enhan_pesq = 0.0
#     sum_enhan_stoi = 0.0

#     # CSV Result Output
#     f = open(result_model_path + 'Results_Report[%s].csv' % model_detail, 'w')
#     w = csv.writer(f)
#     w.writerow(('File_Name', 'Noisy_PESQ', 'Enhan_PESQ', 'Noisy_STOI', 'Enhan_STOI'))

#     for score in scores:
#         #db_num = score[0].split('_')[0]

#         file_name, noisy_pesq, enhan_pesq, noisy_stoi, enhan_stoi = score
#         file_name_db_num = file_name.split('_')[0]

#         sum_noisy_pesq += noisy_pesq
#         sum_noisy_stoi += noisy_stoi
#         sum_enhan_pesq += enhan_pesq
#         sum_enhan_stoi += enhan_stoi

#         w.writerow((file_name, noisy_pesq, enhan_pesq, noisy_stoi, enhan_stoi))


#     mean_noisy_pesq = sum_noisy_pesq / count
#     mean_noisy_stoi = sum_noisy_stoi / count
#     mean_enhan_pesq = sum_enhan_pesq / count
#     mean_enhan_stoi = sum_enhan_stoi / count

#     print()
#     print('mean_noisy_pesq = %5.3f, mean_noisy_stoi = %5.3f' % (mean_noisy_pesq, mean_noisy_stoi))
#     print('mean_enhan_pesq = %5.3f, mean_enhan_stoi = %5.3f' % (mean_enhan_pesq, mean_enhan_stoi))
#     print()

#     w.writerow(())
#     w.writerow(('total mean', mean_noisy_pesq, mean_enhan_pesq, mean_noisy_stoi, mean_enhan_stoi))
#     f.close()

#     # remove the by-product created by the PESQ execute file
#     #subprocess.call(['rm', '_pesq_itu_results.txt'])
#     #subprocess.call(['rm', '_pesq_results.txt'])

#     end_time = time.time()

#     score_time = cal_time(start_time, end_time)
#     print('Scoring complete.\n')

#     return score_time
