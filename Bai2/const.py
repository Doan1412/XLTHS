import torch
from tqdm import tqdm
import os
import torchaudio
import csv
import numpy as np


N_FFT = 2048
SAMPLE_RATE = 16000
FRAME_SIZE = int(30*(10**-3)*SAMPLE_RATE)
HOP_SIZE = int(10*(10**-3)*SAMPLE_RATE)


def to_fft(frame):
    hamming = torch.hann_window(frame.size(1)).unsqueeze(0)
    frame = torch.mul(frame, hamming)
    fft = torch.fft.fft(input=frame, n=N_FFT)
    return torch.abs(fft.reshape(-1)[:int(N_FFT/2)])


def to_mean_fft_one(person, set, vowel_path):
    waveform, sample_rate = torchaudio.load(os.path.join(
        os.path.dirname(__file__), set, person, vowel_path))
    fft_list = []
    wave_len = waveform.size(1)
    startIndex = int(wave_len/3)
    endIndex = int(startIndex*2-0.03*SAMPLE_RATE)
    for i in range(startIndex, endIndex, HOP_SIZE):
        fft_list.append(to_fft(waveform[:, i:i+FRAME_SIZE]))
    stacked_tensors = torch.stack(fft_list, dim=0)
    mean_fft = torch.mean(stacked_tensors, dim=0)
    return mean_fft


def to_mean_fft_multi(people, set, vowel_path):
    fft_mean_list = []
    for person in tqdm(people):
        fft_mean_list.append(to_mean_fft_one(person, set, vowel_path))
    stacked_tensors = torch.stack(fft_mean_list, dim=0)
    mean_fft_multi = torch.mean(stacked_tensors, dim=0)
    return mean_fft_multi.numpy()


def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5
