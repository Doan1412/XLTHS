import os
import torchaudio
import torch
import csv
from const import *


def to_fft(frame):
    fft = torch.fft.fft(input=frame, n=N_FFT)
    return torch.abs(fft.reshape(-1)[:int(N_FFT/2)])


def to_mean_fft_one(person, vowel_path):
    waveform, sample_rate = torchaudio.load(os.path.join(
        os.path.dirname(__file__), 'train_clean', person, vowel_path))
    fft_list = []
    for i in range(START_INDEX, 2*START_INDEX, HOP_SIZE):
        fft_list.append(to_fft(waveform[:, i:i+FRAME_SIZE]))
    stacked_tensors = torch.stack(fft_list, dim=0)
    mean_fft = torch.mean(stacked_tensors, dim=0)
    return mean_fft


def to_mean_fft_multi(people, vowel_path):
    fft_mean_list = []
    for person in people:
        fft_mean_list.append(to_mean_fft_one(person, vowel_path))
    stacked_tensors = torch.stack(fft_mean_list, dim=0)
    mean_fft_multi = torch.mean(stacked_tensors, dim=0)
    return mean_fft_multi.numpy()


people = os.listdir(os.path.join(os.path.dirname(__file__), 'train_clean'))

data = {
    'a': to_mean_fft_multi(people, 'a.wav'),
    'e': to_mean_fft_multi(people, 'e.wav'),
    'i': to_mean_fft_multi(people, 'i.wav'),
    'o': to_mean_fft_multi(people, 'o.wav'),
    'u': to_mean_fft_multi(people, 'u.wav'),
}

output = os.path.join(os.path.dirname(__file__), 'mean.csv')

# Write data to CSV
with open(output, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write header row
    csv_writer.writerow(data.keys())

    # Write data rows
    for row in zip(*data.values()):
        csv_writer.writerow(row)
