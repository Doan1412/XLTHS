import torch
import torchaudio
import os
import csv
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

N_MFCC = 13
SAMPLE_RATE = 16000
WIN_SIZE = int(40*(10**-3)*SAMPLE_RATE)
HOP_SIZE = int(5*(10**-3)*SAMPLE_RATE)


def to_mfcc(waveform):
    transform = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={"n_fft": 1024, "win_length": WIN_SIZE, "hop_length": HOP_SIZE,
                   "n_mels": 46, "center": True, "window_fn": torch.hann_window},
        # rectangle: torch.ones, hamming: torch.hann_window
    )
    mfccs = transform(waveform)
    mfcc = torch.mean(mfccs, dim=2)
    return mfcc[0]


def concentrate_mfcc(people, vowel_path):
    mfccs = []
    for person in people:
        waveform, sample_rate = torchaudio.load(os.path.join(
            os.path.dirname(__file__), 'train_clean', person, vowel_path))
        wave_len = waveform.size(1)
        startIndex = int(wave_len/3)
        endIndex = int(startIndex*2)
        mfccs.append(to_mfcc(waveform[:, startIndex:endIndex]))
    return torch.stack(mfccs)


def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5


def predict_one(data, person, vowel_path):
    dist = {}
    waveform, sample_rate = torchaudio.load(os.path.join(
        os.path.dirname(__file__), 'test_clean', person, vowel_path))
    mean = to_mfcc(waveform)
    for label in data.keys():
        dist[label] = euclidean(data[label], mean)
    return min(dist, key=lambda k: dist[k])


def means(X):
    return torch.mean(X, dim=0).numpy()
