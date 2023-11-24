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
WIN_SIZE = int(30*(10**-3)*SAMPLE_RATE)
HOP_SIZE = int(10*(10**-3)*SAMPLE_RATE)
NUM_CLUSTER = 5


def to_mfcc(waveform):
    transform = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={"n_fft": WIN_SIZE, "win_length": WIN_SIZE, "hop_length": HOP_SIZE,
                   "n_mels": 23, "center": True, "window_fn": torch.hann_window},
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
        endIndex = int(startIndex*2-0.03*SAMPLE_RATE)
        mfccs.append(to_mfcc(waveform[:, startIndex:endIndex]))
    return torch.stack(mfccs)


def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5


def predict_one(data, person, vowel_path):
    dist = {label: 99999 for label in data.keys()}
    waveform, sample_rate = torchaudio.load(os.path.join(
        os.path.dirname(__file__), 'test_clean', person, vowel_path))
    wave_len = waveform.size(1)
    startIndex = int(wave_len/3)
    endIndex = startIndex*2
    mfcc = to_mfcc(waveform[:, startIndex:endIndex])
    for label in data.keys():
        for vector in data[label]:
            dist[label] = min(dist[label], euclidean(vector, mfcc))
    return min(dist, key=lambda k: dist[k])


def k_means(num_clusters, X):
    input = X.numpy()
    return torch.tensor(KMeans(n_clusters=num_clusters, n_init='auto').fit(input).cluster_centers_)
