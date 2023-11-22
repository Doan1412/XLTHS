import torch
import torchaudio
import os
import csv
from tqdm import tqdm
from kmeans_pytorch import kmeans
from collections import OrderedDict
import matplotlib.pyplot as plt

N_FFT = 1024
N_MFCC = 13
STABLE_DELTA = 0.1
SAMPLE_RATE = 16000
WIN_SIZE = int(30*(10**-3)*SAMPLE_RATE)
HOP_SIZE = int(10*(10**-3)*SAMPLE_RATE)
START_TIME = 0.065
START_INDEX = int(START_TIME * SAMPLE_RATE)
END_INDEX = int(START_INDEX + STABLE_DELTA*SAMPLE_RATE)
NUM_CLUSTER = 4


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
    for person in tqdm(people):
        waveform, sample_rate = torchaudio.load(os.path.join(
            os.path.dirname(__file__), 'train_clean', person, vowel_path))
        # startIndex = less_variant_segment(waveform)
        mfccs.append(to_mfcc(waveform[:, START_INDEX:END_INDEX]))
    return torch.stack(mfccs)


def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5


def predict_one(data, person, vowel_path):
    dist = {label: 99999 for label in data.keys()}
    waveform, sample_rate = torchaudio.load(os.path.join(
        os.path.dirname(__file__), 'test_clean', person, vowel_path))
    mfcc = to_mfcc(waveform[:, START_INDEX:END_INDEX])
    for label in data.keys():
        for vector in data[label]:
            dist[label] = min(dist[label], euclidean(vector, mfcc))
    return min(dist, key=lambda k: dist[k])
