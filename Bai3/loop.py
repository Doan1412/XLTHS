import numpy as np
import matplotlib.pyplot as plt
import os
import torchaudio
import torch
import csv
from kmeans_pytorch import kmeans
from const import *
from tqdm import tqdm
from collections import OrderedDict


def to_mfcc(waveform):
    transform = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={"n_fft": WIN_SIZE, "win_length": WIN_SIZE, "hop_length": HOP_SIZE,
                   "n_mels": 23, "center": False},
    )
    mfccs = transform(waveform)
    mfcc = torch.mean(mfccs, dim=2)
    return mfcc[0]


def concentrate_mfcc(people, vowel_path):
    mfccs = []
    for person in people:
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


counter = {
    'a': 0,
    'e': 0,
    'i': 0,
    'o': 0,
    'u': 0,
}
people = os.listdir(os.path.join(os.path.dirname(__file__), 'train_clean'))
mfcc_clusters = {
    'a': concentrate_mfcc(people, 'a.wav'),
    'e': concentrate_mfcc(people, 'e.wav'),
    'i': concentrate_mfcc(people, 'i.wav'),
    'o': concentrate_mfcc(people, 'o.wav'),
    'u': concentrate_mfcc(people, 'u.wav'),
}


def run(total):
    data = {
        'a': kmeans(num_clusters=NUM_CLUSTER, X=mfcc_clusters['a'])[1],
        'e': kmeans(num_clusters=NUM_CLUSTER, X=mfcc_clusters['e'])[1],
        'i': kmeans(num_clusters=NUM_CLUSTER, X=mfcc_clusters['i'])[1],
        'o': kmeans(num_clusters=NUM_CLUSTER, X=mfcc_clusters['o'])[1],
        'u': kmeans(num_clusters=NUM_CLUSTER, X=mfcc_clusters['u'])[1],
    }

    people = os.listdir(os.path.join(os.path.dirname(__file__), 'test_clean'))

    result = {
        'a': counter.copy(),
        'e': counter.copy(),
        'i': counter.copy(),
        'o': counter.copy(),
        'u': counter.copy(),
    }
    for person in people:
        result['a'][predict_one(data, person, 'a.wav')] += 1
        result['e'][predict_one(data, person, 'e.wav')] += 1
        result['i'][predict_one(data, person, 'i.wav')] += 1
        result['o'][predict_one(data, person, 'o.wav')] += 1
        result['u'][predict_one(data, person, 'u.wav')] += 1
    total['a'] += result['a']['a']
    total['e'] += result['e']['e']
    total['i'] += result['i']['i']
    total['o'] += result['o']['o']
    total['u'] += result['u']['u']


total = counter.copy()
loop = 15
for i in tqdm(range(loop)):
    run(total)
for key in total.keys():
    total[key] /= loop

output = os.path.join(os.path.dirname(__file__), 'average.csv')
# Write result to CSV
with open(output, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write header row
    csv_writer.writerow(total.keys())

    csv_writer.writerow(total.values())
