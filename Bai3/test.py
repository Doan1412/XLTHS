import os
import torchaudio
import torch
import csv
import numpy as np
from const import *
from collections import OrderedDict
from kmeans_pytorch import kmeans

num_clusters = 0


def to_mfcc(waveform):
    transform = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={"n_fft": N_FFT, "win_length": WIN_SIZE, "hop_length": HOP_SIZE,
                   "n_mels": 23, "center": True},
    )
    mfccs = transform(waveform)
    mfcc = torch.mean(mfccs, dim=2)
    return mfcc[0]


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


people = os.listdir(os.path.join(os.path.dirname(__file__), 'test_clean'))

mean_file = os.path.join(os.path.dirname(__file__), 'kmean.csv')

with open(mean_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    # Read the header row
    header = next(csv_reader)
    num_clusters = int(len(header)/5)
    labels = list(OrderedDict.fromkeys(header))

    # Initialize a dictionary to store the data
    data = {label: [[] for _ in range(num_clusters)] for label in labels}

    # Read the data rows
    for row in csv_reader:
        for i, value in enumerate(row):
            value = float(value)
            data[header[i]][i % num_clusters].append(value)

counter = {
    'a': 0,
    'e': 0,
    'i': 0,
    'o': 0,
    'u': 0,
}
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

output = os.path.join(os.path.dirname(__file__), 'test.csv')
# Write result to CSV
with open(output, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    truths = result.keys()
    predicts = counter.keys()
    # Write header row
    csv_writer.writerow(['']+list(predicts))

    # Write result rows
    for truth in truths:
        row = [truth] + list(result[truth].values())
        csv_writer.writerow(row)
