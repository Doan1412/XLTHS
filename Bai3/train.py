import numpy as np
import matplotlib.pyplot as plt
import os
import torchaudio
import torch
import csv
from kmeans_pytorch import kmeans
from const import *


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


def concentrate_mfcc(people, vowel_path):
    mfccs = []
    for person in people:
        waveform, sample_rate = torchaudio.load(os.path.join(
            os.path.dirname(__file__), 'train_clean', person, vowel_path))
        mfccs.append(to_mfcc(waveform[:, START_INDEX:END_INDEX]))
    return torch.stack(mfccs)


# def k_means_loop(num_clusters, X, loop=1):
#     k_means = [kmeans(num_clusters=num_clusters, X=X)[1]
#                for _ in range(loop)]
#     stack_tensors = torch.stack(k_means, dim=0)
#     result = torch.mean(stack_tensors, dim=0)
#     return result


people = os.listdir(os.path.join(os.path.dirname(__file__), 'train_clean'))

mfcc_clusters = {
    'a': concentrate_mfcc(people, 'a.wav'),
    'e': concentrate_mfcc(people, 'e.wav'),
    'i': concentrate_mfcc(people, 'i.wav'),
    'o': concentrate_mfcc(people, 'o.wav'),
    'u': concentrate_mfcc(people, 'u.wav'),
}

mfcc_means = {
    'a': kmeans(num_clusters=NUM_CLUSTER, X=mfcc_clusters['a'])[1],
    'e': kmeans(num_clusters=NUM_CLUSTER, X=mfcc_clusters['e'])[1],
    'i': kmeans(num_clusters=NUM_CLUSTER, X=mfcc_clusters['i'])[1],
    'o': kmeans(num_clusters=NUM_CLUSTER, X=mfcc_clusters['o'])[1],
    'u': kmeans(num_clusters=NUM_CLUSTER, X=mfcc_clusters['u'])[1],
}

output = os.path.join(os.path.dirname(__file__), 'kmean.csv')

# Write data to CSV
with open(output, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write header row
    csv_writer.writerow([item for item in mfcc_means.keys()
                        for _ in range(NUM_CLUSTER)])
    data = [value for mfcc_mean in mfcc_means.values()
            for value in mfcc_mean.tolist()]

    # Write data rows
    for row in zip(*data):
        csv_writer.writerow(row)