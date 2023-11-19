import os
import torchaudio
import torch
import csv
import numpy as np
from const import *


def to_fft(frame):
    fft = torch.fft.fft(input=frame, n=N_FFT)
    return fft.reshape(-1)[:int(N_FFT/2)]


def to_mean_fft_one(person, vowel_path):
    waveform, sample_rate = torchaudio.load(os.path.join(
        os.path.dirname(__file__), 'test_clean', person, vowel_path))
    fft_list = []
    for i in range(START_INDEX, 2*START_INDEX, HOP_SIZE):
        fft_list.append(to_fft(waveform[:, i:i+FRAME_SIZE]))
    stacked_tensors = torch.stack(fft_list, dim=0)
    mean_fft = torch.mean(stacked_tensors, dim=0)
    return mean_fft


# def euclidean(v1, v2):
#     return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5

def euclidean_distance_complex(list1, list2):
    # Convert lists to numpy arrays
    array1 = np.array(list1)
    array2 = np.array(list2)

    # Calculate the squared differences of real and imaginary parts
    squared_diff_real = np.square(array1.real - array2.real)
    squared_diff_imag = np.square(array1.imag - array2.imag)

    # Sum the squared differences
    sum_squared_diff = np.sum(squared_diff_real + squared_diff_imag)

    # Calculate the square root to get the Euclidean distance
    distance = np.sqrt(sum_squared_diff)

    return distance


def predict_one(data, person, vowel_path):
    dist = {}
    fft = to_mean_fft_one(person, vowel_path).tolist()
    for label in data.keys():
        dist[label] = euclidean_distance_complex(data[label], fft)
    return min(dist, key=lambda k: dist[k])


people = os.listdir(os.path.join(os.path.dirname(__file__), 'test_clean'))

mean_file = os.path.join(os.path.dirname(__file__), 'mean.csv')

with open(mean_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    # Read the header row
    header = next(csv_reader)

    # Initialize a dictionary to store the data
    data = {label: [] for label in header}

    # Read the data rows
    for row in csv_reader:
        for i, value in enumerate(row):
            data[header[i]].append(complex(value))
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
