import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import os


def STE(y, rate):
    ste = []
    window_size = int(25*(10**-3)*rate)
    hop_size = int(10*(10**-3)*rate)
    y = torch.nn.functional.pad(y, (0, window_size-1))
    y = y.numpy()
    for i in range(0, len(y) - window_size + 1, hop_size):
        window_ste = np.sum(y[i:i+window_size]**2)
        ste.append(window_ste)
    return ste


def MA(y, rate):
    ma = []
    window_size = int(25*(10**-3)*rate)
    hop_size = int(10*(10**-3)*rate)
    y = torch.nn.functional.pad(y, (0, window_size-1))
    y = y.numpy()
    for i in range(0, len(y) - window_size + 1, hop_size):
        window_ma = np.sum(np.abs(y[i:i+window_size]))
        ma.append(window_ma)
    return ma


raw_wav, sr = torchaudio.load(os.path.join(
    os.path.dirname(__file__), 'train_raw', '40MHS', 'u.wav'))
plt.plot(STE(raw_wav.reshape(-1), sr))
plt.show()
