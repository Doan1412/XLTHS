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


folder_path = os.path.join(os.path.dirname(__file__), 'train_raw', '24FTL')

plt.figure(figsize=(10, 8))

for i, filename in enumerate(os.listdir(folder_path)[:5]):
    waveform, sr = torchaudio.load(os.path.join(folder_path, filename))
    
    ste = STE(waveform.reshape(-1), sr)
    
    threshold = 0.03
    threshold_line = [threshold] * len(ste)
    
    plt.subplot(5, 1, i + 1)
    plt.plot(ste)
    plt.plot(threshold_line, 'r--')
    plt.title(filename)
    plt.xlabel('Frames')
    plt.ylabel('STE')

plt.tight_layout()
plt.show()


# raw_wav, sr = torchaudio.load(os.path.join(
#     os.path.dirname(__file__), 'train_raw', '30FTN', 'a.wav'))
# clean_wav, sr = torchaudio.load(os.path.join(
#     os.path.dirname(__file__), 'train_clean', '30FTN', 'a.wav'))
# plt.subplot(2, 1, 1)
# plt.plot(STE(raw_wav.reshape(-1), sr))
# plt.subplot(2, 1, 2)
# plt.plot(STE(clean_wav.reshape(-1), sr))
# plt.show()


