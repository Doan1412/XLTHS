import matplotlib.pyplot as plt
import torch
import torchaudio
import os


def STE(y, rate):
    ste = []
    window_size = int(30*(10**-3)*rate)
    y = torch.nn.functional.pad(y, (0, window_size-1))
    y = y.tolist()
    for i in range(0, len(y) - window_size + 1, 1):
        window_ste = sum(x**2 for x in y[i:i+window_size])
        ste.append(window_ste)
    return ste


raw_wav, sr = torchaudio.load(os.path.join(
    os.path.dirname(__file__), 'train_raw', '40MHS', 'i.wav'))
plt.plot(STE(raw_wav.reshape(-1), sr))
plt.show()
