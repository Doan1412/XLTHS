import numpy as np
import os
import torchaudio
import torch
import matplotlib.pyplot as plt
from const import *


waveform, sample_rate = torchaudio.load(os.path.join(
    os.path.dirname(__file__), 'train_clean', '23MTL', 'a.wav'))

fft_list = []

wave_len = waveform.size(1)
startIndex = int(wave_len/3)
endIndex = int(startIndex*2-0.03*SAMPLE_RATE)
for i in range(startIndex, endIndex, HOP_SIZE):
    fft_list.append(to_fft(waveform[:, i:i+FRAME_SIZE]))

stacked_tensors = torch.stack(fft_list, dim=0)
average_fft = torch.mean(stacked_tensors, dim=0)
# plt.plot(frame.t().numpy())
# plt.show()

# np.savetxt(os.path.join(os.path.dirname(__file__), 'output.csv'),
#            fft.t(), delimiter=',', fmt='%.2f%+.2fj')

k = np.arange(1, N_FFT, 1)
frequencies = k * SAMPLE_RATE / N_FFT
plt.plot(frequencies[:int(N_FFT/2)], torch.abs(average_fft))
plt.title('One-sided Average Linear Magnitude Spectrum')
plt.xlabel('Frequency Axis (Hz)')
plt.ylabel('Magnitude')
plt.show()
