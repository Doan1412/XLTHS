import numpy as np
import os
import torchaudio
import torch
import matplotlib.pyplot as plt
from const import *


waveform, sample_rate = torchaudio.load(os.path.join(
    os.path.dirname(__file__), 'train_clean', '23MTL', 'a.wav'))

frame = waveform[:, START_INDEX:START_INDEX+FRAME_SIZE]
# plt.plot(frame.t().numpy())
# plt.show()

fft = torch.fft.fft(input=frame, n=N_FFT)
# np.savetxt(os.path.join(os.path.dirname(__file__), 'output.csv'),
#            fft.t(), delimiter=',', fmt='%.2f%+.2fj')

k = np.arange(1, N_FFT, 1)
frequencies = k * SAMPLE_RATE / N_FFT
plt.plot(frequencies[:int(N_FFT/2)], torch.abs(fft.reshape(-1)[:int(N_FFT/2)]))
plt.title('One-sided Linear Magnitude Spectrum')
plt.xlabel('Frequency Axis (Hz)')
plt.ylabel('Magnitude')
plt.show()
