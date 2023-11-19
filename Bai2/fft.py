import numpy as np
import os
import torchaudio
import torch
import matplotlib.pyplot as plt

N_FFT = 2048
STABLE_DELTA = 0.13
SAMPLE_RATE = 16000
FRAME_SIZE = int(30*(10**-3)*SAMPLE_RATE)
HOP_SIZE = int(10*(10**-3)*SAMPLE_RATE)
START_INDEX = int(STABLE_DELTA * SAMPLE_RATE)

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
