import numpy as np
import os
import torchaudio
import torch
import matplotlib.pyplot as plt
from const import *


waveform, sample_rate = torchaudio.load(os.path.join(
    os.path.dirname(__file__), 'train_clean', '23MTL', 'a.wav'))
#Xác định chiều dài của vùng chứa nguyên âm tìm được ở bước 1
wave_len = waveform.size(1) 

# Khoảng ổn định
startIndex = int(wave_len/3)
endIndex = int(startIndex*2-0.03*sample_rate)

plt.figure(figsize=(10, 5))
plt.plot(waveform[:, startIndex:endIndex].t().numpy())
plt.title('Stable Region of the Signal')
plt.ylabel('Amplitude')
plt.xlabel('Sample')
plt.show()
