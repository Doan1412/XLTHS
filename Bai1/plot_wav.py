import matplotlib.pyplot as plt
import torchaudio
import os

raw_wav, sr = torchaudio.load(os.path.join(
    os.path.dirname(__file__), 'raw', '23MTL', 'i.wav'))
clean_wav, sr = torchaudio.load(os.path.join(
    os.path.dirname(__file__), 'clean', '23MTL', 'i.wav'))
plt.subplot(2, 1, 1)
plt.plot(raw_wav.t().numpy())
plt.subplot(2, 1, 2)
plt.plot(clean_wav.t().numpy())
plt.show()
