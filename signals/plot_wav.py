import matplotlib.pyplot as plt
import torchaudio
import os

raw_wav, sr = torchaudio.load(os.path.join(
    os.path.dirname(__file__), 'train_raw', '30FTN', 'o.wav'))
clean_wav, sr = torchaudio.load(os.path.join(
    os.path.dirname(__file__), 'train_clean', '30FTN', 'o.wav'))

plt.subplot(2, 1, 1)
plt.plot(raw_wav.t().numpy())
plt.subplot(2, 1, 2)
plt.plot(clean_wav.t().numpy())
plt.show()




