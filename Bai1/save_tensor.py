import torchaudio
import os
import numpy as np

clean_wav, sr = torchaudio.load(os.path.join(
    os.path.dirname(__file__), 'clean', '23MTL', 'i.wav'))

# Convert PyTorch tensor to NumPy array
numpy_array = clean_wav.t().numpy()

# Save NumPy array to CSV
np.savetxt(os.path.join(os.path.dirname(__file__), 'output.csv'),
           numpy_array, delimiter=',', fmt='%f')
