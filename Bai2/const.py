import torch
N_FFT = 2048
STABLE_DELTA = 0.1
SAMPLE_RATE = 16000
FRAME_SIZE = int(30*(10**-3)*SAMPLE_RATE)
HOP_SIZE = int(10*(10**-3)*SAMPLE_RATE)
START_TIME = 0.07
START_INDEX = int(START_TIME * SAMPLE_RATE)
END_INDEX = int(START_INDEX + STABLE_DELTA*SAMPLE_RATE)


def to_fft(frame):
    hamming = torch.hann_window(frame.size(1)).unsqueeze(0)
    frame = torch.mul(frame, hamming)
    fft = torch.fft.fft(input=frame, n=N_FFT)
    return torch.abs(fft.reshape(-1)[:int(N_FFT/2)])
