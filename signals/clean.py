import os
import torchaudio
import torch
from tqdm import tqdm

CLEAN = os.path.join(os.path.dirname(__file__), 'test_clean')
RAW = os.path.join(os.path.dirname(__file__), 'test_raw')
THRESHOLD = 0.01
SAMPLE_RATE = 16000


# def filter(y, rate):
#     mask = []
#     window_size = int(5*(10**-3)*rate)
#     y_abs = torch.abs(y)
#     y_abs = torch.nn.functional.pad(y_abs, (0, window_size-1))
#     y_mean = y_abs.unfold(0, window_size, 1).max(1).values
#     mask = y_mean > THRESHOLD
#     return mask


def STE_filter(y, rate):
    mask = []
    window_size = int(5*(10**-3)*rate)
    y = torch.nn.functional.pad(y, (0, window_size-1))
    y = y.tolist()
    for i in range(0, len(y) - window_size + 1, 1):
        window_ste = sum(x**2 for x in y[i:i+window_size])
        mask.append(window_ste > THRESHOLD)
    return torch.tensor(mask)


def downsample_mono(path, sr):
    waveform, sample_rate = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=sr)
        waveform = resampler(waveform)
    return sr, waveform


def save_sample(sample, rate, target_dir, fn):
    ext = fn.split('.')[-1]
    fn = '.'.join(fn.split('.')[:-1])
    fullname = f"{fn}.{ext}"
    dst_path = os.path.join(target_dir, fullname)
    if os.path.exists(dst_path):
        return
    torchaudio.save(dst_path, sample, rate, format="wav")


def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)


def split_wavs():
    check_dir(CLEAN)
    classes = os.listdir(RAW)
    for _cls in classes:
        target_dir = os.path.join(CLEAN, _cls)
        check_dir(target_dir)
        src_dir = os.path.join(RAW, _cls)
        for fn in tqdm(os.listdir(src_dir)):
            src_fn = os.path.join(src_dir, fn)
            rate, wav = downsample_mono(src_fn, SAMPLE_RATE)
            mask = STE_filter(wav[0], rate)
            wav = wav[:, mask]
            mask = STE_filter(torch.flip(wav[0], dims=[-1]), rate)
            mask = torch.flip(mask, dims=[-1])
            wav = wav[:, mask]
            save_sample(wav, rate, target_dir, fn)


if __name__ == '__main__':
    split_wavs()
