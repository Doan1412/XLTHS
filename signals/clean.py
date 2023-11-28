import os
import torchaudio
import torch
from tqdm import tqdm

# Đường dẫn đến thư mục chứa dữ liệu âm thanh gốc và dữ liệu đã xử lý
ON_SET = 'train'
CLEAN = os.path.join(os.path.dirname(__file__), f"{ON_SET}_clean")
RAW = os.path.join(os.path.dirname(__file__), f"{ON_SET}_raw")

# Ngưỡng năng lượng 
THRESHOLD = 0.03

# Tần số lấy mẫu
SAMPLE_RATE = 16000

# Lọc tín hiệu âm thanh dựa trên ngưỡng năng lượng STE
def STE_filter(y, rate):
    origin_len = len(y)
    window_size = int(25*(10**-3)*rate)
    hop_size = int(10*(10**-3)*rate) 
    y = torch.nn.functional.pad(y, (0, window_size-1))
    mask = [False]*len(y)
    y = y.tolist()
    for i in range(0, len(y) - window_size + 1, hop_size):
        window_ste = sum(x**2 for x in y[i:i+window_size])
        mask[i:i+hop_size] = [window_ste > THRESHOLD] * hop_size
    return torch.tensor(mask[:origin_len])

# Điều chỉnh tần số lấy mẫu cho âm thanh về một tần số mong muốn
def downsample_mono(path, sr):
    waveform, sample_rate = torchaudio.load(path) 
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=sr)
        waveform = resampler(waveform)
    return sr, waveform

# Lưu trữ tín hiệu âm thanh đã được xử lý
def save_sample(sample, rate, target_dir, fn):
    ext = fn.split('.')[-1]
    fn = '.'.join(fn.split('.')[:-1])
    fullname = f"{fn}.{ext}"
    dst_path = os.path.join(target_dir, fullname)
    if os.path.exists(dst_path):
        return
    torchaudio.save(dst_path, sample, rate, format="wav")

# Kiểm tra và tạo thư mục nếu chưa tồn tại
def check_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)

# Xử lý và tách các tệp âm thanh
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
            save_sample(wav, rate, target_dir, fn)

if __name__ == '__main__':
    split_wavs()
