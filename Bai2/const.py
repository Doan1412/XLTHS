import torch
from tqdm import tqdm
import os
import torchaudio
import csv
import numpy as np
from scipy.spatial.distance import euclidean

N_FFT = 2048
FRAME_SIZE = 480
HOP_SIZE = 160


print(str(FRAME_SIZE)+" " + str(HOP_SIZE))

#Trích xuất vector FFT của 1 khung tín hiệu với số chiều là
def to_fft(frame):
    
    #Sử dụng cửa sổ Hann để tăng độ chính xác 
    hamming = torch.hann_window(frame.size(1)).unsqueeze(0)
    rectangular_window = torch.ones(frame.size(1)).unsqueeze(0)
    
    #Nhân khung âm thanh với cửa sổ Hann để giảm thiểu các hiện tượng biên.
    frame = torch.mul(frame, hamming) #mul() nhân từng phần tử tương ứng của hai tensor
    fft = torch.fft.fft(input=frame, n=N_FFT)
    return torch.abs(fft.reshape(-1)[:int(N_FFT/2)])


def to_mean_fft_one(person, set, vowel_path):
    #person: Tên của người cần tính toán.
    # set: Tên của bộ dữ liệu.
    # vowel_path: Đường dẫn đến file âm thanh cụ thể của người đó.
    
    
    #load âm thanh từ đường dẫn được chỉ định.
    waveform, sample_rate = torchaudio.load(os.path.join(
        os.path.dirname(__file__), set, person, vowel_path)) 
    #Xác định chiều dài của vùng chứa nguyên âm tìm được ở bước 1
    wave_len = waveform.size(1) 
    
    # Khoảng ổn định
    startIndex = int(wave_len/3)
    # endIndex = int(startIndex*2) 
    endIndex = int(startIndex*2-0.03*sample_rate)
    
    fft_list = []
    for i in range(startIndex, endIndex, HOP_SIZE):
        fft_list.append(to_fft(waveform[:, i:i+FRAME_SIZE]))
        
    #tạo tensor bằng cách xếp chồng các vector fft lên nhau . Kích thước sẽ là (số lượng tensors trong fft_list, kích thước của mỗi tensor)
    stacked_tensors = torch.stack(fft_list, dim=0) 
    #tính trung bình các vector fft
    mean_fft = torch.mean(stacked_tensors, dim=0) 
    
    # trả về giá trị vector đặc trưng cho 1 nguyên âm của 1 người nói
    return mean_fft 


def to_mean_fft_multi(people, set, vowel_path):
    #people: Danh sách các người cần tính toán.
    # set: Tên của bộ dữ liệu.
    # vowel_path: Đường dẫn đến file âm thanh cụ thể của người đó.
    
    fft_mean_list = []
    for person in tqdm(people):
        fft_mean_list.append(to_mean_fft_one(person, set, vowel_path))
    stacked_tensors = torch.stack(fft_mean_list, dim=0)
    mean_fft_multi = torch.mean(stacked_tensors, dim=0)
    return mean_fft_multi.numpy()

