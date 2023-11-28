from const import *


people = os.listdir(os.path.join(os.path.dirname(__file__), 'train_clean'))

#Tạo dict chứa dữ liệu về các vector đặc trưng của từng nguyên âm
data = {
    'a': to_mean_fft_multi(people, 'train_clean', 'a.wav'),
    'e': to_mean_fft_multi(people, 'train_clean', 'e.wav'),
    'i': to_mean_fft_multi(people, 'train_clean', 'i.wav'),
    'o': to_mean_fft_multi(people, 'train_clean', 'o.wav'),
    'u': to_mean_fft_multi(people, 'train_clean', 'u.wav'),
}

output = os.path.join(os.path.dirname(__file__), 'mean.csv')

# Write data to CSV
with open(output, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write header row
    csv_writer.writerow(data.keys())

    # Write data rows
    for row in zip(*data.values()):
        csv_writer.writerow(row)
