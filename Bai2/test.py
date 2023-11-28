from const import *
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Dự đoán nguyên âm của một người dựa trên đặc trưng phổ FFT trung bình.
def predict_one(data, person, vowel_path):
    dist = {}
    fft = to_mean_fft_one(person, 'test_clean', vowel_path).tolist()
    for label in data.keys():
        dist[label] = euclidean(data[label], fft)
    return min(dist, key=lambda k: dist[k])


people = os.listdir(os.path.join(os.path.dirname(__file__), 'test_clean'))

mean_file = os.path.join(os.path.dirname(__file__), 'mean.csv')

# Đọc dữ liệu vector đặc trưng từ file CSV
with open(mean_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    # Read the header row
    header = next(csv_reader)

    # Initialize a dictionary to store the data
    data = {label: [] for label in header}

    # Read the data rows
    for row in csv_reader:
        for i, value in enumerate(row):
            value = float(value)
            data[header[i]].append(value)

# Khởi tạo biến đếm kết quả dự đoán và ma trận kết quả
counter = {
    'a': 0,
    'e': 0,
    'i': 0,
    'o': 0,
    'u': 0,
}
result = {
    'a': counter.copy(),
    'e': counter.copy(),
    'i': counter.copy(),
    'o': counter.copy(),
    'u': counter.copy(),
}

# Dự đoán và lưu kết quả cho từng nguyên âm
start_time = time.time()
for person in people:
    result['a'][predict_one(data, person, 'a.wav')] += 1
    result['e'][predict_one(data, person, 'e.wav')] += 1
    result['i'][predict_one(data, person, 'i.wav')] += 1
    result['o'][predict_one(data, person, 'o.wav')] += 1
    result['u'][predict_one(data, person, 'u.wav')] += 1
end_time = time.time()

# In thời gian thực thi
print(end_time - start_time)
output = os.path.join(os.path.dirname(__file__), 'test.csv')

# Ghi kết quả ra file CSV
with open(output, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    truths = result.keys()
    predicts = counter.keys()
    # Write header row
    csv_writer.writerow(['']+list(predicts))

    # Write result rows
    for truth in truths:
        row = [truth] + list(result[truth].values())
        csv_writer.writerow(row)

# Tính toán độ chính xác tổng thể
correct_predictions = sum(result[vowel][vowel] for vowel in result)
total_predictions = sum(sum(result[vowel].values()) for vowel in result)
overall_accuracy = (correct_predictions / total_predictions) * 100

# In độ chính xác tổng thể
print(f"Overall Accuracy: {overall_accuracy:.2f}%")

# Hiển thị ma trận nhầm lẫn
confusion_matrix = np.array([list(result[truth].values()) for truth in result])
labels = list(result.keys())

sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title('Ma trận nhầm lẫn')

# Hiển thị biểu đồ dự đoán nhãn nguyên âm
vowels = ['a', 'e', 'i', 'o', 'u']
num_people = 21
num_vowels = len(vowels)
binary_matrix = np.zeros((num_people, num_vowels), dtype=int)

for i, person in enumerate(people):
    for j, vowel in enumerate(vowels):
        predicted_vowel = predict_one(data, person, f'{vowel}.wav')
        binary_matrix[i, j] = 1 if predicted_vowel == vowel else 0

plt.figure(figsize=(8, 6))
sns.heatmap(binary_matrix, annot=True, cbar=False, cmap='Set1',
            xticklabels=vowels, yticklabels=people)
plt.title('Dự đoán nhãn nguyên âm')
plt.xlabel('Nguyên âm')
plt.ylabel('Người')
plt.show()
