import time
from const import *

num_clusters = 0


people = os.listdir(os.path.join(os.path.dirname(__file__), 'test_clean'))

mean_file = os.path.join(os.path.dirname(__file__), 'kmean.csv')

with open(mean_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    # Read the header row
    header = next(csv_reader)
    num_clusters = int(len(header)/5)
    labels = list(OrderedDict.fromkeys(header))

    # Initialize a dictionary to store the data
    data = {label: [[] for _ in range(num_clusters)] for label in labels}

    # Read the data rows
    for row in csv_reader:
        for i, value in enumerate(row):
            value = float(value)
            data[header[i]][i % num_clusters].append(value)

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

# Record the start time
start_time = time.time()
for person in people:
    result['a'][predict_one(data, person, 'a.wav')] += 1
    result['e'][predict_one(data, person, 'e.wav')] += 1
    result['i'][predict_one(data, person, 'i.wav')] += 1
    result['o'][predict_one(data, person, 'o.wav')] += 1
    result['u'][predict_one(data, person, 'u.wav')] += 1
# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed Time: {elapsed_time} seconds")
keys = result.keys()
for i in keys:
    for j in keys:
        result[i][j] *= 100/21


output = os.path.join(os.path.dirname(__file__), 'test.csv')
# Write result to CSV
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
