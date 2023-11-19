import os
import csv
import matplotlib.pyplot as plt

mean_file = os.path.join(os.path.dirname(__file__), 'mean.csv')

with open(mean_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    # Read the header row
    header = next(csv_reader)

    # Initialize a dictionary to store the data
    data = {label: [] for label in header}

    # Read the data rows
    for row in csv_reader:
        for i, value in enumerate(row):
            data[header[i]].append(complex(value))

plt.subplot(5, 1, 1)
# Plot the data
plt.plot(data['a'], label='a')
plt.subplot(5, 1, 2)
plt.plot(data['e'], label='e')
plt.subplot(5, 1, 3)
plt.plot(data['i'], label='i')
plt.subplot(5, 1, 4)
plt.plot(data['o'], label='o')
plt.subplot(5, 1, 5)
plt.plot(data['u'], label='u')

plt.show()
