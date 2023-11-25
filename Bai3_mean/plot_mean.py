import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from const import *

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
plt.plot(data['a'])
plt.ylabel('a')
plt.subplot(5, 1, 2)
plt.plot(data['e'])
plt.ylabel('e')
plt.subplot(5, 1, 3)
plt.plot(data['i'])
plt.ylabel('i')
plt.subplot(5, 1, 4)
plt.plot(data['o'])
plt.ylabel('o')
plt.subplot(5, 1, 5)
plt.plot(data['u'])
plt.ylabel('u')
plt.show()
