from const import *

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

plt.subplot(5, 1, 1)
for i in range(len(data['a'])):
    plt.plot(data['a'][i])
plt.ylabel('a')
plt.subplot(5, 1, 2)
for i in range(len(data['e'])):
    plt.plot(data['e'][i])
plt.ylabel('e')
plt.subplot(5, 1, 3)
for i in range(len(data['i'])):
    plt.plot(data['i'][i])
plt.ylabel('i')
plt.subplot(5, 1, 4)
for i in range(len(data['o'])):
    plt.plot(data['o'][i])
plt.ylabel('o')
plt.subplot(5, 1, 5)
for i in range(len(data['u'])):
    plt.plot(data['u'][i])
plt.ylabel('u')
plt.show()
