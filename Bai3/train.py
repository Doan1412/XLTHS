from const import *


# def STE(y, rate):
#     ste = []
#     window_size = int(25*(10**-3)*rate)
#     hop_size = int(10*(10**-3)*rate)
#     y = torch.nn.functional.pad(y, (0, window_size-1))
#     y = y.numpy()
#     for i in range(0, len(y) - window_size + 1, hop_size):
#         window_ste = np.sum(y[i:i+window_size]**2)
#         ste.append(window_ste)
#     return ste


# def less_variant_segment(waveform):
#     wav = waveform[0].numpy()
#     ste = np.array(STE(waveform[0], SAMPLE_RATE))
#     window_size = int(STABLE_DELTA*SAMPLE_RATE)
#     delta = np.array([])
#     for i in range(0, len(wav) - window_size + 1, 1):
#         mean = np.mean(wav[i:i+window_size])
#         total_variance = np.sum(np.abs(wav[i:i+window_size]-mean))
#         delta = np.append(delta, total_variance)
#     return np.argmin(delta)


# def k_means_loop(num_clusters, X, loop=1):
#     k_means = [kmeans(num_clusters=num_clusters, X=X)[1]
#                for _ in range(loop)]
#     stack_tensors = torch.stack(k_means, dim=0)
#     result = torch.mean(stack_tensors, dim=0)
#     return result


people = os.listdir(os.path.join(os.path.dirname(__file__), 'train_clean'))

mfcc_clusters = {
    'a': concentrate_mfcc(people, 'a.wav'),
    'e': concentrate_mfcc(people, 'e.wav'),
    'i': concentrate_mfcc(people, 'i.wav'),
    'o': concentrate_mfcc(people, 'o.wav'),
    'u': concentrate_mfcc(people, 'u.wav'),
}

mfcc_means = {
    'a': kmeans(num_clusters=NUM_CLUSTER, X=mfcc_clusters['a'])[1],
    'e': kmeans(num_clusters=NUM_CLUSTER, X=mfcc_clusters['e'])[1],
    'i': kmeans(num_clusters=NUM_CLUSTER, X=mfcc_clusters['i'])[1],
    'o': kmeans(num_clusters=NUM_CLUSTER, X=mfcc_clusters['o'])[1],
    'u': kmeans(num_clusters=NUM_CLUSTER, X=mfcc_clusters['u'])[1],
}

output = os.path.join(os.path.dirname(__file__), 'kmean.csv')

# Write data to CSV
with open(output, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write header row
    csv_writer.writerow([item for item in mfcc_means.keys()
                        for _ in range(NUM_CLUSTER)])
    data = [value for mfcc_mean in mfcc_means.values()
            for value in mfcc_mean.tolist()]

    # Write data rows
    for row in zip(*data):
        csv_writer.writerow(row)
