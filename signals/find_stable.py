import torchaudio
import os

count = 0
sum = 0
CLEAN = os.path.join(os.path.dirname(__file__), 'train_clean')
for person in os.listdir(CLEAN):
    person_dir = os.path.join(CLEAN, person)
    for sound in os.listdir(person_dir):
        info = torchaudio.info(os.path.join(person_dir, sound))
        duration_in_seconds = info.num_frames / info.sample_rate
        sum += duration_in_seconds
        count += 1


print(f"{sum/(3*count):.6f}")
