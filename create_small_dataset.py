import pickle
import random
import os

dataset_names = ['politifact', 'gossipcop', 'lun']
base_path = 'data/news_articles'

for name in dataset_names:
    for split in ['train', 'test']:
        file_path = os.path.join(base_path, f"{name}_{split}.pkl")
        small_path = os.path.join(base_path, f"{name}_{split}_small.pkl")

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # Safely check for keys
        if 'news' not in data or 'labels' not in data:
            print(f"Skipping {file_path} — missing 'news' or 'labels'")
            continue

        combined = list(zip(data['news'], data['labels']))
        sample_size = 50 if split == 'train' else 20
        sampled = random.sample(combined, min(sample_size, len(combined)))

        news, labels = zip(*sampled)
        small_data = {'news': list(news), 'labels': list(labels)}

        with open(small_path, 'wb') as f:
            pickle.dump(small_data, f)

        print(f"Saved {small_path} with {len(news)} samples.")
