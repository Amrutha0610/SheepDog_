import pickle

file_path = 'data/news_articles/politifact_train.pkl'

with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(f"Type: {type(data)}")
print(f"Keys: {list(data.keys())}")

#Try printing first 1–2 examples based on common keys
for key in data:
    print(f"\n--- {key} ---")
    try:
        print(data[key][:2])  #show first 2 items for each key
    except Exception as e:
        print(f"Error printing key '{key}':", e)
