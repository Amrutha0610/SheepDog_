#compare_results.py
#to compare the original SheepDog results with the modified version
import re

def extract_metrics(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    def get_metric(label, group=1):
        match = re.search(label, content)
        return float(match.group(group)) if match else None

    return {
        "Accuracy": get_metric(r'Average acc\.\s*:\s*(\d+\.\d+)'),
        "Precision": get_metric(r'Average Prec / Rec / F1 \(macro\):\s*(\d+\.\d+),'),
        "Recall": get_metric(r'Average Prec / Rec / F1 \(macro\):\s*\d+\.\d+,\s*(\d+\.\d+),'),
        "F1": get_metric(r'Average Prec / Rec / F1 \(macro\):\s*\d+\.\d+,\s*\d+\.\d+,\s*(\d+\.\d+)')
    }


original_log = 'logs/log_politifact_sheepdog.iter10'
filtered_log = 'logs/log_politifact_Pretrained-LM.iter1' 

original = extract_metrics(original_log)
filtered = extract_metrics(filtered_log)

print("\n Summary Comparison Table")
print("-" * 40)
print(f"{'Metric':<10} | {'Original':>10} | {'Filtered':>10}")
print("-" * 40)
for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
    print(f"{metric:<10} | {original[metric]:>10.4f} | {filtered[metric]:>10.4f}")
print("-" * 40)
