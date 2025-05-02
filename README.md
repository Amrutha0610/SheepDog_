# SheepDogMain
# SheepDog Enhanced — Robust Fake News Detection
# Overview
This project reproduces and improves the SheepDog model from KDD 2024, which detects fake news by focusing on content rather than style. Modern LLMs can disguise fake content by mimicking journalistic tone, which makes traditional detectors unreliable. SheepDog addresses this via style-agnostic training, LLM-reframings, and fine-grained content attribution.

We extend the original work by:

- Adding style-cleaning preprocessing to reduce reliance on stylistic cues.

- Implementing a dry run pipeline check for robust model debugging.

- Reproducing results and showing accuracy and F1 improvements.

# improved Features
- Content Filtering: Replaces clickbait phrases (e.g., "BREAKING" → "[BREAKING]"), normalizes punctuation, and converts ALL CAPS to neutral casing. Implemented via clean_article() to reduce style bias in training.
- Dry Run Check:Before training, the dry_run_check() ensures your data, tokenizer, and batch sizes work as expected — saving time and debugging effort.
- Reproducible Evaluation:The script compare_results.py parses training logs and compares metrics like Accuracy, Precision, Recall, and F1
# Setup
# Requirements
- python==3.7
- torch==1.10.0
- transformers==4.13.0
- numpy==1.22.4
- Create a Python 3.7 environment and install packages via pip install -r requirements.txt.
# How to Run
1. Navigate to the extracted folder or clone the repo
   - git clone https://git.txstate.edu/vma61/SheepDogMain.git
2. Run a dry check (optional but recommended)
   "python sheepdog_train.py --use_filtering"
3. Start Training
  "sh train.sh"
4. Compare Original vs Modified Results
   "python compare_results.py"
# Contributors
- Amrutha Damera: Content filtering, training improvements, evaluation
- Sakshith – Dry run checker, pipeline reliability

