# SheepDogMain
# SheepDog Enhanced — Robust Fake News Detection
## Overview
This project reproduces and improves the SheepDog model from KDD 2024, which detects fake news by focusing on content rather than style. Modern LLMs can disguise fake content by mimicking journalistic tone, which makes traditional detectors unreliable. SheepDog addresses this via style-agnostic training, LLM-reframings, and fine-grained content attribution.

We extend the original work by:

- Adding style-cleaning preprocessing to reduce reliance on stylistic cues.

- Implementing a dry run pipeline check for robust model debugging.

- Reproducing results and showing accuracy and F1 improvements.

## improved Features
- Content Filtering: Replaces clickbait phrases (e.g., "BREAKING" → "[BREAKING]"), normalizes punctuation, and converts ALL CAPS to neutral casing. Implemented via clean_article() to reduce style bias in training.
- Dry Run Check: Before training, the dry_run_check() ensures your data, tokenizer, and batch sizes work as expected — saving time and debugging effort.
- Reproducible Evaluation: The script compare_results.py parses training logs and compares metrics like Accuracy, Precision, Recall, and F1
## Project Stucture
    
    ├── sheepdog_train.py             # Main training script (with our improvements)
    ├── utils/
    │   ├── load_data.py              # Data loaders and preprocessing
    ├── logs/
    │   ├── log_politifact_sheepdog  # Original logs
    │   └── log_politifact_Pretrained-LM  # Improved logs
    ├── data/
    │   ├── news_articles/            # Original and sampled datasets
    │   └── reframings/              # Style-based augmented data
    ├── checkpoints/                 # Saved model files
    ├── train.sh                     # Shell script to start training
    ├── compare_results.py           # Compares original vs improved model results
    ├── create_small_dataset.py      # Creates small datasets for debugging
    ├── inspect_dataset.py           # Helps inspect data structure
    └── README.md

## Setup
### Requirements
- python==3.7
- torch==1.10.0
- transformers==4.13.0
- numpy==1.22.4  
- install them with:
  ```bash
  pip install -r requirements.txt
   
### How to Run
1. Navigate to the extracted folder or clone the repo
   ```bash
   git clone https://git.txstate.edu/vma61/SheepDogMain.git
   cd SheepDogMain
2. Create a vitual environment
     ```bash
     python -m venv venv
     source env/bin/activate or
     venv\Scripts\activate # for windows
4. Run a dry check 
    ```bash
    python sheepdog_train.py --use_filtering
5. Start Training
    ```bash
    sh train.sh
6. Compare Original vs Modified Results
    ```bash
    python compare_results.py


