import pickle
import numpy as np
import re
import os


import os
#------Load Training and Test Articles--------
def load_articles(obj, use_filtering=False):
    print('Dataset: ', obj)
    print("loading news articles")
    # Load small subset training and testing data from pickle files
    train_dict = pickle.load(open(f'data/news_articles/{obj}_train_small.pkl', 'rb'))
    test_dict = pickle.load(open(f'data/news_articles/{obj}_test_small.pkl', 'rb'))
    # Try loading adversarial (style-restyled) test set
    try:
        restyle_path = f'data/adversarial_test/{obj}_test_adv_A.pkl'
        if not os.path.exists(restyle_path):
            raise FileNotFoundError(f"Restyle file not found: {restyle_path}")
        restyle_dict = pickle.load(open(restyle_path, 'rb'))
        x_test_res = restyle_dict['news'] # LLM-restyled articles
    except Exception as e:
        print("Warning: Could not load adversarial test set. Defaulting to regular test set.")
        x_test_res = test_dict['news']  #fallback to normal test data
    # Extract inputs and labels
    x_train, y_train = train_dict['news'], train_dict['labels']
    x_test, y_test = test_dict['news'], test_dict['labels']
    # Ensure x_test_res length matches y_test to avoid DataLoader index errors
    x_test_res = x_test_res[:len(y_test)]
    print("Length of x_test_res:", len(x_test_res))
    print("Length of y_test:", len(y_test))
    # Apply content filtering (if enabled via CLI flag)
    if use_filtering:
        x_train = [clean_article(x) for x in x_train]
        x_test = [clean_article(x) for x in x_test]
        x_test_res = [clean_article(x) for x in x_test_res]

    return x_train, x_test, x_test_res, y_train, y_test


#----------Content Filtering Function-------------
def clean_article(text):
    """
    Cleans up stylistic cues in fake news articles to focus on core content.
    - Replaces clickbait/sensational words with tags
    - Tones down all-caps words
    - Normalizes punctuation and spacing
    """

    #Replace specific sensational keywords with placeholder tags
    replacements = {
        'BREAKING': '[BREAKING]',
        'SHOCKING': '[SENSATIONAL]',
        'UNBELIEVABLE': '[UNCREDIBLE]',
        'CLICK HERE': '[LINK]',
        'YOU WON’T BELIEVE': '[HYPE]'
    }
    for word, tag in replacements.items():
        text = re.sub(rf'\b{re.escape(word)}\b', tag, text, flags=re.IGNORECASE)

    #Tone down ALL CAPS but keep content (e.g., "FAKE" → "Fake")
    def lowercase_caps(match):
        return match.group(0).capitalize()

    text = re.sub(r'\b[A-Z]{4,}\b', lowercase_caps, text)

    #Normalize repeated punctuation but preserve one like "!!!" or "???" to a single character
    text = re.sub(r'[!?.,]{2,}', lambda m: m.group(0)[0], text)

    #Clean up extra whitespace
    return re.sub(r'\s+', ' ', text).strip()

def load_reframing(obj):
    print("loading news augmentations")
    print('Dataset: ', obj)

    restyle_dict_train1_1 = pickle.load(open('data/reframings/' + obj+ '_train_objective.pkl', 'rb'))
    restyle_dict_train1_2 = pickle.load(open('data/reframings/' + obj+ '_train_neutral.pkl', 'rb'))
    restyle_dict_train2_1 = pickle.load(open('data/reframings/' + obj+ '_train_emotionally_triggering.pkl', 'rb'))
    restyle_dict_train2_2 = pickle.load(open('data/reframings/' + obj+ '_train_sensational.pkl', 'rb'))

    finegrain_dict1 = pickle.load(open('data/veracity_attributions/' + obj+ '_fake_standards_objective_emotionally_triggering.pkl', 'rb'))
    finegrain_dict2 = pickle.load(open('data/veracity_attributions/' + obj+ '_fake_standards_neutral_sensational.pkl', 'rb'))

    x_train_res1 = np.array(restyle_dict_train1_1['rewritten'])
    x_train_res1_2 = np.array(restyle_dict_train1_2['rewritten'])
    x_train_res2 = np.array(restyle_dict_train2_1['rewritten'])
    x_train_res2_2 = np.array(restyle_dict_train2_2['rewritten'])

    y_train_fg, y_train_fg_m, y_train_fg_t = finegrain_dict1['orig_fg'], finegrain_dict1['mainstream_fg'], finegrain_dict1['tabloid_fg']
    y_train_fg2, y_train_fg_m2, y_train_fg_t2 = finegrain_dict2['orig_fg'], finegrain_dict2['mainstream_fg'], finegrain_dict2['tabloid_fg']

    replace_idx = np.random.choice(len(x_train_res1), len(x_train_res1) // 2, replace=False)

    x_train_res1[replace_idx] = x_train_res1_2[replace_idx]
    x_train_res2[replace_idx] = x_train_res2_2[replace_idx]
    y_train_fg[replace_idx] = y_train_fg2[replace_idx]
    y_train_fg_m[replace_idx] = y_train_fg_m2[replace_idx]
    y_train_fg_t[replace_idx] = y_train_fg_t2[replace_idx]


    return x_train_res1, x_train_res2, y_train_fg, y_train_fg_m, y_train_fg_t
