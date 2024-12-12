import re
from collections import Counter

def tokenize_text(text):
    """Tokenize the input text into a list of tokens."""
    return re.findall(r'\w+(?:[-_]\w+)*|\S', text)

# count all labels in the test dataset
def count_labels(data):
    label_count = {}
    for example in data:
        ner_data = example.get("ner", [])
        for entity in ner_data:
            label = entity[2]  # Assuming the label is the third element in the entity list
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
    return label_count


# get list of tokenized text
def get_tokenized_text(data):
    return [data[i]['tokenized_text'] for i in range(len(data))]

# get list of text lengths
def get_text_lengths(data):
    return [len(data[i]['tokenized_text']) for i in range(len(data))]

# join tokens in 'tokeinzed_text'
def join_tokens(tokens):
    # code from Gliner_Studio: https://colab.research.google.com/drive/1Kl3TrpiGBpMw569ek_AL6Ee3uqBK-Gfw?usp=sharing
    # Joining tokens with space, but handling special characters correctly
    text = ""
    for token in tokens:
        if token in {",", ".", "!", "?", ":", ";", "..."}:
            text = text.rstrip() + token
        else:
            text += " " + token
    return text.strip()

# get all labels in the dataset
def get_all_labels(data):
    all_labels = []
    for example in data:
        ner_data = example.get("ner", [])
        for entity in ner_data:
            label = entity[2]  # Assuming the label is the third element in the entity list
            if label not in all_labels:
                all_labels.append(label)
    return all_labels

# view which text is pointed at by the 'ner' key in a tokenized text
def view(tokenized_text, ners):
    for ner in ners:
        start, end, label = ner
        print(f"{label}: {tokenized_text[start:end+1]}")

# view which text, joined using join_tokens, is pointed at by the 'ner' key in a tokenized text
def view_joined(tokenized_text, ners):
    for ner in ners:
        start, end, label = ner
        print(f"{label}: {join_tokens(tokenized_text[start:end+1])}")

# view which text, joined using join_tokens, is pointed at by the 'ner' key in a tokenized text
def view_joined_from_dict(tokens_ners_pair : dict):
    tokenized_text = tokens_ners_pair.get('tokenized_text',None)
    ners = tokens_ners_pair.get('ner',None)

    if tokenized_text and ners:
         view_joined(tokenized_text,ners)
    else:
        raise KeyError("Key 'tokenized_text' and or 'ner' missing")


def chunk_data(sample, chunk_size, offset):
    tokenized_text = sample['tokenized_text']
    ners = sample['ner']
    new_data = []
    for i in range(0, len(tokenized_text), chunk_size):
        new_data_dict = {}
        start = i
        end = i + chunk_size + offset
        new_data_dict['tokenized_text'] = tokenized_text[start:end]
        new_data_dict['ner'] = []
        for ner_label in ners:
            if ner_label[0] >= i and ner_label[0] < end:
                new_data_dict['ner'].append([ner_label[0] - i, ner_label[1] - i, ner_label[2]])
        if len(new_data_dict['ner']) > 0:
            new_data.append(new_data_dict)
    return new_data

def chunk_dataset(data, chunk_size, offset):
    new_data = []
    for i, sample in enumerate(data):
        print(f"Processing sample {i + 1} of {len(data)}", end='\r')
        new_data.extend(chunk_data(sample, chunk_size, offset))
    print()
    return new_data


# convert NER labels to citation and law from the dataset rcds/swiss_citation_extraction
def convert_ner_labels(ner_labels):
    ner = []
    i = 0
    while i < len(ner_labels):
        if ner_labels[i] == 1:
            start = i
            while i<len(ner_labels) - 1 and ner_labels[i+1] in [1,2]:
                i += 1
            ner.append([start, i, "citation"])
        elif ner_labels[i] == 3:
            start = i
            while i<len(ner_labels) - 1 and ner_labels[i+1] in [3,4]:
                i += 1
            ner.append([start, i, "law"])
        else:
            i += 1

    return ner

# create test data for each specific labels (filter data points for a specific label, throw out examples that do not have the label)
def create_test_data_for_label(data, label):
    test_data_label = []
    for example in data:
        ner_data = example.get("ner", [])
        new_ners = []
        for entity in ner_data:
            if entity[2] == label:
                new_ners.append(entity)
        if len(new_ners) > 0:
            test_data_label.append({"tokenized_text": example["tokenized_text"], "ner": new_ners})
    return test_data_label

def get_label_dict(data : list[dict],most_common = 0):
    output =  Counter([ner[2] for sample in data for ner in sample['ner']])

    if most_common:
        return output.most_common(most_common)
    
    else:
        return output
    
def exclude_samples_by_labels(sample,labels_to_exclude : list) -> bool:

    ner_labels = set([x[2] for x in sample['ner']])

    return set(labels_to_exclude) & ner_labels
