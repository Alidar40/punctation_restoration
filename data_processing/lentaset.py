import gc
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

from data_processing.augmentations import augmentations
from utils.text import clean_up, remove_consec_duplicates
from config import config, char2label


SEED = config["seed"]
DATASET_PATH = config["dataset_path"]
SEQUENCE_LEN = config["sequence_len"]
AUGMENT_RATE = config["augment_rate"]
BATCH_SIZE = config["batch_size"]
NUM_WORKERS = config["num_workers"]
DEV_MODE = config["dev_mode"]
CHUNK_SIZE = config["chunk_size"]


class LentaSet(Dataset):
    def __init__(self, texts, tokenizer, sequence_len, augment_rate, is_train, augment_type="none"):
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len
        self.is_train = is_train
        self.augment_rate = augment_rate
        self.augment_type = augment_type

        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x, y, attention_mask, y_mask, capitalization = self.parse_text(self.texts[idx])

        if self.is_train and self.augment_rate > 0:
            x, y, attn_mask, y_mask, capitalization = self.augment(x, y, y_mask, capitalization)

        x = torch.tensor(x)
        y = torch.tensor(y)
        attention_mask = torch.tensor(attention_mask)
        y_mask = torch.tensor(y_mask)
        capitalization = torch.tensor(capitalization)

        return x, y, attention_mask, y_mask, capitalization

    def get_raw_text(self, idx):
        text = re.sub(r'[^a-zA-ZА-Яа-яёЁ0-9 ]', ' ', self.texts[idx])
        return remove_consec_duplicates(text).lower()

    def augment(self, x, y, y_mask, capitalization):
        x_aug = []
        y_aug = []
        attention_mask_aug = []
        y_mask_aug = []
        capitalization_aug = []
        for i in range(len(x)):
            r = np.random.rand()
            if r < self.augment_rate:
                unk_token = self.tokenizer('[UNK]', add_special_tokens=False)['input_ids'][0]
                vocab_size = self.tokenizer.vocab_size
                augmentations[self.augment_type](x, y, y_mask, capitalization, x_aug, y_aug, y_mask_aug, capitalization_aug, i, vocab_size, unk_token)
            else:
                x_aug.append(x[i])
                y_aug.append(y[i])
                y_mask_aug.append(y_mask[i])
                capitalization_aug.append(capitalization[i])

        if len(x_aug) > self.sequence_len:
            # len increased due to insert
            x_aug = x_aug[0:self.sequence_len]
            y_aug = y_aug[0:self.sequence_len]
            y_mask_aug = y_mask_aug[0:self.sequence_len]
            capitalization_aug = capitalization_aug[0:self.sequence_len]
        elif len(x_aug) < self.sequence_len:
            # len decreased due to delete
            x_aug = x_aug + self.tokenizer('[PAD]', add_special_tokens=False)['input_ids'] * (self.sequence_len - len(x_aug))
            y_aug = y_aug + [0 for _ in range(self.sequence_len - len(y_aug))]
            y_mask_aug = y_mask_aug + [0 for _ in range(self.sequence_len - len(y_mask_aug))]
            attention_mask_aug = attention_mask_aug + [0 for _ in range(self.sequence_len - len(attention_mask_aug))]
            capitalization_aug = capitalization_aug + [0 for _ in range(self.sequence_len - len(capitalization_aug))]

        return x_aug, y_aug, attention_mask_aug, y_mask_aug, capitalization_aug

    def parse_text(self, text):
        text = clean_up(text)

        split = re.split(r'([ .,!?"()\-])', text)

        x = self.tokenizer('[CLS]', add_special_tokens=False)['input_ids']
        attention_mask = [1]
        y = [0]
        y_mask = [1]
        capitalization = [0]
        for i in range(0, len(split), 2):
            word = split[i]
            if i + 1 >= len(split):
                punct_mark = text[-1]
            else:
                punct_mark = label_strip(split[i + 1])
            if word == '':
                word = ' '

            if word[0].isupper():
                capitalization_label = 1
            elif word.isupper():
                capitalization_label = 2
            else:
                capitalization_label = 0

            encoded = self.tokenizer(word.lower(), add_special_tokens=False)
            tokenized_word = encoded['input_ids']
            num_subtokens = len(tokenized_word)

            if len(x) + num_subtokens >= self.sequence_len:
                break

            x.extend(tokenized_word)
            attention_mask.extend(encoded['attention_mask'])
            y.extend([char2label[punct_mark]] * num_subtokens)
            y_mask.extend([1] * num_subtokens)
            capitalization.extend([capitalization_label] * num_subtokens)

        x.extend(self.tokenizer('[SEP]', add_special_tokens=False)['input_ids'])
        y.append(0)
        attention_mask.append(1)
        y_mask.append(1)
        capitalization.append(0)
        if len(x) < self.sequence_len:
            x = x + self.tokenizer('[PAD]', add_special_tokens=False)['input_ids'] * (self.sequence_len - len(x))
            y = y + [0 for _ in range(self.sequence_len - len(y))]
            y_mask = y_mask + [0 for _ in range(self.sequence_len - len(y_mask))]
            attention_mask = attention_mask + [0 for _ in range(self.sequence_len - len(attention_mask))]
            capitalization = capitalization + [0 for _ in range(self.sequence_len - len(capitalization))]

        return x, y, attention_mask, y_mask, capitalization


def label_strip(label):
    label = label.strip()
    if label == '':
        return ' '
    return label


def get_dataloaders(tokenizer):
    texts = np.array([])
    # texts = list()
    for chunk in pd.read_csv(DATASET_PATH, chunksize=CHUNK_SIZE):
        texts = np.concatenate((texts, chunk["text"].to_list()))
        # texts.extend(chunk["text"].to_list())
        if DEV_MODE:
            break
        gc.collect()
    # texts = np.array(texts)

    indexes = list(range(0, len(texts)))

    train_indexes, val_indexes = train_test_split(indexes, test_size=0.2, shuffle=True, random_state=SEED)
    val_indexes, test_indexes = train_test_split(val_indexes, test_size=0.5, shuffle=True, random_state=SEED)

    train_dataset = LentaSet(texts[train_indexes], tokenizer, SEQUENCE_LEN, AUGMENT_RATE, augment_type="all", is_train=True)
    val_dataset = LentaSet(texts[val_indexes], tokenizer, SEQUENCE_LEN, AUGMENT_RATE, is_train=False)
    test_dataset = LentaSet(texts[test_indexes], tokenizer, SEQUENCE_LEN, AUGMENT_RATE, is_train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    return train_dataloader, val_dataloader, test_dataloader
