import re

from config import char2label, label2char


punctuations = list(char2label.keys())
punctuations.remove(' ')


def remove_consec_duplicates(s):
    """
    remove consecutive duplicates from string
    """
    new_s = ""
    prev = ""
    for c in s:
        if c == prev and c in punctuations + [' ']:
            continue
        else:
            new_s += c
            prev = c
    return new_s


def clean_up(text):
    text = text.strip()\
        .replace('\n', ' ')\
        .replace('\'', '"')\
        .replace('«', '\"').replace('»', '\"') \
        .replace('[', '\"').replace(']', '\"') \
        .replace('—', '-') \
        .replace('ё', 'е')\
        .replace('Ё', 'Е')
    text = re.sub(r'[^a-zA-ZА-Яа-я0-9.,!?"()\- ]', '', text)

    if text[-1] not in ['.', ',', '!', '?']:
        text += '.'

    text = remove_consec_duplicates(text)
    split = re.split(r'([ .,!?"()\-]+)', text)
    new_split = list()
    for s in split:
        s = s.strip()
        if len(s) == 0:
            s = ' '
        elif not s.isalnum() and len(s) > 1:
            s = " ".join([c for c in s if c != ' '])
        new_split.append(s)
    text = ''.join(new_split)

    return text


def pad_punct(punct):
    if punct in [',', '.', '!', '?']:
        punct = punct + ' '
    elif punct in ['"', '(', ')']:
        punct = ' ' + punct
    elif punct in ['-']:
        punct = ' ' + punct + ' '

    return punct


def get_text_with_cap_from_predictions(tokens, labels, capitalization):
    sentence = ""
    for token, label, cap in zip(tokens, labels, capitalization):
        punct = pad_punct(label2char[label])

        if token.startswith("##"):
            if cap == 2:
                sentence = sentence[:-len(punct)] + token[2:].upper() + punct
            else:
                sentence = sentence[:-len(punct)] + token[2:] + punct
        else:
            if cap == 1:
                sentence += token.title() + punct
            elif cap == 2:
                sentence += token.upper() + punct
            else:
                sentence += token + punct
    return sentence


def get_text_from_predictions(tokens, labels):
    sentence = ""
    for token, label in zip(tokens, labels):
        punct = pad_punct(label2char[label])

        if token.startswith("##"):
            sentence = sentence[:-len(punct)] + token[2:] + punct
        else:
            sentence += token + punct
    return sentence

