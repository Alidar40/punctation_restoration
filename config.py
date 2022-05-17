import yaml


config = yaml.safe_load(open("config.yaml", "rb"))

if config['wandb']['name'] == 'as_model':
    config['wandb']['name'] = config['model']
    if config['use_crf']:
        config['wandb']['name'] += '_crf'

char2label = {
    ' ': 0,
    ',': 1,
    '.': 2,
    '?': 3,
    '!': 4,
    '-': 5,
    '"': 6,
    '(': 7,
    ')': 8
}

label2char = dict()
for key, value in char2label.items():
    label2char[value] = key

