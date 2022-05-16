import transformers
from tokenizers import pre_tokenizers

from models.base_punctuator import BasePunctuator


def get_model(model_name, encoder_model_name):
    if model_name == "base_punctuator":
        model = BasePunctuator(encoder_model_name)
    else:
        raise NotImplemented("Such a model is not implemented")

    return model


def get_tokenizer(model_name: str) -> transformers.AutoTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    # tokenizer._tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Punctuation(), tokenizer._tokenizer.pre_tokenizer])
    return tokenizer
