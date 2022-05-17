import transformers
from tokenizers import pre_tokenizers

from models.base_punctuator import BasePunctuator
from models.two_head import TwoHead


def get_model(model_name, encoder_model_name):
    if model_name == "base_punctuator":
        model = BasePunctuator(encoder_model_name)
        two_head = False
    elif model_name == "two_head":
        model = TwoHead(encoder_model_name)
        two_head = True
    elif model_name == "two_head_linear":
        model = TwoHeadLinear(encoder_model_name)
        two_head = True
    else:
        raise NotImplemented("Such a model is not implemented")

    return model, two_head


def get_tokenizer(model_name: str) -> transformers.AutoTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    # tokenizer._tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Punctuation(), tokenizer._tokenizer.pre_tokenizer])
    return tokenizer
