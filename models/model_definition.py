import transformers
from tokenizers import pre_tokenizers

from models.base_punctuator import BasePunctuator
from models.base_punctuator_crf import BasePunctuatorCRF
from models.two_head import TwoHead, TwoHeadBaseline, TwoHeadBaselineLinear, TwoHeadLinear
from models.two_head_crf import TwoHeadCRF


def get_model(model_name, encoder_model_name, use_crf):
    if model_name == "base_punctuator":
        model = BasePunctuator(encoder_model_name)
        two_head = False
    elif model_name == "two_head":
        model = TwoHead(encoder_model_name)
        two_head = True
    elif model_name == "two_head_baseline":
        model = TwoHeadBaseline(encoder_model_name)
        two_head = True
    elif model_name == "two_head_baseline_linear":
        model = TwoHeadBaselineLinear(encoder_model_name)
        two_head = True
    elif model_name == "two_head_linear":
        model = TwoHeadLinear(encoder_model_name)
        two_head = True
    else:
        raise NotImplemented("Such a model is not implemented")

    if use_crf and two_head:
        model = TwoHeadCRF(model)
    if use_crf and not two_head:
        model = BasePunctuatorCRF(model)

    return model, two_head


def get_tokenizer(model_name: str) -> transformers.AutoTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    # tokenizer._tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Punctuation(), tokenizer._tokenizer.pre_tokenizer])
    return tokenizer
