from torchcrf import CRF
import torch
from torch import nn

from config import char2label


class BasePunctuatorCRF(nn.Module):
    def __init__(self, base_model):
        super(BasePunctuatorCRF, self).__init__()

        self.base_model = base_model
        self.crf = CRF(len(char2label), batch_first=True)

    def log_likelihood(self, x, attn_masks, y):
        x = self.base_model(x, attn_masks)
        attn_masks = attn_masks.byte()
        return -self.crf(x, y, mask=attn_masks, reduction='token_mean')

    def forward(self, input_ids, attention_mask):
        x = self.base_model(input_ids, attention_mask)

        attn_masks = attention_mask.byte()

        dec_out = self.crf.decode(x, mask=attn_masks)

        batch_size, seq_len = input_ids.size()

        y_pred = torch.zeros((batch_size, seq_len, len(char2label)))
        for i in range(batch_size):
            dec = dec_out[i] + [0 for _ in range(seq_len - len(dec_out[i]))]
            for j in range(seq_len):
                y_pred[i, j, dec[j]] = 1

        return y_pred
