from torchcrf import CRF
import torch
from torch import nn

from config import char2label


class TwoHeadCRF(nn.Module):
    def __init__(self, base_model):
        super(TwoHeadCRF, self).__init__()

        self.base_model = base_model
        self.crf = CRF(len(char2label), batch_first=True)
        self.crf_cap = CRF(3, batch_first=True)

    def log_likelihood(self, x, attn_masks, y, y_cap):
        x, x_cap = self.base_model(x, attn_masks)
        attn_masks = attn_masks.byte()
        return -self.crf(x, y, mask=attn_masks, reduction='token_mean')\
               -self.crf_cap(x_cap, y_cap, mask=attn_masks, reduction='token_mean')

    def forward(self, input_ids, attention_mask):
        x, x_cap = self.base_model(input_ids, attention_mask)

        attn_masks = attention_mask.byte()

        dec_out = self.crf.decode(x, mask=attn_masks)
        dec_out_cap = self.crf_cap.decode(x_cap, mask=attn_masks)

        batch_size, seq_len = input_ids.size()

        y_pred = torch.zeros((batch_size, seq_len, len(char2label))).to(x.device)
        y_pred_cap = torch.zeros((batch_size, seq_len, 3)).to(x.device)
        for i in range(batch_size):
            dec = dec_out[i] + [0 for _ in range(seq_len - len(dec_out[i]))]
            dec_cap = dec_out_cap[i] + [0 for _ in range(seq_len - len(dec_out_cap[i]))]
            for j in range(seq_len):
                y_pred[i, j, dec[j]] = 1
                y_pred_cap[i, j, dec_cap[j]] = 1

        return y_pred, y_pred_cap
