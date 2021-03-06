import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from config import char2label


class BasePunctuator(nn.Module):
    def __init__(self, encoder_model_name):
        super(BasePunctuator, self).__init__()

        self.encoder = AutoModel.from_pretrained(encoder_model_name)
        count_params = len(list(self.encoder.parameters()))
        for i, p in enumerate(self.encoder.parameters()):
            if i >= 0: #count_params - 16 * 1:
                p.requires_grad = True
            else:
                p.requires_grad = False

        self.lstm = nn.LSTM(input_size=self.encoder.config.hidden_size, hidden_size=self.encoder.config.hidden_size, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(in_features=self.encoder.config.hidden_size * 2, out_features=len(char2label))

    def forward(self, input_ids, attention_mask):
        embs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        x, _ = self.lstm(embs)
        x = self.linear(x)
        return x
