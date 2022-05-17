import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from config import char2label


class TwoHead(nn.Module):
    def __init__(self, encoder_model_name):
        super(TwoHead, self).__init__()

        self.encoder = AutoModel.from_pretrained(encoder_model_name)
        count_params = len(list(self.encoder.parameters()))
        for i, p in enumerate(self.encoder.parameters()):
            if i >= 0: #count_params - 16 * 1:
                p.requires_grad = True
            else:
                p.requires_grad = False

        self.lstm = nn.LSTM(input_size=self.encoder.config.hidden_size, hidden_size=self.encoder.config.hidden_size,
                            num_layers=1, bidirectional=True)
        self.linear = nn.Linear(in_features=self.encoder.config.hidden_size * 2, out_features=len(char2label))

        self.lstm_cap = nn.LSTM(input_size=self.encoder.config.hidden_size, hidden_size=self.encoder.config.hidden_size,
                                num_layers=1, bidirectional=True)
        self.linear_cap = nn.Linear(in_features=self.encoder.config.hidden_size * 2, out_features=3)

    def forward(self, input_ids, attention_mask):
        embs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        x, _ = self.lstm(embs)
        x = self.linear(x)

        x_cap, _ = self.lstm_cap(embs)
        x_cap = self.linear_cap(x_cap)

        return x, x_cap


class TwoHeadLinear(nn.Module):
    def __init__(self, encoder_model_name):
        super(TwoHeadLinear, self).__init__()

        self.encoder = AutoModel.from_pretrained(encoder_model_name)
        count_params = len(list(self.encoder.parameters()))
        for i, p in enumerate(self.encoder.parameters()):
            if i >= 0: #count_params - 16 * 1:
                p.requires_grad = True
            else:
                p.requires_grad = False

        self.linear = nn.Sequential(
            nn.Linear(in_features=self.encoder.config.hidden_size, out_features=512),
            nn.Linear(in_features=512, out_features=128),
            nn.Linear(in_features=128, out_features=len(char2label)),
        )

        self.linear_cap = nn.Sequential(
            nn.Linear(in_features=self.encoder.config.hidden_size, out_features=512),
            nn.Linear(in_features=512, out_features=128),
            nn.Linear(in_features=128, out_features=3),
        )

    def forward(self, input_ids, attention_mask):
        embs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.linear(embs)

        x_cap = self.linear_cap(embs)

        return x, x_cap
