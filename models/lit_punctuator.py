import time

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional import f1_score, precision, recall
import wandb

from config import label2char


class LitPunctuator(pl.LightningModule):
    def __init__(self, punctuator, tokenizer):
        super().__init__()
        # self.save_hyperparameters()
        self.punctuator = punctuator
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask):
        return self.punctuator(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids, labels, attention_mask, labels_mask = batch
        pred = self(input_ids, attention_mask)
        pred = pred.view(-1, pred.shape[2])
        labels = labels.view(-1)
        loss = F.cross_entropy(pred, labels)
        self.log("train_loss", loss)
        self.log("train_f1_all", f1_score(pred, labels), on_step=True, on_epoch=False)
        self.log("train_f1", f1_score(pred, labels, ignore_index=0), on_step=True, on_epoch=False)
        self.log("train_precision", precision(pred, labels, ignore_index=0), on_step=True, on_epoch=False)
        self.log("train_recall", recall(pred, labels, ignore_index=0), on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels, attention_mask, labels_mask = batch
        pred = self(input_ids, attention_mask)
        pred_size = pred.size()
        labels_size = labels.size()
        pred = pred.view(-1, pred.shape[2])
        labels = labels.view(-1)
        val_loss = F.cross_entropy(pred, labels)
        self.log("val_loss", val_loss)
        self.log("val_f1_all", f1_score(pred, labels), on_step=False, on_epoch=True)
        self.log("val_f1", f1_score(pred, labels, ignore_index=0), on_step=False, on_epoch=True)
        self.log("val_precision", precision(pred, labels, ignore_index=0), on_step=False, on_epoch=True)
        self.log("val_recall", recall(pred, labels, ignore_index=0), on_step=False, on_epoch=True)

        if batch_idx == 1:
            columns = ["Reference", "Predicted"]
            table = wandb.Table(columns=columns)
            for i in range(5):
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i], skip_special_tokens=True)
                true_labels = labels.view(labels_size)[i, 1:len(tokens)].cpu().detach().tolist()
                pred_labels = torch.argmax(pred.view(pred_size), dim=-1)[i, 1:len(tokens)].cpu().detach().tolist()

                input_sentence = ""
                pred_sentence = ""
                for token, true_punct, pred_punct in zip(tokens, true_labels, pred_labels):
                    if token.startswith("##"):
                        input_sentence = input_sentence[:-1] + token[2:] + label2char[true_punct]
                        pred_sentence = pred_sentence[:-1] + token[2:] + label2char[pred_punct]
                    else:
                        input_sentence += token + label2char[true_punct]
                        pred_sentence += token + label2char[pred_punct]

                table.add_data(input_sentence, pred_sentence)
            wandb.log({"examples": table})

        return val_loss

    def test_step(self, batch, batch_idx):
        input_ids, labels, attention_mask, labels_mask = batch
        pred = self(input_ids, attention_mask)
        pred = pred.view(-1, pred.shape[2])
        labels = labels.view(-1)

        test_loss = F.cross_entropy(pred, labels)
        self.log("test_loss", test_loss)
        self.log("test_f1_all", f1_score(pred, labels), on_step=False, on_epoch=True)
        self.log("test_f1", f1_score(pred, labels, ignore_index=0), on_step=False, on_epoch=True)
        self.log("test_precision", precision(pred, labels, ignore_index=0), on_step=False, on_epoch=True)
        self.log("test_recall", recall(pred, labels, ignore_index=0), on_step=False, on_epoch=True)

        return pred, labels

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
