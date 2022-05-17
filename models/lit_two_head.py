import time

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional import f1_score, precision, recall
import wandb

from models.lit_punctuator import LitPunctuator
from utils.text import get_text_with_cap_from_predictions


class LitTwoHead(LitPunctuator):
    def __init__(self, punctuator, tokenizer):
        super().__init__(punctuator, tokenizer)

    def training_step(self, batch, batch_idx):
        input_ids, labels, attention_mask, labels_mask, capitalization = batch
        pred, pred_cap = self(input_ids, attention_mask)

        pred = pred.view(-1, pred.shape[2])
        labels = labels.view(-1)
        pred_cap = pred_cap.view(-1, pred_cap.shape[2])
        capitalization = capitalization.view(-1)

        loss = F.cross_entropy(pred, labels) + F.cross_entropy(pred_cap, capitalization)

        self.log("train_loss", loss)
        self.log("train_f1_all", f1_score(pred, labels), on_step=True, on_epoch=False)
        self.log("train_f1", f1_score(pred, labels, ignore_index=0), on_step=True, on_epoch=False)
        self.log("train_precision", precision(pred, labels, ignore_index=0), on_step=True, on_epoch=False)
        self.log("train_recall", recall(pred, labels, ignore_index=0), on_step=True, on_epoch=False)

        self.log("train_f1_cap_all", f1_score(pred_cap, capitalization), on_step=True, on_epoch=False)
        self.log("train_f1_cap", f1_score(pred_cap, capitalization, ignore_index=0), on_step=True, on_epoch=False)
        self.log("train_precision_cap", precision(pred_cap, capitalization, ignore_index=0), on_step=True, on_epoch=False)
        self.log("train_recall_cap", recall(pred_cap, capitalization, ignore_index=0), on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels, attention_mask, labels_mask, capitalization = batch
        pred, pred_cap = self(input_ids, attention_mask)

        pred_size = pred.size()
        pred_cap_size = pred_cap.size()
        labels_size = labels.size()
        capitalization_size = capitalization.size()

        pred = pred.view(-1, pred.shape[2])
        labels = labels.view(-1)
        pred_cap = pred_cap.view(-1, pred_cap.shape[2])
        capitalization = capitalization.view(-1)

        val_loss = F.cross_entropy(pred, labels) + F.cross_entropy(pred_cap, capitalization)

        self.log("val_loss", val_loss)
        self.log("val_f1_all", f1_score(pred, labels), on_step=True, on_epoch=False)
        self.log("val_f1", f1_score(pred, labels, ignore_index=0), on_step=True, on_epoch=False)
        self.log("val_precision", precision(pred, labels, ignore_index=0), on_step=True, on_epoch=False)
        self.log("val_recall", recall(pred, labels, ignore_index=0), on_step=True, on_epoch=False)

        self.log("val_f1_cap_all", f1_score(pred_cap, capitalization), on_step=True, on_epoch=False)
        self.log("val_f1_cap", f1_score(pred_cap, capitalization, ignore_index=0), on_step=True, on_epoch=False)
        self.log("val_precision_cap", precision(pred_cap, capitalization, ignore_index=0), on_step=True, on_epoch=False)
        self.log("val_recall_cap", recall(pred_cap, capitalization, ignore_index=0), on_step=True, on_epoch=False)

        if batch_idx == 1:
            columns = ["Reference", "Predicted"]
            table = wandb.Table(columns=columns)
            for i in range(5):
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i], skip_special_tokens=True)
                true_labels = labels.view(labels_size)[i, 1:len(tokens)].cpu().detach().tolist()
                true_capitalization = capitalization.view(capitalization_size)[i, 1:len(tokens)].cpu().detach().tolist()
                pred_labels = torch.argmax(pred.view(pred_size), dim=-1)[i, 1:len(tokens)].cpu().detach().tolist()
                pred_capitalization = torch.argmax(pred_cap.view(pred_cap_size), dim=-1)[i, 1:len(tokens)].cpu().detach().tolist()

                input_sentence = get_text_with_cap_from_predictions(tokens, true_labels, true_capitalization)
                pred_sentence = get_text_with_cap_from_predictions(tokens, pred_labels, pred_capitalization)

                table.add_data(input_sentence, pred_sentence)
            wandb.log({"examples": table})

        return val_loss

    def test_step(self, batch, batch_idx):
        input_ids, labels, attention_mask, labels_mask, capitalization = batch
        pred, pred_cap = self(input_ids, attention_mask)

        pred = pred.view(-1, pred.shape[2])
        labels = labels.view(-1)
        pred_cap = pred_cap.view(-1, pred_cap.shape[2])
        capitalization = capitalization.view(-1)

        test_loss = F.cross_entropy(pred, labels) + F.cross_entropy(pred_cap, capitalization)
        self.log("test_loss", test_loss)
        self.log("test_f1_all", f1_score(pred, labels), on_step=False, on_epoch=True)
        self.log("test_f1", f1_score(pred, labels, ignore_index=0), on_step=False, on_epoch=True)
        self.log("test_precision", precision(pred, labels, ignore_index=0), on_step=False, on_epoch=True)
        self.log("test_recall", recall(pred, labels, ignore_index=0), on_step=False, on_epoch=True)

        self.log("test_f1_cap_all", f1_score(pred_cap, capitalization), on_step=False, on_epoch=True)
        self.log("test_f1_cap", f1_score(pred_cap, capitalization, ignore_index=0), on_step=False, on_epoch=True)
        self.log("test_precision_cap", precision(pred_cap, capitalization, ignore_index=0), on_step=False, on_epoch=True)
        self.log("test_recall_cap", recall(pred_cap, capitalization, ignore_index=0), on_step=False, on_epoch=True)

        return pred, labels, pred_cap
