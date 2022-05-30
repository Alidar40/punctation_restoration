import torch
from torch import optim
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional import f1_score, precision, recall
import wandb

from models.lit_punctuator import LitPunctuator
from utils.text import get_text_with_cap_from_predictions


class LitTwoHead(LitPunctuator):
    def __init__(self, punctuator, tokenizer, use_crf):
        super(LitTwoHead, self).__init__(punctuator, tokenizer, use_crf)

    def training_step(self, batch, batch_idx):
        input_ids, labels, attention_mask, labels_mask, capitalization = batch
        pred, pred_cap = self(input_ids, attention_mask)

        if self.use_crf:
            loss = self.punctuator.log_likelihood(input_ids, attention_mask, labels, capitalization)
            pred = pred.view(-1, pred.shape[2])
            pred_cap = pred_cap.view(-1, pred_cap.shape[2])
            labels = labels.view(-1)
            capitalization = capitalization.view(-1)
        else:
            pred = pred.view(-1, pred.shape[2])
            pred_cap = pred_cap.view(-1, pred_cap.shape[2])
            labels = labels.view(-1)
            capitalization = capitalization.view(-1)
            loss = F.cross_entropy(pred, labels) + F.cross_entropy(pred_cap, capitalization)

        self.log_metrics("train", loss, pred, labels, pred_cap, capitalization)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels, attention_mask, labels_mask, capitalization = batch
        pred, pred_cap = self(input_ids, attention_mask)

        pred_size = pred.size()
        pred_cap_size = pred_cap.size()
        labels_size = labels.size()
        capitalization_size = capitalization.size()

        if self.use_crf:
            val_loss = self.punctuator.log_likelihood(input_ids, attention_mask, labels, capitalization)
            pred = pred.view(-1, pred.shape[2])
            pred_cap = pred_cap.view(-1, pred_cap.shape[2])
            labels = labels.view(-1)
            capitalization = capitalization.view(-1)
        else:
            pred = pred.view(-1, pred.shape[2])
            pred_cap = pred_cap.view(-1, pred_cap.shape[2])
            labels = labels.view(-1)
            capitalization = capitalization.view(-1)
            val_loss = F.cross_entropy(pred, labels) + F.cross_entropy(pred_cap, capitalization)

        self.log_metrics("val", val_loss, pred, labels, pred_cap, capitalization)

        if batch_idx == 1:
            columns = ["Reference", "Predicted"]
            table = wandb.Table(columns=columns)
            for i in range(5):
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i], skip_special_tokens=True)
                true_labels = labels.view(labels_size)[i, 1:1+len(tokens)].cpu().detach().tolist()
                true_capitalization = capitalization.view(capitalization_size)[i, 1:1+len(tokens)].cpu().detach().tolist()
                pred_labels = torch.argmax(pred.view(pred_size), dim=-1)[i, 1:1+len(tokens)].cpu().detach().tolist()
                pred_capitalization = torch.argmax(pred_cap.view(pred_cap_size), dim=-1)[i, 1:1+len(tokens)].cpu().detach().tolist()

                input_sentence = get_text_with_cap_from_predictions(tokens, true_labels, true_capitalization)
                pred_sentence = get_text_with_cap_from_predictions(tokens, pred_labels, pred_capitalization)

                table.add_data(input_sentence, pred_sentence)
            wandb.log({"examples": table})

        return val_loss

    def test_step(self, batch, batch_idx):
        input_ids, labels, attention_mask, labels_mask, capitalization = batch
        pred, pred_cap = self(input_ids, attention_mask)

        if self.use_crf:
            test_loss = self.punctuator.log_likelihood(input_ids, attention_mask, labels, capitalization)
            pred = pred.view(-1, pred.shape[2])
            pred_cap = pred_cap.view(-1, pred_cap.shape[2])
            labels = labels.view(-1)
            capitalization = capitalization.view(-1)
        else:
            pred = pred.view(-1, pred.shape[2])
            pred_cap = pred_cap.view(-1, pred_cap.shape[2])
            labels = labels.view(-1)
            capitalization = capitalization.view(-1)
            test_loss = F.cross_entropy(pred, labels) + F.cross_entropy(pred_cap, capitalization)

        self.log_metrics("test", test_loss, pred, labels, pred_cap, capitalization)

        return pred, labels, pred_cap

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        input_ids, labels, attention_mask, labels_mask, capitalization = batch
        pred, pred_cap = self(input_ids, attention_mask)

        i = 0
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i], skip_special_tokens=True)
        pred_labels = torch.argmax(pred, dim=-1)[i, 1:1+len(tokens)].cpu().detach().tolist()
        pred_capitalization = torch.argmax(pred_cap, dim=-1)[i, 1:1+len(tokens)].cpu().detach().tolist()
        pred_sentence = get_text_with_cap_from_predictions(tokens, pred_labels, pred_capitalization)

        return pred_sentence

    def log_metrics(self, split: str, loss, pred, labels, pred_cap, capitalization):
        if split == "train":
            on_step = True
            on_epoch = False
        else:
            on_step = False
            on_epoch = True
        self.log(f"loss/{split}_loss", loss)
        self.log(f"{split}/punct/f1_all", f1_score(pred, labels), on_step=on_step, on_epoch=on_epoch)
        self.log(f"{split}/punct/f1", f1_score(pred, labels, ignore_index=0), on_step=on_step, on_epoch=on_epoch)
        self.log(f"{split}/punct/precision", precision(pred, labels, ignore_index=0), on_step=on_step, on_epoch=on_epoch)
        self.log(f"{split}/punct/recall", recall(pred, labels, ignore_index=0), on_step=on_step, on_epoch=on_epoch)

        self.log(f"{split}/capitalization/f1_all", f1_score(pred_cap, capitalization), on_step=on_step, on_epoch=on_epoch)
        self.log(f"{split}/capitalization/f1", f1_score(pred_cap, capitalization, ignore_index=0), on_step=on_step, on_epoch=on_epoch)
        self.log(f"{split}/capitalization/precision", precision(pred_cap, capitalization, ignore_index=0), on_step=on_step, on_epoch=on_epoch)
        self.log(f"{split}/capitalization/recall", recall(pred_cap, capitalization, ignore_index=0), on_step=on_step, on_epoch=on_epoch)
