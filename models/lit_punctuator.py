import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from torchmetrics.functional import f1_score, precision, recall
import wandb

from utils.text import get_text_from_predictions


class LitPunctuator(pl.LightningModule):
    def __init__(self, punctuator, tokenizer, use_crf):
        super().__init__()
        # self.save_hyperparameters()
        self.punctuator = punctuator
        self.tokenizer = tokenizer
        self.use_crf = use_crf

    def forward(self, input_ids, attention_mask):
        return self.punctuator(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        input_ids, labels, attention_mask, labels_mask, capitalization = batch
        pred = self(input_ids, attention_mask)

        if self.use_crf:
            loss = self.punctuator.log_likelihood(input_ids, attention_mask, labels)
            pred = pred.view(-1, pred.shape[2])
            labels = labels.view(-1)
        else:
            pred = pred.view(-1, pred.shape[2])
            labels = labels.view(-1)
            loss = F.cross_entropy(pred, labels)

        self.log_metrics("train", loss, pred, labels)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, labels, attention_mask, labels_mask, capitalization = batch
        pred = self(input_ids, attention_mask)

        pred_size = pred.size()
        labels_size = labels.size()

        if self.use_crf:
            val_loss = self.punctuator.log_likelihood(input_ids, attention_mask, labels)
            pred = pred.view(-1, pred.shape[2])
            labels = labels.view(-1)
        else:
            pred = pred.view(-1, pred.shape[2])
            labels = labels.view(-1)
            val_loss = F.cross_entropy(pred, labels)

        self.log_metrics("val", val_loss, pred, labels)

        if batch_idx == 1:
            columns = ["Reference", "Predicted"]
            table = wandb.Table(columns=columns)
            for i in range(5):
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i], skip_special_tokens=True)
                true_labels = labels.view(labels_size)[i, 1:1+len(tokens)].cpu().detach().tolist()
                pred_labels = torch.argmax(pred.view(pred_size), dim=-1)[i, 1:1+len(tokens)].cpu().detach().tolist()

                input_sentence = get_text_from_predictions(tokens, true_labels)
                pred_sentence = get_text_from_predictions(tokens, pred_labels)

                table.add_data(input_sentence, pred_sentence)
            wandb.log({"examples": table})

        return val_loss

    def test_step(self, batch, batch_idx):
        input_ids, labels, attention_mask, labels_mask, capitalization = batch
        pred = self(input_ids, attention_mask)

        if self.use_crf:
            test_loss = self.punctuator.log_likelihood(input_ids, attention_mask, labels)
            pred = pred.view(-1, pred.shape[2])
            labels = labels.view(-1)
        else:
            pred = pred.view(-1, pred.shape[2])
            labels = labels.view(-1)
            test_loss = F.cross_entropy(pred, labels)

        self.log_metrics("test", test_loss, pred, labels)

        return pred, labels

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        input_ids, labels, attention_mask, labels_mask, capitalization = batch
        pred = self(input_ids, attention_mask)

        i = 0
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[i], skip_special_tokens=True)
        pred_labels = torch.argmax(pred, dim=-1)[i, 1:len(tokens)].cpu().detach().tolist()
        pred_sentence = get_text_from_predictions(tokens, pred_labels)

        return pred_sentence

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'loss/train_loss'
        }

    def log_metrics(self, split: str, loss, pred, labels):
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
