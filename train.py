import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data_processing.lentaset import get_dataloaders
from models.model_definition import get_model, get_tokenizer
from models.lit_punctuator import LitPunctuator
from models.lit_two_head import LitTwoHead
from utils.initialization import seed_everything
from config import config

if __name__ == "__main__":
    SEED = config["seed"]
    MODEL = config["model"]
    USE_CRF = config["use_crf"]
    ENCODER = config["encoder"]
    CONTINUE_FROM_CKPT = config["continue_from_ckpt"]
    CKPT_PATH = config["ckpt_path"]
    EPOCHS = config["epochs"]
    LOG_EVERY_N_STEP = config["log_every_n_step"]
    VAL_CHECK_INTERVAL = config["val_check_interval"]
    WANDB_ARGS = config["wandb"]
    ACCELERATOR = config["accelerator"]

    seed_everything(SEED)

    model, is_two_head = get_model(MODEL, ENCODER, USE_CRF)
    tokenizer = get_tokenizer(ENCODER)
    if is_two_head:
        lit_model = LitTwoHead(model, tokenizer, USE_CRF)
    else:
        lit_model = LitPunctuator(model, tokenizer, USE_CRF)

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(tokenizer)

    wandb_logger = WandbLogger(project=WANDB_ARGS["project"], name=WANDB_ARGS["name"], mode=WANDB_ARGS["mode"])

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=3,
        monitor="loss/val_loss",
        mode="min",
        filename=MODEL+"-{epoch:02d}-{val_loss:.2f}",
        dirpath=f"./checkpoints/{wandb_logger.experiment.id[-8:]}/"
    )

    early_stopping = EarlyStopping(
        monitor="loss/val_loss", min_delta=0.01, mode="min", patience=3
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        val_check_interval=VAL_CHECK_INTERVAL,
        logger=wandb_logger,
        log_every_n_steps=LOG_EVERY_N_STEP,
        callbacks=[checkpoint_callback, early_stopping],
        gpus=1,
        accelerator="gpu",
        devices=1,
    )

    if CONTINUE_FROM_CKPT:
        trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=CKPT_PATH)
    else:
        trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    trainer.test(model=lit_model, dataloaders=test_dataloader)
