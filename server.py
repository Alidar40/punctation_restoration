import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


from torch.utils.data import DataLoader

from models.model_definition import get_model, get_tokenizer
from models.lit_punctuator import LitPunctuator
from models.lit_two_head import LitTwoHead
from data_processing.lentaset import LentaSet
from config import config, char2label
from utils.initialization import seed_everything


app = FastAPI()
templates = Jinja2Templates(directory="templates")


SEED = config["seed"]
MODEL = config["model"]
USE_CRF = config["use_crf"]
ENCODER = config["encoder"]
DATASET_PATH = config["dataset_path"]
SEQUENCE_LEN = config["sequence_len"]
AUGMENT_RATE = config["augment_rate"]
BATCH_SIZE = config["batch_size"]
NUM_WORKERS = config["num_workers"]
DEV_MODE = config["dev_mode"]
CHUNK_SIZE = config["chunk_size"]
CKPT_PATH = config["ckpt_path"]

seed_everything(SEED)
model, is_two_head = get_model(MODEL, ENCODER, USE_CRF)
model.eval()
tokenizer = get_tokenizer(ENCODER)
if is_two_head:
    lit_model = LitTwoHead.load_from_checkpoint(CKPT_PATH, punctuator=model, tokenizer=tokenizer, use_crf=USE_CRF)
else:
    lit_model = LitPunctuator.load_from_checkpoint(CKPT_PATH, punctuator=model, tokenizer=tokenizer, use_crf=USE_CRF)
lit_model.eval()

class Item(BaseModel):
    input_text: str


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/")
def home_predict(request: Request, input_text: str = Form()):
    pred = None
    error = None
    try:
        dataset = LentaSet([input_text], tokenizer, SEQUENCE_LEN, AUGMENT_RATE, is_train=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
        for batch in dataloader:
            batch_idx = 0
            pred = lit_model.predict_step(batch, batch_idx)
            break
        input_text = dataset.get_raw_text(0)
    except Exception as ex:
        error = ex
        print(error)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": {"input": input_text, "predicted": pred},
            "error": error
        }
    )


if __name__ == "__main__":
    uvicorn.run("fastapi_code:app")
