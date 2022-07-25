# punctation_restoration
MIPT Course project

[Презентация](https://docs.google.com/presentation/d/1_Y0DnY4xu-t4wb0QwQd38zfFLz17LDAZT007oSUTLos/edit?usp=sharing)

## Installation
1) ```python 3.8.10``` was used in this work
2) Check PyTorch version in ```requirements.txt``` according to your CUDA version
3) ```pip3 install -r requirements.txt```
4) Download [Lenta.ru dataset](https://github.com/natasha/corus#load_lenta2)
5) Review ```config.yaml``` and make changes to suit your needs

## Run training
```python train.py```

## Run server
```uvicorn server:app```
