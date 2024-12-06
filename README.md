# Redes Neurais Player IA Super-Mario-World
Uma IA desenvolvida com redes neurais com aprendizado de reforço, com o objetivo de passar a fase e concluir os objetivos do jogo

## Dependências
- python 3.6 a 3.8
- PIP (Preferred installer Program)
- gym 0.21.0
- gym-retro
- cv2
- torch
- torchvision
- torchaudio

## Preparação de ambiente

Crie um ambiente virtual com a versão 3.6 a 3.8 do python

```sh
py -3.8 -m venv nome_do_env
```
Instale a biblioteca de setuptools na versão 65.5.0

```sh
pip install setuptools==65.5.0
```
Instale a biblioteca de wheel na versão 0.38.4

```sh
pip install wheel==0.38.4
```
Instale a biblioteca de treinamento da IA da OpenIA na versão 0.21.0

```sh
pip install gym==0.21.0
```
Instale a biblioteca que emula e capturas os frames de jogos retro

```sh
pip install gym-retro
```

Importe a ROM do Super Mario World em .sfc

```sh
py -m retro.import caminho/da/pasta/da/ROM
```
 ou instale as dependências em [requirements](requeriments.txt) com:

 ```sh
 pip install -r requirements.txt
 ```

 Rode o projeto com:
 ```sh
 py run.py
 ```

 ## Dicas