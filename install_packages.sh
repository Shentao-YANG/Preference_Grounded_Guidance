#!/bin/bash

pip3 install --upgrade pip
pip uninstall typing
pip install --upgrade --no-cache-dir --ignore-installed tbb
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install --upgrade --no-cache-dir typing-extensions accelerate datasets sentencepiece protobuf==3.20.* rouge-score nltk py7zr evaluate undecorated tensorflow
pip install --upgrade --no-cache-dir numpy pandas typing wandb transformers hydra-core==1.2.0 absl-py gym dm-tree gast urllib3 packaging numexpr requests daal pathlib pyqt5 pyqtwebengine ruamel-yaml
