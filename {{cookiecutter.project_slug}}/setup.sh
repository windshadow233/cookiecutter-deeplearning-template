#!/bin/bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/{{ cookiecutter.torch_version }}
pip install -r requirements.txt