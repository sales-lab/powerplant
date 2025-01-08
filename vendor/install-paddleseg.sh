#!/bin/bash
set -e

git clone --branch release/2.10 --single-branch --depth=1 https://github.com/PaddlePaddle/PaddleSeg

cd PaddleSeg
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install wheel
pip install paddlepaddle
echo "" >>requirements.txt
echo "numpy<2" >>requirements.txt
pip install -r requirements.txt
pip install -v -e .
