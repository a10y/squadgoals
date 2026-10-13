#!/usr/bin/env bash

if [[ $(uname -a) =~ "Darwin" ]]; then
    PYTHON=python
else
    PYTHON=python3
fi

$PYTHON train.py --embed_dim=300 --embed_path=data/squad/glove.squad.300d.npy --nosave --novalid --epochs=100 --batch_size=10 --subset=10 "$@"

