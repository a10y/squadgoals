#!/usr/bin/env bash

if [[ $(uname -a) =~ "Darwin" ]]; then
    PYTHON=python
else
    PYTHON=python3
fi

NOW="$(date +%F-%H_%M_%S)"

$PYTHON -u train.py --embed_dim=300 --embed_path=data/squad/glove.squad.300d.npy --epochs=10 --batch_size=30 "$@" 2>&1 | tee output_${NOW}.log

