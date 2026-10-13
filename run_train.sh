#!/usr/bin/env bash

NOW="$(date +%F-%H_%M_%S)"

uv run python -u train.py --embed_dim=300 --embed_path=data/squad/glove.squad.300d.npy --epochs=10 --batch_size=30 "$@" 2>&1 | tee output_${NOW}.log
