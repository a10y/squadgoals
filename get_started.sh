#!/usr/bin/env bash
# Installs dependencies and downloads SQuAD and GloVe into ./data
set -euo pipefail

CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$CODE_DIR"

uv sync

# Tokenizer models for nltk.word_tokenize
uv run python -m nltk.downloader punkt punkt_tab

# SQuAD preprocess is in charge of downloading
# and formatting the data to be consumed later
uv run python preprocessing/squad_preprocess.py

# Download GloVe 840B 300d word vectors (about 2 GB)
uv run python preprocessing/dwr.py
