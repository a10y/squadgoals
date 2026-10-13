# #SQuADGoals


## Building

Needs [uv](https://docs.astral.sh/uv/). It installs Python 3.12 and TensorFlow 2.21 (Keras 3) into `.venv`.

```
./get_started.sh             # uv sync, then download SQuAD v1.1 and GloVe 840B

uv run python preprocess.py  # build data/squad/*.npz and the trimmed GloVe matrix

./run_train.sh               # train for 10 epochs, checkpoints in save/

uv run python answer.py --checkpoint=save/model_best
uv run python evaluate.py data/squad/dev-v1.1.json predictions.json
```

To train on a free GPU, open [notebooks/colab_train.ipynb](notebooks/colab_train.ipynb) in Google Colab.

Checkpoints from the old TensorFlow 1.x version of this code can not be loaded.

## Tests

```
uv run pytest
```

## Results

My implementation achieve 53.47 F1 score, and 41.425 Exact Match on the SQuAD hidden test set.
It falls short of the original authors' scores, but still handily beats a heavily feature-engineered
logistic regression implmentation.

You can see the run [here on CodaLab](https://worksheets.codalab.org/bundles/0x8803cedd7f1f4f7981312d1b2c472cf2/).
