# #SQuADGoals


## Building

```
./get_started.sh

./split_glove.sh

python preprocess.py

python train.py

python answer.py --checkpoint=save/model_best
```

Needs Python 3.10+ and TensorFlow 2.16+ (tested with TensorFlow 2.21, Keras 3).
Checkpoints from the old TensorFlow 1.x version of this code can not be loaded.

## Tests

```
python -m pytest test_model.py
```

## Results

My implementation achieve 53.47 F1 score, and 41.425 Exact Match on the SQuAD hidden test set.
It falls short of the original authors' scores, but still handily beats a heavily feature-engineered
logistic regression implmentation.

You can see the run [here on CodaLab](https://worksheets.codalab.org/bundles/0x8803cedd7f1f4f7981312d1b2c472cf2/).
