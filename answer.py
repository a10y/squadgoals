"""
Perform generation of answers here.

Expects that the dev files are at data/squad/dev.*
To get this, run preprocessing/squad_preprocess.py
"""

import json
import os
import time
from pprint import pprint

import numpy as np
import tensorflow as tf
from absl import app, flags
from tensorflow.io import gfile
from tqdm import tqdm

from models import MatchLSTMModel

"""
Configuration   -- I hate this file because I make the model config depend upon the flags :/ BAD ANDREW
"""
flags.DEFINE_integer('hidden_size', 150, "Size of hidden states for encoder")
flags.DEFINE_integer('batch_size', 30, 'size of mini-batches')
flags.DEFINE_integer('embed_dim', 300, 'embedding dimension')
flags.DEFINE_integer('epochs', 10, 'number of epochs for training')
flags.DEFINE_integer('layers', 2, 'number of hidden layers')

flags.DEFINE_string('model', 'match', 'type of model, either "match" for Match-LSTM w/Answer Pointer or "bilstm" for simple BiLSTM baseline.')
flags.DEFINE_string('cell_type', 'lstm', "Cell type for RNN")
flags.DEFINE_float('lr', 0.01, 'learning rate')

flags.DEFINE_string('optim', 'adam', 'Optimizer, one of "adam", "adadelta", "sgd"')

flags.DEFINE_integer('subset', 0, 'If > 0, only trains on a subset of the train data of given size')

flags.DEFINE_float('keep_prob', 0.6, 'Keep probability for dropout.')

flags.DEFINE_string('embed_path', 'data/squad/glove.squad.300d.npy', "Path to a .npy file holding the GloVe vectors")
flags.DEFINE_string('train_path', 'data/squad/train.npz', "Path to training data as an .npz file")
flags.DEFINE_string('val_path', 'data/squad/val.npz', "Path to validation data as an .npz file")
flags.DEFINE_string('save_dir', 'save', 'directory to save model checkpoints after each epoch')

flags.DEFINE_boolean('resume', False, 'Resume from latest checkpoint in save_dir when flags is passed. Default is not to')
flags.DEFINE_boolean('nosave', False, 'When passed, saving the model to disk is not performed.')

flags.DEFINE_string("dev_root", "data/squad/", "Root directory of dev info")
flags.DEFINE_string("checkpoint", "", "Checkpoint file to restore from, defaults to latest")
flags.DEFINE_string("out", "predictions.json", "output file name")
flags.DEFINE_integer("max_answer_len", 15, "Predicted end token is at most this many tokens after the start token")

FLAGS = flags.FLAGS  # Alias longer name


def load_vocabulary():
    vocab_path = os.path.join(FLAGS.dev_root, "vocab.dat")
    with gfile.GFile(vocab_path, "r") as vocab_file:
        return list(map(lambda l: l.strip(), vocab_file.readlines()))

def read_dev():
    # Open all of the files
    compressed_dev = np.load(os.path.join(FLAGS.dev_root, "dev.npz"))
    qid_path = os.path.join(FLAGS.dev_root, "dev.qid")
    with gfile.GFile(qid_path, "r") as qid_file:
        question_uids = list(map(lambda l: l.strip(), qid_file.readlines()))
    return compressed_dev, question_uids


def restore(model):
    ckpt = FLAGS.checkpoint or tf.train.latest_checkpoint(FLAGS.save_dir)
    print("Restoring from checkpoint file {}".format(ckpt))
    model.checkpoint().restore(ckpt).expect_partial()
    print("Restoration finished")


def build_answer(id_to_word):

    # Load the data
    print("Configuration")
    pprint(FLAGS.flag_values_dict(), indent=4)

    tic = time.time()
    model = MatchLSTMModel(FLAGS)
    toc = time.time()
    print("Took {:.2f}s to build model.".format(toc - tic))

    # Perform iterations here over the question dataset
    dev_data, qids = read_dev()
    question = dev_data["question"]
    context = dev_data["context"]
    qlens = dev_data["question_lens"]
    clens = dev_data["context_lens"]

    # Perform prediction using the model.
    restore(model)

    answers = {}    # Answer dictionar. answers[qid] = "Answer"
    # Perform answering in batches, no reason to do this
    for i in tqdm(range(0, len(qids), FLAGS.batch_size)):
        batch_qids = qids[i:i+FLAGS.batch_size]
        qs, cs, ql, cl = (question[i:i+FLAGS.batch_size], context[i:i+FLAGS.batch_size],
                          qlens[i:i+FLAGS.batch_size], clens[i:i+FLAGS.batch_size])
        # Some of them for some reason have unequal batch sizes. Oh well, skip them
        preds_start, preds_end = model.predict(qs, cs, ql, cl, max_answer_len=FLAGS.max_answer_len)

        for i, (start, end) in enumerate(zip(preds_start, preds_end)):
            qid = batch_qids[i]
            tokens = ' '.join(id_to_word[int(j)] for j in cs[i][start:end+1])   # Join tokens basd on space, what about 's, 't, etc.?
            answers[qid] = tokens

    # Dump all of the answers out to a JSON file
    print("Dumping predictions to JSON file...")
    with gfile.GFile(FLAGS.out, "w") as preds_file:
        preds_file.write(json.dumps(answers, ensure_ascii=False, indent=4)) # Write out pretty so we can read it ourselves


def main(_):
    id_to_word = load_vocabulary()
    build_answer(id_to_word)

if __name__ == "__main__":
    app.run(main)
