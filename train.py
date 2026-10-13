from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import math
import os
import time
from pprint import pprint

import numpy as np
import tensorflow as tf
from absl import app, flags

from models import MatchLSTMModel
from utils import minibatch_index_iterator, Progress


###############################################################
#       Code to setup the environment, configuration, etc.
###############################################################


"""
Configuration options:
"""
flags.DEFINE_integer('hidden_size', 150, "Size of hidden states for encoder")
flags.DEFINE_integer('batch_size', 30, 'size of mini-batches')
flags.DEFINE_integer('embed_dim', 300, 'embedding dimension')
flags.DEFINE_integer('epochs', 10, 'number of epochs for training')
flags.DEFINE_integer('layers', 2, 'number of hidden layers')

flags.DEFINE_string('cell_type', 'lstm', "Cell type for RNN")
flags.DEFINE_float('lr', 0.01, 'learning rate')

flags.DEFINE_integer('subset', 0, 'If > 0, only trains on a subset of the train data of given size')

flags.DEFINE_float('keep_prob', 0.6, 'Keep probability for dropout.')

flags.DEFINE_string('embed_path', 'data/squad/glove.squad.300d.npy', "Path to a .npy file holding the GloVe vectors")
flags.DEFINE_string('train_path', 'data/squad/train.npz', "Path to training data as an .npz file")
flags.DEFINE_string('val_path', 'data/squad/val.npz', "Path to validation data as an .npz file")
flags.DEFINE_string('save_dir', 'save', 'directory to save model checkpoints after each epoch')

flags.DEFINE_boolean('resume', False, 'Resume from latest checkpoint in save_dir when flags is passed. Default is not to')
flags.DEFINE_boolean('save', True, 'Checkpoint the model after each epoch.')
flags.DEFINE_boolean('valid', True, 'Perform validation periodically.')

"""
Utilities
"""
def load_data(path):
    data = np.load(path)

    questions = data["question"]
    contexts = data["context"]
    answers = data["answer"]

    questions_lens = data["question_lens"]
    contexts_lens = data["context_lens"]

    return questions, contexts, answers, questions_lens, contexts_lens

def minibatch_indexes(maxidx, batch_size):
    # Perform random batch ordering
    order = np.random.permutation(maxidx)
    batch_idxs = []
    for i in range(0, maxidx, batch_size):
        batch_idxs.append(order[i:i + batch_size])
    return batch_idxs


###############################################################
#           Training
###############################################################


def main(_):
    print("Starting")
    # Train the model
    config = flags.FLAGS
    print("Configuration:")
    pprint(config.flag_values_dict(), indent=4)

    tic = time.time()
    model = MatchLSTMModel(config)
    toc = time.time()
    print("Took {:.2f}s to build model.".format(toc - tic))

    questions, contexts, answers, q_lens, c_lens = load_data(config.train_path)
    num_examples = config.subset if config.subset > 0 else questions.shape[0]

    print("Parameters:")
    pprint(list(v.path for v in model.net.trainable_variables))
    total_params = sum(int(np.prod(v.shape)) for v in model.net.trainable_variables)
    print("Total params: {}".format(total_params))

    # Setup saving
    ckpt = model.checkpoint()
    manager = tf.train.CheckpointManager(ckpt, config.save_dir, max_to_keep=None)  # Cleanup manually
    best_path = os.path.join(config.save_dir, "model_best")
    tf.io.gfile.makedirs(config.save_dir)

    # Resume training
    if config.resume:
        latest = tf.train.latest_checkpoint(config.save_dir)
        print("Restoring from {}  ...".format(latest))
        ckpt.restore(latest)

    val_data = load_data(config.val_path) if config.valid else None

    # Perform training pass
    all_losses = []
    epoch_losses = []
    validation_losses = []
    print("Begin training!")

    for epoch in range(config.epochs):
        num_batches = math.ceil(num_examples / config.batch_size)
        losses = []

        # Halve the learning rate after each epoch
        lr = config.lr * (0.5)**epoch

        # Create progress bar over this
        bar = Progress('Epoch {} of {}'.format(epoch + 1, config.epochs), steps=num_batches, width=20, sameline=False)
        for batch, idxs in enumerate(minibatch_index_iterator(num_examples, config.batch_size)):
            # Read batch_size indexes for constructing training batch
            qs = questions[idxs]
            cs = contexts[idxs]
            ans = answers[idxs]
            q_ls = q_lens[idxs]
            c_ls = c_lens[idxs]

            # Perform train step
            loss, norm = model.train(qs, cs, ans, q_ls, c_ls, norms=True, lr=lr)
            losses.append(loss)
            all_losses.append(loss)

            # Calculate some stats to print
            bar.tick(loss=loss, avg10=np.average(losses[-10:]), hi=max(losses), lo=min(losses), norm=float(norm), lr=lr)

            if config.save and (batch % 100 == 0):
                print("Checkpointing...")
                manager.save(checkpoint_number=epoch)
                print("Done")

            # Calculate validation
            if config.valid and batch % 500 == 0 and batch > 0:
                # Perform validation here
                val_loss = validation(model, val_data)
                if len(validation_losses) == 0 or val_loss < np.min(validation_losses):
                    # Save best model
                    print("===> New best validation! Saving to {}".format(best_path))
                    ckpt.write(best_path)

                validation_losses.append(val_loss)
                # Write the losses out to a file for later
                print("Saving statistics...")
                np.savez("statistics.npz", epoch_losses=epoch_losses, validation=validation_losses)


        if config.valid:
            val_loss = validation(model, val_data)
            if len(validation_losses) == 0 or val_loss < np.min(validation_losses):
                # Save best model
                print("===> New best validation! Saving to {}".format(best_path))
                ckpt.write(best_path)
            validation_losses.append(val_loss)


        avg_loss = np.average(losses)
        epoch_losses.append(avg_loss)
        print("\n--- Epoch {} Average Train Loss: {:.7f}".format(epoch + 1, avg_loss))

    # Write the losses out to a file for later
    if config.save:
        print("Saving statistics...")
        np.savez("statistics.npz", epoch_losses=epoch_losses, validation=validation_losses)


def validation(model, val_data):
    print("Performing validation...")
    val_qs, val_cs, val_as, val_q_lens, val_c_lens = val_data
    num_examples = val_qs.shape[0]
    BATCH_SIZE = 30
    losses = []
    for i in range(0, num_examples, BATCH_SIZE):
        qs = val_qs[i:i+BATCH_SIZE]
        cs = val_cs[i:i+BATCH_SIZE]
        ans = val_as[i:i+BATCH_SIZE]
        qlens = val_q_lens[i:i+BATCH_SIZE]
        clens = val_c_lens[i:i+BATCH_SIZE]
        loss = model.evaluate(qs, cs, ans, qlens, clens)
        losses.append(loss)
    val_loss = np.average(losses)
    print("  \ Validation Loss: {:.7f}".format(val_loss))
    return val_loss


if __name__ == '__main__':
    app.run(main)
