import logging
from os.path import join as pjoin

import tensorflow as tf
from tensorflow import nn
from tensorflow.contrib import rnn

from core import SquadModel

logging.getLogger().setLevel(logging.INFO)


class BaselineModel(SquadModel):
    """
    BiLSTM SQuAD model, simple baseline.
    """

    def __init__(self):
        super(BaselineModel, self).__init__()
        self._question_placeholder = None
        self._passage_placeholder = None
        self._answer_placeholder = None
        self._cell = None
        self._loss = None
        self._train_step = None
        self._model_output = None
        self._preds = None

    def initialize_graph(self, config):
        self._question_placeholder = tf.placeholder(tf.float32, [None, None, config.embed_size], "question_embedded")
        self._passage_placeholder = tf.placeholder(tf.float32, [None, None, config.embed_size], "passage_embedded")
        self._answer_placeholder = tf.placeholder(tf.int32, [None, None], "answer_batch")

        self._model_output = config.save_dir

        if config.cell_type == "lstm":
            cell = rnn.MultiRNNCell([rnn.LSTMCell(config.hidden_size)] * 2)
        elif config.cell_type == "gru":
            cell = rnn.MultiRNNCell([rnn.GRUCell(config.hidden_size)] * 2)
        else:
            raise ValueError("Invalid cell_type {}".format(config.cell_type))

        """
        Run the question through an RNN, using the final output as the "summary" of the question.

        We then feed this summary h_q as the first hidden state to an encoder of the passage, which yields outputs
        h_p. We then perform a final LSTM step where we encode the states h_p through a third RNN that outputs
        a probability distribution per-token of the passage.
        """
        with tf.variable_scope("question_rnn"):
            _, h_q = nn.dynamic_rnn(cell, self._question_placeholder, dtype=tf.float32)
        with tf.variable_scope("passage_rnn"):
            h_p, _ = nn.dynamic_rnn(cell, self._passage_placeholder, initial_state=h_q)

        with tf.variable_scope("seq_classifier"):
            classifier_cell = rnn.LSTMCell(2)
            self._preds, _ = nn.dynamic_rnn(classifier_cell, h_p, dtype=tf.float32)

        # Perform a classification for each token individually
        losses = nn.sparse_softmax_cross_entropy_with_logits(labels=self._answer_placeholder, logits=self._preds)
        self._loss = tf.reduce_mean(losses)
        self._train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self._loss)

    def train_batch(self, question_batch, passage_batch, answer_batch, sess=None):
        """
        TODO: replace feed_dict with whatever is supposed to be more efficient
        Link: https://www.tensorflow.org/programmers_guide/reading_data
        """

        if sess is None:
            sess = tf.get_default_session()

        feeds = {
            self._question_placeholder: question_batch,
            self._passage_placeholder: passage_batch,
            self._answer_placeholder: answer_batch,
        }

        _, loss = sess.run([self._train_step, self._loss], feed_dict=feeds)
        return loss

    def predict(self, question_batch, passage_batch, answer_batch, sess=None):
        """
        Predicts the (start, end) span representing the answer for the given question over the given passage.
        """
        if sess is None:
            sess = tf.get_default_session()

        feeds = {
            self._question_placeholder: question_batch,
            self._passage_placeholder: passage_batch,
            self._answer_placeholder: answer_batch,
        }

        return sess.run(self._loss, feed_dict=feeds)

    def checkpoint(self, save_dir, sess=None):
        if sess is None:
            sess = tf.get_default_session()

        save_file = pjoin(self._model_output, "model.weights")
        saver = tf.train.Saver()
        saver.save(sess, save_file)

    def restore_from_checkpoint(self, save_dir, sess=None):
        if sess is None:
            sess = tf.get_default_session()

        restorer = tf.train.Saver()
        save_path = tf.train.latest_checkpoint(save_dir)
        logging.info("Restoring from {}".format(save_path))
        restorer.restore(sess, save_path)