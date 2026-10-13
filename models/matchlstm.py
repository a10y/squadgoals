import keras
import numpy as np
import tensorflow as tf

from cells import *
from utils import *


def best_span(start_logits, end_logits, max_answer_len=15):
    """Find the (start, end) pair with the highest joint probability, such that
    start <= end <= start + max_answer_len.

    Taking the argmax of the start and the end on their own can give end < start,
    which gives an empty answer.
    """
    log_p_s = tf.nn.log_softmax(start_logits, axis=1)
    log_p_e = tf.nn.log_softmax(end_logits, axis=1)
    P = tf.shape(log_p_s)[1]
    scores = tf.expand_dims(log_p_s, 2) + tf.expand_dims(log_p_e, 1)    # [batch, start, end]
    valid = tf.cast(tf.linalg.band_part(tf.ones([P, P]), 0, tf.minimum(max_answer_len, P)), tf.bool)
    scores = tf.where(valid, scores, tf.fill(tf.shape(scores), NEG_INF))
    flat = tf.argmax(tf.reshape(scores, [tf.shape(scores)[0], -1]), axis=1, output_type=tf.int32)
    return flat // P, flat % P


class MatchLSTMNetwork(keras.Model):
    """Model based on the MatchLSTM with Answer Pointer paper."""
    def __init__(self, embeddings, hidden_size, keep_prob, cell_type="lstm", **kwargs):
        super().__init__(**kwargs)
        # GloVe embeddings are a constant: do not train them, and do not save them.
        self._embed = tf.constant(embeddings, dtype=tf.float32)

        if cell_type == "lstm":
            cell = keras.layers.LSTM
        elif cell_type == "gru":
            cell = keras.layers.GRU
        else:
            raise ValueError("Unknown cell_type {}".format(cell_type))

        rate = 1.0 - keep_prob
        self.encode_question = keras.layers.Bidirectional(cell(hidden_size, return_sequences=True), name="encode_question")
        self.encode_passage = keras.layers.Bidirectional(cell(hidden_size, return_sequences=True), name="encode_passage")
        self.match_lstm = MatchLSTM(2 * hidden_size, name="match_lstm")
        self.answer_ptr = AnswerPointer(2 * hidden_size, name="answer_ptr")
        self.dropout = keras.layers.Dropout(rate)

    def call(self, question, context, qlens, clens, training=False):
        questions = tf.nn.embedding_lookup(self._embed, question)
        contexts = tf.nn.embedding_lookup(self._embed, context)
        q_mask = tf.sequence_mask(qlens, tf.shape(question)[1])
        c_mask = tf.sequence_mask(clens, tf.shape(context)[1])

        # H_q and H_p from the paper, shape [batch_size, Q or P, 2*hidden_size]
        H_q = self.dropout(self.encode_question(questions, mask=q_mask), training=training)
        H_p = self.dropout(self.encode_passage(contexts, mask=c_mask), training=training)

        # Attention over the question for each token of the passage, shape [batch_size, P, 4*hidden_size]
        H_r = self.dropout(self.match_lstm((H_p, H_q), clens, qlens), training=training)

        # Unscaled logits for the start and the end of the answer
        return self.answer_ptr(H_r, clens)


_BATCH_SIGNATURE = [
    tf.TensorSpec([None, None], tf.int32),  # questions
    tf.TensorSpec([None, None], tf.int32),  # contexts
    tf.TensorSpec([None], tf.int32),        # qlens
    tf.TensorSpec([None], tf.int32),        # clens
]
_SPAN_SIGNATURE = [tf.TensorSpec([None], tf.int32), tf.TensorSpec([None], tf.int32)]


class MatchLSTMModel(object):
    """Wraps MatchLSTMNetwork with an optimizer, training, prediction and checkpoints."""
    def __init__(self, config):
        self._config = config
        self._global_norm = 5.0  # Maximum norm at which point we want to clip.

        print("{}: Loading GloVe vectors from {}".format(self.__class__.__name__, config.embed_path))
        embeddings = np.load(config.embed_path)
        self.net = MatchLSTMNetwork(embeddings, config.hidden_size, config.keep_prob, config.cell_type)
        self.optimizer = keras.optimizers.Adamax(learning_rate=config.lr)

        # Create all variables now, so that checkpoints can be restored into them.
        one = tf.ones([1, 1], tf.int32)
        self.net(one, one, tf.ones([1], tf.int32), tf.ones([1], tf.int32))
        self.optimizer.build(self.net.trainable_variables)

        self._train_step = tf.function(self._train_step, input_signature=_BATCH_SIGNATURE + _SPAN_SIGNATURE)
        self._eval_step = tf.function(self._eval_step, input_signature=_BATCH_SIGNATURE + _SPAN_SIGNATURE)
        self._predict_step = tf.function(self._predict_step,
                                         input_signature=_BATCH_SIGNATURE + [tf.TensorSpec([], tf.int32)])

    def _loss(self, starts, ends, B_s, B_e):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=starts, logits=B_s) \
               + tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ends, logits=B_e)
        return tf.reduce_mean(loss)

    def _train_step(self, questions, contexts, qlens, clens, starts, ends):
        with tf.GradientTape() as tape:
            B_s, B_e = self.net(questions, contexts, qlens, clens, training=True)
            loss = self._loss(starts, ends, B_s, B_e)
        variables = self.net.trainable_variables
        gradients = tape.gradient(loss, variables)
        gradients, grad_norm = tf.clip_by_global_norm(gradients, self._global_norm)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss, grad_norm

    def _eval_step(self, questions, contexts, qlens, clens, starts, ends):
        B_s, B_e = self.net(questions, contexts, qlens, clens, training=False)
        return self._loss(starts, ends, B_s, B_e)

    def _predict_step(self, questions, contexts, qlens, clens, max_answer_len):
        B_s, B_e = self.net(questions, contexts, qlens, clens, training=False)
        return best_span(B_s, B_e, max_answer_len)

    def train(self, questions, contexts, answers, qlens, clens, norms=False, lr=0.01):
        """Performs one training step. Returns the loss, and the gradient norm before clipping when norms=True."""
        self.optimizer.learning_rate.assign(lr)
        starts, ends = self._spans(answers)
        loss, grad_norm = self._train_step(*self._batch(questions, contexts, qlens, clens), starts, ends)
        if norms:
            return loss.numpy(), grad_norm.numpy()
        else:
            return loss.numpy()

    def predict(self, questions, contexts, qlens, clens, max_answer_len=15):
        """Returns the predicted start and end token for each example, with start <= end."""
        starts, ends = self._predict_step(*self._batch(questions, contexts, qlens, clens),
                                          tf.constant(max_answer_len, tf.int32))
        return starts.numpy(), ends.numpy()

    def evaluate(self, questions, contexts, answers, qlens, clens):
        starts, ends = self._spans(answers)
        return self._eval_step(*self._batch(questions, contexts, qlens, clens), starts, ends).numpy()

    def checkpoint(self):
        return tf.train.Checkpoint(model=self.net, optimizer=self.optimizer)

    @staticmethod
    def _batch(questions, contexts, qlens, clens):
        return (np.asarray(questions, dtype=np.int32), np.asarray(contexts, dtype=np.int32),
                np.asarray(qlens, dtype=np.int32), np.asarray(clens, dtype=np.int32))

    @staticmethod
    def _spans(answers):
        spans = np.zeros([answers.shape[0], 2], dtype=np.int32)
        for i, line in enumerate(answers):
            places = np.where(line != 0)[0]
            spans[i] = [places[0], places[-1]]
        return spans[:, 0], spans[:, 1]
