import keras
import tensorflow as tf

from utils import *

# Logit value for masked (padded) positions. Large enough that softmax gives them ~0 mass.
NEG_INF = -1e30


def masked_logits(logits, mask):
    """Replace logits at positions where mask is False with NEG_INF."""
    return tf.where(mask, logits, tf.fill(tf.shape(logits), NEG_INF))


class MatchLSTM(keras.layers.Layer):
    """Match-LSTM layer from Wang & Jiang (2016).

    For each passage token, attends over the question and feeds [h_p; attention] into an LSTM.
    The same weights are used for the forward and the backward pass.
    """
    def __init__(self, hidden_size, **kwargs):
        super().__init__(**kwargs)
        # Padding is handled with explicit lengths, so a Keras mask on the input is fine.
        self.supports_masking = True
        self._hidden_size = hidden_size
        self._cell = keras.layers.LSTMCell(hidden_size, name="cell")

    def build(self, input_shape):
        H_p_shape, H_q_shape = input_shape
        p_dim, q_dim, h = H_p_shape[-1], H_q_shape[-1], self._hidden_size
        init = "glorot_uniform"
        self.Wq = self.add_weight(name="Wq", shape=[q_dim, h], initializer=init)
        self.Wp = self.add_weight(name="Wp", shape=[p_dim, h], initializer=init)
        self.Wr = self.add_weight(name="Wr", shape=[h, h], initializer=init)
        self.Wg = self.add_weight(name="Wg", shape=[h, 1], initializer=init)
        self.bp = self.add_weight(name="bp", shape=[h], initializer="zeros")
        self.b = self.add_weight(name="b", shape=[], initializer="zeros")
        # Build the cell here, because variables can not be created inside the tf.scan loop.
        self._cell.build((None, p_dim + q_dim))

    def call(self, inputs, p_lens, q_lens):
        H_p, H_q = inputs
        fw = self._run(H_p, H_q, p_lens, q_lens)
        bw = self._run(tf.reverse_sequence(H_p, p_lens, seq_axis=1, batch_axis=0), H_q, p_lens, q_lens)
        bw = tf.reverse_sequence(bw, p_lens, seq_axis=1, batch_axis=0)
        return tf.concat([fw, bw], axis=2)

    def _run(self, H_p, H_q, p_lens, q_lens):
        batch_size = tf.shape(H_p)[0]
        P, Q = tf.shape(H_p)[1], tf.shape(H_q)[1]
        p_mask = tf.sequence_mask(p_lens, P)
        q_mask = tf.sequence_mask(q_lens, Q)

        # The question term does not change across time steps, so compute it once.
        qterm = batch_matmul(H_q, self.Wq)

        def step(carry, elems):
            _, h, c = carry
            hp, valid = elems
            rpterm = tf.matmul(hp, self.Wp) + tf.matmul(h, self.Wr) + self.bp
            G = tf.nn.tanh(qterm + tf.expand_dims(rpterm, 1))
            logits = tf.squeeze(batch_matmul(G, self.Wg), axis=2) + self.b
            # Do not attend to padded question positions.
            alpha = tf.nn.softmax(masked_logits(logits, q_mask), axis=1)
            att = tf.einsum("bq,bqd->bd", alpha, H_q)
            z = tf.concat([hp, att], axis=1)
            out, (h_new, c_new) = self._cell(z, [h, c])

            # Past the end of the passage, keep the state and output zeros.
            valid = tf.expand_dims(valid, 1)
            out = tf.where(valid, out, tf.zeros_like(out))
            h_new = tf.where(valid, h_new, h)
            c_new = tf.where(valid, c_new, c)
            return out, h_new, c_new

        zeros = tf.zeros([batch_size, self._hidden_size], dtype=H_p.dtype)
        outputs, _, _ = tf.scan(step,
                                (tf.transpose(H_p, [1, 0, 2]), tf.transpose(p_mask)),
                                initializer=(zeros, zeros, zeros))
        return tf.transpose(outputs, [1, 0, 2])


class AnswerPointer(keras.layers.Layer):
    """Boundary Answer Pointer layer. Returns masked logits for the start and the end token."""
    def __init__(self, hidden_size, **kwargs):
        super().__init__(**kwargs)
        # Padding is handled with explicit lengths, so a Keras mask on the input is fine.
        self.supports_masking = True
        self._hidden_size = hidden_size
        self._cell = keras.layers.LSTMCell(hidden_size, name="cell")

    def build(self, input_shape):
        r_dim, h = input_shape[-1], self._hidden_size
        init = "glorot_uniform"
        self.V = self.add_weight(name="V", shape=[r_dim, h], initializer=init)
        self.W_a = self.add_weight(name="W_a", shape=[h, h], initializer=init)
        self.v = self.add_weight(name="v", shape=[h, 1], initializer=init)
        self.b_a = self.add_weight(name="b_a", shape=[h], initializer="zeros")
        self.c = self.add_weight(name="c", shape=[1], initializer="zeros")
        self._cell.build((None, r_dim))

    def call(self, H_r, p_lens):
        batch_size = tf.shape(H_r)[0]
        p_mask = tf.sequence_mask(p_lens, tf.shape(H_r)[1])

        # H_r V does not change between the two steps, so compute it once.
        HrV = batch_matmul(H_r, self.V)
        h = tf.zeros([batch_size, self._hidden_size], dtype=H_r.dtype)
        c = tf.zeros_like(h)

        logits = []
        for _ in range(2):
            # F = tanh(H_r V + h_(k)a W_a + b_a)
            F = tf.nn.tanh(HrV + tf.expand_dims(tf.matmul(h, self.W_a), 1) + self.b_a)
            # B = softmax(F_k v + c), with padded passage positions masked out
            B_logits = masked_logits(tf.squeeze(batch_matmul(F, self.v), axis=2) + self.c, p_mask)
            B = tf.nn.softmax(B_logits, axis=1)
            logits.append(B_logits)

            # h_(k)a = LSTM(B_k H_r, h_(k-1)a)
            cell_in = tf.einsum("bp,bpd->bd", B, H_r)
            _, (h, c) = self._cell(cell_in, [h, c])

        return logits[0], logits[1]
