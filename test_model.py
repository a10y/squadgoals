"""
Tests for the cells and the model. Run with `python test_model.py` or `pytest test_model.py`.
"""
import tempfile
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from cells import AnswerPointer, MatchLSTM
from models import MatchLSTMModel
from models.matchlstm import best_span


def test_match_lstm_shape_and_padding():
    BATCH_SIZE, P, Q, HIDDEN_SIZE = [4, 12, 7, 10]
    H_p = tf.random.normal([BATCH_SIZE, P, 2 * HIDDEN_SIZE])
    H_q = tf.random.normal([BATCH_SIZE, Q, 2 * HIDDEN_SIZE])
    p_lens = tf.constant([12, 5, 1, 8])
    q_lens = tf.constant([7, 3, 2, 7])
    layer = MatchLSTM(HIDDEN_SIZE)
    H_r = layer((H_p, H_q), p_lens, q_lens).numpy()
    assert H_r.shape == (BATCH_SIZE, P, 2 * HIDDEN_SIZE), "Actual shape: {}".format(H_r.shape)
    # Outputs past the end of the passage are zero
    assert np.all(H_r[1, 5:] == 0.0)
    assert np.any(H_r[1, :5] != 0.0)

    # Values in padded question positions must not change the output
    noise = tf.concat([tf.zeros([BATCH_SIZE, 3, 2 * HIDDEN_SIZE]),
                       100.0 * tf.ones([BATCH_SIZE, Q - 3, 2 * HIDDEN_SIZE])], axis=1)
    H_r2 = layer((H_p, H_q + noise), p_lens, q_lens).numpy()
    np.testing.assert_allclose(H_r[1], H_r2[1], atol=1e-5)


def test_answer_pointer_masks_padding():
    BATCH_SIZE, P, HIDDEN_SIZE = [5, 20, 10]
    H_r = tf.random.normal([BATCH_SIZE, P, 2 * HIDDEN_SIZE])
    p_lens = tf.constant([20, 10, 3, 1, 15])
    B_s, B_e = AnswerPointer(HIDDEN_SIZE)(H_r, p_lens)
    for B in [tf.nn.softmax(B_s).numpy(), tf.nn.softmax(B_e).numpy()]:
        assert np.allclose(np.sum(B, axis=1), 1.0), "Rows must sum to 1"
        for i, length in enumerate(p_lens.numpy()):
            assert np.all(B[i, length:] == 0.0), "Padded positions must get no probability"


def test_best_span_end_not_before_start():
    # The argmax of the end comes before the argmax of the start
    start = tf.math.log(tf.constant([[0.1, 0.1, 0.6, 0.1, 0.1]]))
    end = tf.math.log(tf.constant([[0.5, 0.05, 0.05, 0.35, 0.05]]))
    s, e = best_span(start, end, max_answer_len=15)
    assert (s.numpy()[0], e.numpy()[0]) == (2, 3)

    # The answer can not be longer than max_answer_len
    start = tf.math.log(tf.constant([[0.9, 0.025, 0.025, 0.025, 0.025]]))
    end = tf.math.log(tf.constant([[0.01, 0.1, 0.09, 0.0, 0.8]]))
    s, e = best_span(start, end, max_answer_len=2)
    assert (s.numpy()[0], e.numpy()[0]) == (0, 1)

    # Random inputs always give start <= end <= start + max_answer_len
    s, e = best_span(tf.random.normal([50, 40]), tf.random.normal([50, 40]), max_answer_len=15)
    s, e = s.numpy(), e.numpy()
    assert np.all(s <= e) and np.all(e <= s + 15)


def _toy_data(n=8, P=15, Q=6, vocab=30):
    rng = np.random.RandomState(0)
    questions = rng.randint(1, vocab, size=[n, Q])
    contexts = rng.randint(1, vocab, size=[n, P])
    clens = rng.randint(5, P + 1, size=[n])
    qlens = rng.randint(2, Q + 1, size=[n])
    answers = np.zeros([n, P], dtype=np.int32)
    for i in range(n):
        contexts[i, clens[i]:] = 0
        questions[i, qlens[i]:] = 0
        start = rng.randint(0, clens[i] - 1)
        answers[i, start:start + 2] = 1
    return questions, contexts, answers, qlens, clens


def test_model_trains_and_restores():
    vocab, embed_dim = 30, 8
    with tempfile.TemporaryDirectory() as tmp:
        embed_path = tmp + "/embed.npy"
        np.save(embed_path, np.random.RandomState(1).randn(vocab, embed_dim).astype(np.float32))
        config = SimpleNamespace(embed_path=embed_path, hidden_size=6, keep_prob=1.0, cell_type="lstm", lr=0.01)

        qs, cs, ans, qlens, clens = _toy_data(vocab=vocab)
        model = MatchLSTMModel(config)
        first = model.evaluate(qs, cs, ans, qlens, clens)
        for _ in range(60):
            loss, norm = model.train(qs, cs, ans, qlens, clens, norms=True, lr=0.01)
        assert np.isfinite(loss) and np.isfinite(norm)
        assert model.evaluate(qs, cs, ans, qlens, clens) < first, "Loss should go down when overfitting"

        starts, ends = model.predict(qs, cs, qlens, clens)
        assert np.all(starts <= ends) and np.all(ends < clens)

        path = model.checkpoint().write(tmp + "/model_best")
        restored = MatchLSTMModel(config)
        restored.checkpoint().restore(path).expect_partial()
        s2, e2 = restored.predict(qs, cs, qlens, clens)
        np.testing.assert_array_equal(starts, s2)
        np.testing.assert_array_equal(ends, e2)


# Run the tests when called from command line
if __name__ == '__main__':
    test_match_lstm_shape_and_padding()
    test_answer_pointer_masks_padding()
    test_best_span_end_not_before_start()
    test_model_trains_and_restores()
    print("all tests passed!")
