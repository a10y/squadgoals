from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from collections import Counter
from os.path import join as pjoin

import numpy as np
from absl import app, flags
from tensorflow.io import gfile
from tqdm import tqdm

import logging
logging.getLogger().setLevel(logging.INFO)

_PAD = "<pad>"
_SOS = "<sos>"
_UNK = "<unk>"
_START_VOCAB = [_PAD, _SOS, _UNK]

PAD_ID = 0
SOS_ID = 1
UNK_ID = 2


def assert_exists(path, message):
    if not gfile.exists(path):
        raise AssertionError("Path {} not found: {}".format(path, message))


def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def build_vocabulary(corpus, dest):
    """
    Build a vocabulary from the files in the list |corpus|
    Outputs the final vocabulary list to |dest| if the file does not already exist.

    Return: (id_to_word, word_to_id)
    """
    if gfile.exists(dest):
        # Load the file
        with gfile.GFile(dest, "r") as vocab_file:
            logging.info("Loading pre-built vocabulary from {}".format(dest))
            id_to_word = [l.strip() for l in vocab_file.readlines()]
            word_to_id = dict([(word, i) for i, word in enumerate(id_to_word)])
            return id_to_word, word_to_id

    logging.info("Building vocabulary from {}".format(corpus))
    word_and_freq = Counter()
    for document in tqdm(corpus):
        with gfile.GFile(document, "r") as fh:
            for line in tqdm(fh):
                for token in basic_tokenizer(line):
                    word_and_freq[token] += 1

    ordered_vocab = list(map(lambda t: t[0], word_and_freq.most_common()))
    id_to_word = _START_VOCAB + ordered_vocab
    if not gfile.exists(dest):
        logging.info("Serializing vocabulary to {}".format(dest))
        with gfile.GFile(dest, "w") as vocab_file:
            for word in id_to_word:
                vocab_file.write(word + "\n")

    word_to_id = dict([(word, i) for i, word in enumerate(id_to_word)])
    return id_to_word, word_to_id


def load_or_create(glove_root_dir, processed_dir, embed_size, id_to_word):
    """
    Loads from memory a set of GloVe vectors.
    """
    glove_final_path = pjoin(processed_dir, "glove.squad.{}d.npy".format(embed_size))
    if gfile.exists(glove_final_path):
        logging.info("Loading pre-built GloVe npy from {}".format(glove_final_path))
        return np.load(glove_final_path)
    else:
        logging.info("Parsing GloVe vectors from text format (try to only do this once, it is must slower than npy)")
        vec_path = pjoin(glove_root_dir, "vec.{}d.txt".format(embed_size))

        assert_exists(vec_path, "Please run ./split_vec.sh from the root of the project.")
        glove_mat = np.loadtxt(vec_path)

        # Word order
        glove_order = {}
        tokens_path = pjoin(glove_root_dir, "token.{}d.txt".format(embed_size))
        assert_exists(tokens_path, "Please run ./split_vec.sh from root of the project.")
        # Re-arrange based on our vocabulary
        with gfile.GFile(tokens_path) as glove_tokens:
            for i, token in enumerate(glove_tokens):
                # print(token)
                glove_order[token.strip()] = i

        print("Read {} tokens from {}".format(len(glove_order), tokens_path))

        # We want to create an embedding matrix that is the same size as the vocabulary
        trimmed_glove = np.zeros((len(id_to_word), embed_size))

        for i, vocab_word in enumerate(id_to_word):
            if vocab_word in glove_order:
                trimmed_glove[i] = glove_mat[glove_order[vocab_word]]

        logging.info("Saving GloVe vectors to compact file {}".format(glove_final_path))
        np.save(glove_final_path, trimmed_glove)
        return trimmed_glove


def build_glove(id_to_word):
    glove_root_dir = flags.FLAGS.glove_dir
    processed_dir = flags.FLAGS.squad_dir
    embed_size = flags.FLAGS.embed_size
    return load_or_create(glove_root_dir, processed_dir, embed_size, id_to_word)


def tokens_to_ids(source_file, word_to_id, max_len=None, useindexes=None):
    """Reads the given source file that is assumed to be sequence-per-line,
    where the sequences are raw English tokens.

    Returns: A new np.array with shape [num_examples, max_seq_len, embedding_size]
    """
    longest = 0
    sequences = []
    lengths = []
    indexes = []
    logging.info("Preprocessing {} into an unrolled np.array".format(source_file))
    with gfile.GFile(source_file, "r") as data_file:
        for i, line in enumerate(tqdm(data_file.readlines())):
            if useindexes is not None and i not in useindexes:
                continue
            seq = []
            tokens = basic_tokenizer(line)

            for token in tokens:
                token_id = word_to_id.get(token, UNK_ID)
                seq.append(token_id)

            if max_len is not None and len(seq) > max_len:
                continue
            if len(seq) > longest:
                longest = len(seq)

            indexes.append(i)
            sequences.append(seq)
            lengths.append(len(seq))

    # Now that we know the longest example, we want to place all the numpy arrays
    # into one large matrix, and for each we want to keep track of the original sequence length
    num_examples = len(sequences)

    if max_len is None:
        max_len = longest

    processed = np.zeros((num_examples, max_len))

    logging.info("Building the unrolled np array...")
    for i, sequence in enumerate(tqdm(sequences)):
        for t, token_id in enumerate(sequence):
            processed[i, t] = token_id

    return processed, indexes, lengths


def build_dataset(prefix, id_to_word, word_to_id, glove_vectors, squad_root):
    context_path = pjoin(squad_root, prefix + ".context")
    question_path = pjoin(squad_root, prefix + ".question")
    span_path = pjoin(squad_root, prefix + ".span")

    # Setup a max_len here
    if prefix == "dev":
        context_max_len = None  # Do not truncate dev dataset
    else:
        context_max_len = flags.FLAGS.max_len
    context, ind, context_lens = tokens_to_ids(context_path, word_to_id, max_len=context_max_len)
    question, _, question_lens = tokens_to_ids(question_path, word_to_id, useindexes=ind)
    answer = np.zeros(context.shape[:2], dtype='int32')
    with gfile.GFile(span_path, "r") as span_file:
        for i, line in enumerate(span_file):
            if i in ind:
                # Find actual position
                shifted_index = ind.index(i)
                start, end = line.split()
                start, end = int(start), int(end)
                answer[shifted_index, start:end + 1] = 1

    # Write out the correct thing
    output_path = pjoin(squad_root, prefix + ".npz")
    logging.info("Saving all processed {} data to {}".format(prefix, output_path))
    if prefix == "dev":
        np.savez(output_path, context=context, question=question,
                 context_lens=context_lens, question_lens=question_lens)
    else:
        np.savez(output_path, context=context, question=question,
                 answer=answer, context_lens=context_lens, question_lens=question_lens)


def main(_):
    squad_dir = flags.FLAGS.squad_dir

    # Setup the files we will need
    train_question = pjoin(squad_dir, "train.question")
    train_context = pjoin(squad_dir, "train.context")

    val_question = pjoin(flags.FLAGS.squad_dir, "val.question")
    val_context = pjoin(flags.FLAGS.squad_dir, "val.context")

    # Build the vocabulary
    vocab_file = pjoin(flags.FLAGS.squad_dir, "vocab.dat")
    id_to_word, word_to_id = build_vocabulary([train_question, train_context, val_question, val_context], vocab_file)
    glove_vectors = build_glove(id_to_word)

    # Using the vocabulary and the GloVe vectors, we create a serialized version of the training
    # and validation set.
    build_dataset('train', id_to_word, word_to_id, glove_vectors, squad_dir)
    build_dataset('val', id_to_word, word_to_id, glove_vectors, squad_dir)
    build_dataset('dev', id_to_word, word_to_id, glove_vectors, squad_dir)


if __name__ == '__main__':
    flags.DEFINE_string("glove_dir", "data/dwr", "directory with GloVe files")
    flags.DEFINE_string("squad_dir", "data/squad", "directory to save preprocessed files")

    flags.DEFINE_integer("embed_size", 300, "size of embeddings to generate")
    flags.DEFINE_integer("max_len", 300, "maximum nubmber of timesteps for a sequence")
    app.run(main)
