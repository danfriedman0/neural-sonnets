# Sample from a trained LSTM language model

from __future__ import division
from __future__ import print_function

import sys
import argparse
import dill as pickle
import os

from copy import deepcopy
from timeit import default_timer as timer

import tensorflow as tf

import lstm_ops
import data_reader

parser = argparse.ArgumentParser(description="Sample from an LSTM language model")
parser.add_argument("save_dir", help="Directory with the checkpoints")
args = parser.parse_args()


def sample(save_dir):
    path_to_config = save_dir + "/config"
    if not os.path.isfile(path_to_config):
        raise IOError("Could not find " + path_to_config)
    with open(path_to_config, "rb") as f:
        gen_config = pickle.load(f)

    # # Load vocabulary encoder
    # glove_dir = '/Users/danfriedman/Box Sync/My Box Files/9 senior spring/gen/glove/glove.6B/glove.6B.50d.txt'
    # #glove_dir = '/data/corpora/word_embeddings/glove/glove.6B.50d.txt'
    if gen_config.use_glove:
        _, _, _, L = data_reader.glove_encoder(gen_config.glove_dir)
    else:
        L = None

    # Rebuild the model
    with tf.variable_scope("LSTM"):
        gen_model = lstm_ops.seq2seq_model(
                      encoder_seq_length=gen_config.d_len,
                      decoder_seq_length=1,
                      num_layers=gen_config.num_layers,
                      embed_size=gen_config.embed_size,
                      batch_size=gen_config.batch_size,
                      hidden_size=gen_config.hidden_size,
                      vocab_size=gen_config.vocab_size,
                      dropout=gen_config.dropout,
                      max_grad_norm=gen_config.max_grad_norm,
                      use_attention=gen_config.use_attention,
                      embeddings=L,
                      is_training=False,
                      is_gen_model=True,
                      token_type=gen_config.token_type,
                      reuse=False)

    with tf.Session() as session:
        saver = tf.train.Saver()
        saver.restore(session,tf.train.latest_checkpoint('./' + args.save_dir))

        def generate(description, temperature):
            return lstm_ops.generate_text_beam_search(
                        session=session,
                        model=gen_model,
                        encode=gen_config.encode,
                        decode=gen_config.decode,
                        description=description,
                        d_len=gen_config.d_len,
                        beam=5,
                        stop_length=gen_config.c_len,
                        temperature=temperature)

        seed = "Three huge birds wait outside of the window of a man's room. The man is talking on the phone."
        temp = 1.0

        print(generate(seed, temp))

        while raw_input("Sample again? ([y]/n): ") != "n":
            new_seed = raw_input("seed: ")
            if len(gen_config.encode(seed)) > gen_config.d_len:
                print(
                    "Description must be < {} tokens".format(gen_config.d_len))
                continue
            new_temp = raw_input("temp: ")

            if new_seed != "":
                seed = new_seed
            if new_temp != "":
                temp = float(new_temp)

            print(generate(seed, temp))


if __name__ == "__main__":
    sample(args.save_dir)

