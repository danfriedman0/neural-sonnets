# Trains the LSTM language model defined in lstm.py
# Dan Friedman 03/17
#
# Implementation based on:
#   http://cs224d.stanford.edu/assignment2/index.html
#   http://colah.github.io/posts/2015-08-Understanding-LSTMs/
#   https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py
# and
#   https://github.com/karpathy/char-rnn
from __future__ import division
from __future__ import print_function

import argparse
import dill as pickle
import os

from copy import deepcopy
from timeit import default_timer as timer

import tensorflow as tf
from tensorflow.python.client import timeline

import lstm_ops
import data_reader
from configs import Config

test_description = "Three huge birds wait outside of the window of a man's room. The man is talking on the phone."

parser = argparse.ArgumentParser(description="Train an LSTM language model")
parser.add_argument("--data_fn", help="Path to data file",
                    default="data/dataset.json")
parser.add_argument("--config_size", help="Config size {small, medium, large}",
                    default="small")
parser.add_argument("--temperature", help="Temperature for sampling [0,1.0]",
                    type=float, default=1.0)
parser.add_argument("--sample_every", help="How often to sample (in batches)",
                    type=int, default=1000)
parser.add_argument("--save_every", help="How often to save (in epochs)",
                    type=int, default=1)
parser.add_argument("--log_every", help="How often to log status (in batches)",
                    type=int, default=100)
parser.add_argument("--early_stopping",help="Stop after n epochs w. flat loss",
                    type=int, default=2)
parser.add_argument("--num_layers", help="Number of RNN layers",
                    type=int, default=None)
parser.add_argument("--batch_size", help="Batch size",
                    type=int, default=None)
parser.add_argument("--hidden_size", help="Size of the RNN hidden state",
                    type=int, default=None)
parser.add_argument("--dropout", help="keep_prob for dropout",
                    type=float, default=None)
parser.add_argument("--token_type", help="Predict words or chars",
                    default="words_lower")

parser.add_argument("--use_glove", help="Use glove vectors",
                    action="store_true", default=False)
parser.add_argument("--use_attention", help="LSTM with attention",
                    action="store_true", default=False)
parser.add_argument("--glove_dir", help="Where is glove",
                    default="/data/corpora/word_embeddings/glove/glove.6B.50d.txt")

parser.add_argument("--save_dir", help="Name of directory for saving models",
                    default="cv/test/")
parser.add_argument("--resume_from",help="Reload a model for more training",
                    default=None)

parser.add_argument("--debug", help="Debug", action="store_true",
                    default=False)

args = parser.parse_args()


# Some sample configurations

configs = {}

configs["test"] = Config(
    max_grad_norm = 5,
    num_layers = 1,
    hidden_size = 20,
    max_epochs = 1,
    max_max_epoch = 1,
    dropout = 1.0,
    batch_size = 5,
    embed_size = 16,
    token_type= "words_lower")

configs["small"] = Config(
    max_grad_norm = 5,
    num_layers = 2,
    hidden_size = 256,
    max_epochs = 32,
    max_max_epoch = 55,
    dropout = 0.9,
    batch_size = 50,
    embed_size = 50,
    token_type = "words_lower")

configs["medium"] = Config(
    max_grad_norm = 5,
    num_layers = 2,
    hidden_size = 512,
    max_epochs = 32,
    max_max_epoch = 55,
    dropout = 0.7,
    batch_size = 50,
    embed_size = 50,
    token_type = "words_lower")

configs["large"] = Config(
    max_grad_norm = 10,
    num_layers = 2,
    hidden_size = 1026,
    max_epochs = 32,
    max_max_epoch = 55,
    dropout = 0.5,
    batch_size = 50,
    embed_size = 50,
    token_type = "words_lower")

def train(config):
    # Load the data
    print("Loading data...")
    data = data_reader.load_data(config.data_fn)
    if args.debug:
        data = sorted(data, key=lambda d: len(d[0]))
        data = data[:10]

    # Split data
    num_train = int(0.8*len(data))
    train_data = data[:num_train]

    if config.use_glove:
        config.token_type = "glove"
        config.embed_size = 50
        encode, decode, vocab_size, L = data_reader.glove_encoder(
                                            config.glove_dir)
    else:
        L = None
        encode, decode, vocab_size = data_reader.make_encoder(
                                        train_data, config.token_type)
    
    config.encode = encode
    config.decode = decode
    config.vocab_size = vocab_size

    if config.token_type == "chars":
        max_c_len = 100
        max_d_len = 200
    else:
        max_c_len = 15 if args.debug else 25
        max_d_len = 50

    encoded_data = data_reader.encode_data(data, encode, max_c_len)
    encoded_data = [(d,cs) for d,cs in encoded_data if len(d) <= max_d_len]
    encoded_train = encoded_data[:num_train]
    encoded_valid = encoded_data[num_train:]

    # Padding width
    config.d_len = d_len = max([len(d) for d,_ in encoded_data])
    config.c_len = c_len = max([max([len(c) for c in cs])
                                for _,cs in encoded_data]) + 1
    print('Padding to {} and {}'.format(d_len, c_len))

    train_producer, num_train = data_reader.get_producer(
        encoded_train, config.batch_size, d_len, c_len)
    valid_producer, num_valid = data_reader.get_producer(
        encoded_valid, config.batch_size, d_len, c_len)

    print("Done. Building model...")

    if config.token_type == "chars":
        config.embed_size = vocab_size

    # Create a duplicate of the training model for generating text
    gen_config = deepcopy(config)
    gen_config.batch_size = 1
    gen_config.dropout = 1.0

    # Save gen_model config so we can sample later
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    path_to_model = os.path.join(args.save_dir, "config")
    with open(path_to_model, "wb") as f:
        pickle.dump(gen_config, f)

    path_to_index = os.path.join(args.save_dir, "index")
    with open(path_to_index, "w") as f:
        f.write("loss per epoch:\n")
        f.write("---------------\n")


    # Create training model
    with tf.variable_scope("LSTM") as scope:
        model = lstm_ops.seq2seq_model(
                  encoder_seq_length=d_len,
                  decoder_seq_length=c_len,
                  num_layers=config.num_layers,
                  embed_size=config.embed_size,
                  batch_size=config.batch_size,
                  hidden_size=config.hidden_size,
                  vocab_size=vocab_size,
                  dropout=config.dropout,
                  max_grad_norm=config.max_grad_norm,
                  use_attention=args.use_attention,
                  embeddings=L,
                  is_training=True,
                  is_gen_model=False,
                  token_type=config.token_type,
                  reuse=False)
        gen_model = lstm_ops.seq2seq_model(
                  encoder_seq_length=d_len,
                  decoder_seq_length=1,
                  num_layers=gen_config.num_layers,
                  embed_size=gen_config.embed_size,
                  batch_size=gen_config.batch_size,
                  hidden_size=config.hidden_size,
                  vocab_size=vocab_size,
                  dropout=gen_config.dropout,
                  max_grad_norm=gen_config.max_grad_norm,
                  use_attention=args.use_attention,
                  embeddings=L,
                  is_training=False,
                  is_gen_model=True,
                  token_type=config.token_type,
                  reuse=True)

    print("Done.")

    def generate():
        return lstm_ops.generate_text_beam_search(
                    session=session,
                    model=gen_model,
                    encode=gen_config.encode,
                    decode=gen_config.decode,
                    description=test_description,
                    d_len=gen_config.d_len,
                    beam=5,
                    stop_length=gen_config.c_len,
                    temperature=args.temperature)


    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as session:
        if args.resume_from is not None:
            reload_saver = tf.train.Saver()
            reload_saver.restore(session,
                tf.train.latest_checkpoint('./' + args.resume_from))

        best_val_pp = float('inf')
        best_val_epoch = 0

        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # Sample some text
        print(generate())

        for epoch in range(config.max_epochs):
            print('Epoch {}'.format(epoch))
            start = timer()

            # Train on the epoch and validate
            train_pp = lstm_ops.run_epoch(
                session, model, train_producer,
                num_train, args.log_every, args.sample_every, generate)
            print("Validating:")
            valid_pp = lstm_ops.run_epoch(
                session, model, valid_producer,
                num_valid, args.log_every, args.sample_every, generate,
                is_training=False)
            print("Validation loss: {}".format(valid_pp))

            # Save the model if validation loss has dropped
            if valid_pp < best_val_pp:
                with open(path_to_index, "a") as f:
                    f.write("{}: {}*\n".format(epoch, valid_pp))
                best_val_pp = valid_pp
                best_val_epoch = epoch
                path_to_ckpt = os.path.join(args.save_dir, "epoch.ckpt")
                print("Saving model to " + path_to_ckpt)
                saver.save(session, "./" + path_to_ckpt)

            # Otherwise just record validation loss in save_dir/index
            else:
                with open(path_to_index, "a") as f:
                    f.write("{}: {}\n".format(epoch, valid_pp))

            # Stop early if validation loss is getting worse
            if epoch - best_val_epoch > args.early_stopping:
                print("Stopping early")
                break

            print('Total time: {}\n'.format(timer() - start))
            print(generate())

        print(generate())


def main():
    config = configs[args.config_size]

    if args.num_layers is not None: config.num_layers = args.num_layers
    if args.batch_size is not None: config.batch_size = args.batch_size
    if args.hidden_size is not None: config.hidden_size = args.hidden_size
    if args.dropout is not None: config.dropout = args.dropout
    config.token_type = args.token_type
    config.use_attention = args.use_attention
    config.use_glove = args.use_glove
    config.glove_dir = args.glove_dir
    config.data_fn = args.data_fn

    train(config)


if __name__ == "__main__":
    main()
