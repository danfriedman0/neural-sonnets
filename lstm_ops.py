# Long Short Term Memory language model
# Dan Friedman, 03/2017
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

import sys
import re

from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import data_reader
import layers
import cells


UNK = "[UNK]"
NONE = "[0]"
START = "[START]"
STOP = "[STOP]"


def load_data(fn,
              batch_size,
              token_type,
              min_count=3,
              split_size=0.8,
              debug=False):
    data = data_reader.load_data(fn)
    if debug:
        data = data[:10]

    # Split data
    num_train = int(split_size*len(data))
    train_data = data[:num_train]
    encode, decode, vocab_size = data_reader.make_encoder(
                                    train_data, token_type, min_count)
    
    encoded_data = data_reader.encode_data(data, encode)
    encoded_data = [(d,cs) for d,cs in encoded_data if len(d) <= 50]
    encoded_train = encoded_data[:num_train]
    encoded_valid = encoded_data[num_train:]

    # Padding width
    d_len = max([len(d) for d,_ in encoded_data])
    c_len = max([max([len(c) for c in cs]) for _,cs in encoded_data]) + 1
    print('Padding to {} and {}'.format(d_len, c_len))

    train_producer, num_train = data_reader.get_producer(
        encoded_train, batch_size, d_len, c_len)
    valid_producer, num_valid = data_reader.get_producer(
        encoded_valid, batch_size, d_len, c_len)
    return (train_producer, valid_producer,
            num_train, num_valid,
            encode, decode,
            vocab_size, d_len, c_len)


def seq2seq_model(encoder_seq_length,
                  decoder_seq_length,
                  num_layers,
                  embed_size,
                  batch_size,
                  hidden_size,
                  vocab_size,
                  dropout,
                  max_grad_norm,
                  use_attention,
                  embeddings,
                  is_training,
                  is_gen_model,
                  token_type,
                  reuse):
    lstm_encoder, init_state = layers.make_lstm(
                            embed_size, hidden_size, batch_size,
                            encoder_seq_length, num_layers, dropout)
    if use_attention:
        lstm_decoder, _ = layers.make_lstm_with_attention(
                            embed_size+hidden_size, hidden_size, batch_size,
                            decoder_seq_length, num_layers, dropout)
    else:
        lstm_decoder, _ = layers.make_lstm(
                            embed_size, hidden_size, batch_size,
                            decoder_seq_length, num_layers, dropout)

    encoder_input_placeholder = tf.placeholder(tf.int32,
        shape=(batch_size, encoder_seq_length), name="encoder_inputs")
    decoder_input_placeholder = tf.placeholder(tf.int32,
        shape=(batch_size, decoder_seq_length), name="decoder_inputs")
    labels_placeholder = tf.placeholder(tf.int32,
        shape=(batch_size, decoder_seq_length), name="labels")

    if token_type == "chars":
        encoder_inputs = tf.one_hot(encoder_input_placeholder,
                                    vocab_size, dtype=tf.float32)
        decoder_inputs = tf.one_hot(decoder_input_placeholder,
                                    vocab_size, dtype=tf.float32)
    else:
        encoder_inputs = layers.embed_inputs(encoder_input_placeholder,
                            vocab_size, embed_size, embeddings, reuse=reuse)
        decoder_inputs = layers.embed_inputs(decoder_input_placeholder,
                            vocab_size, embed_size, embeddings, reuse=True)

    with tf.variable_scope("Encoder", reuse=reuse):
        encoder_outputs, encoding = lstm_encoder(
                                        init_state, encoder_inputs, reuse)
        Hs = tf.stack(encoder_outputs, axis=1)
    with tf.variable_scope("Decoder", reuse=reuse):
        if use_attention:
            decoder_outputs, final_state = lstm_decoder(
                                                encoding, decoder_inputs,
                                                Hs, reuse)
        else:
            decoder_outputs, final_state = lstm_decoder(
                                                encoding, decoder_inputs,
                                                reuse)

    logits, predictions = layers.project_output(
        decoder_outputs, hidden_size, vocab_size, reuse)

    if is_training:
        loss = layers.calculate_sequence_loss(logits, labels_placeholder,
                    batch_size, decoder_seq_length, vocab_size)
        optimizer = tf.train.AdamOptimizer()
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss, tvars), max_grad_norm)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
    else:
        train_op = loss = None

    model = {
        "encoder_input_placeholder": encoder_input_placeholder,
        "decoder_input_placeholder": decoder_input_placeholder,
        "labels_placeholder": labels_placeholder,
        "init_state": init_state,
        "encoder_outputs": encoder_outputs,
        "encoding": encoding,
        "final_state": final_state,
        "predictions": predictions,
        "loss": loss,
        "train_op": train_op
    }

    return model


def tf_seq2seq_model(encoder_seq_length,
                     decoder_seq_length,
                     num_layers,
                     embed_size,
                     batch_size,
                     hidden_size,
                     vocab_size,
                     dropout,
                     max_grad_norm,
                     use_attention,
                     embeddings,
                     is_training,
                     is_gen_model,
                     reuse):
    encoder_input_placeholder = tf.placeholder(tf.int32,
        shape=(batch_size, encoder_seq_length), name="encoder_inputs")
    decoder_input_placeholder = tf.placeholder(tf.int32,
        shape=(batch_size, decoder_seq_length), name="decoder_inputs")
    labels_placeholder = tf.placeholder(tf.int32,
        shape=(batch_size, decoder_seq_length), name="labels")

    encoder_inputs = layers.embed_inputs(encoder_input_placeholder,
                        vocab_size, embed_size, embeddings, reuse=reuse)
    decoder_inputs = layers.embed_inputs(decoder_input_placeholder,
                        vocab_size, embed_size, embeddings, reuse=True)

    # cells
    encoder_cell, init_state = cells.make_tf_lstm_cell(
                                    batch_size, embed_size, hidden_size,
                                    num_layers, dropout)
    decoder_cell, _ = cells.make_tf_lstm_cell(
                            batch_size, embed_size, hidden_size,
                            num_layers, dropout)

    with tf.variable_scope("Encoder", reuse=reuse):
        encoder_outputs, encoding = tf.nn.dynamic_rnn(
            cell=encoder_cell,
            inputs=encoder_inputs,
            sequence_length=tuple(
                encoder_seq_length for _ in xrange(batch_size)),
            dtype=tf.float32)
    with tf.variable_scope("Decoder", reuse=reuse):
        decoder_outputs, final_state = tf.nn.dynamic_rnn(
            cell=decoder_cell,
            inputs=decoder_inputs,
            sequence_length=tuple(
                decoder_seq_length for _ in xrange(batch_size)),
            initial_state=encoding,
            dtype=tf.float32)

    logits, predictions = layers.project_output(
        decoder_outputs, hidden_size, vocab_size, reuse)

    if is_training:
        loss = layers.calculate_sequence_loss(logits, labels_placeholder,
                    batch_size, decoder_seq_length, vocab_size)
        optimizer = tf.train.AdamOptimizer()
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss, tvars), max_grad_norm)
        train_op = optimizer.apply_gradients(zip(grads, tvars))
    else:
        train_op = loss = None

    model = {
        "encoder_input_placeholder": encoder_input_placeholder,
        "decoder_input_placeholder": decoder_input_placeholder,
        "labels_placeholder": labels_placeholder,
        "init_state": init_state,
        "encoder_outputs": encoder_outputs,
        "encoding": encoding,
        "final_state": final_state,
        "predictions": predictions,
        "loss": loss,
        "train_op": train_op
    }

    return model


def run_epoch(session, model, data_producer, total_steps, log_every,
              sample_every, generate, is_training=True):
    """
    Model should contain:
        input_placeholder, labels_placeholder,
        init_state, final_state,
        predictions, loss, train_op
    """
    total_loss = []

    fetches = {}
    fetches["loss"] = model["loss"]
    fetches["final_state"] = model["final_state"]
    if model["train_op"] is not None and is_training:
        fetches["train_op"] = model["train_op"]

    state = session.run(model["init_state"])

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    start = timer()
    for step, (d, x, y) in enumerate(data_producer()):

        feed_dict = {}
        feed_dict[model["encoder_input_placeholder"]] = d
        feed_dict[model["decoder_input_placeholder"]] = x
        feed_dict[model["labels_placeholder"]] = y
        for i, (c, h) in enumerate(model["init_state"]):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        if step % sample_every == 0:
            vals = session.run(fetches, feed_dict,
                options=run_options, run_metadata=run_metadata)
        else:
            vals = session.run(fetches, feed_dict)
        state = vals["final_state"]
        loss = vals["loss"]

        total_loss.append(loss)

        if log_every and step % log_every == 0:
            time = (timer() - start) / log_every
            log_msg = '{} / {} : pp = {}'.format(
                step, total_steps, np.exp(np.mean(total_loss)))
            if step > 0:
                log_msg += '    {}s/iter'.format(time)
            sys.stdout.write(log_msg + '\r')
            sys.stdout.flush()
            start = timer()
        if sample_every and step > 0 and step % sample_every == 0:
            print(generate())
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline.json', 'w') as f:
              f.write(ctf)
    
    return np.exp(np.mean(total_loss))


def predict(session, model, state, encoder_outputs, x):
    fetches = {}
    fetches["final_state"] = model["final_state"]
    fetches["predictions"] = model["predictions"]

    feed_dict = {}
    feed_dict[model["decoder_input_placeholder"]] = x
    for i, (c,h) in enumerate(model["encoding"]):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h
    for i, h in enumerate(model["encoder_outputs"]):
        feed_dict[h] = encoder_outputs[i]
    # feed_dict[model["encoder_outputs"]] = encoder_outputs

    vals = session.run(fetches, feed_dict)
    state = vals["final_state"]
    predictions = vals["predictions"]

    return state, predictions


def get_encoder_outputs(session, model, state, encoder_inputs):
    feed_dict = {model["encoder_input_placeholder"]: [encoder_inputs]}
    fetches = [model["encoding"], model["encoder_outputs"]]
    for i, (c,h) in enumerate(model["init_state"]):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h
    encoding, encoder_outputs = session.run(fetches, feed_dict)
    return encoding, encoder_outputs


def generate_text(session, model, encode, decode, description, d_len, 
                  stop_length=25, stop_tokens=[STOP, NONE],
                  temperature=1.0):
    init_state = session.run(model["init_state"])
    encoder_inputs = data_reader.pad(encode(description), d_len, 'left')

    state, encoder_outputs = get_encoder_outputs(
                                    session, model, init_state, encoder_inputs)
    start = encode("")[0]
    # state, predictions = predict(session, model,
    #                              encoding, encoder_outputs,
    #                              [[start]])
    # output_ids = [data_reader.sample(predictions[0], temperature=temperature)]
    # inputs = encode(decode(output_ids))
    inputs = [start]
    output_ids = []
    for i in range(stop_length):
        x = inputs[-1]
        state, predictions = predict(session, model,
                                     state, encoder_outputs,
                                     [[x]])
        next_id = data_reader.sample(predictions[0], temperature=temperature)
        output_ids.append(next_id)
        output = decode([next_id])
        inputs.append(next_id)
        if stop_tokens and output in stop_tokens:
            break
    output_text = "[" + description + "]\n"
    output_text += decode(output_ids)
    return output_text


class Node(object):
    def __init__(self, idx, prob, state, prev):
        self.idx = idx
        self.prob = prob
        self.state = state
        self.prev = prev

    def get_outputs(self):
        if self.prev is None:
            return [self.idx]
        else:
            outputs = self.prev.get_outputs()
            outputs.append(self.idx)
            return outputs

    def __repr__(self):
        if self.prev is None:
            return str((self.idx, self.prob, None))
        else:
            return str((self.idx, self.prob, self.prev.idx))


def generate_text_beam_search(session, model, encode, decode, description, 
                              d_len, beam=5, stop_length=25, temperature=1.0,
                              get_output_tokens=False, deterministic=False):
    init_state = session.run(model["init_state"])
    encoder_inputs = data_reader.pad(encode(description), d_len, 'left')

    encoding, encoder_outputs = get_encoder_outputs(
                                    session, model, init_state, encoder_inputs)
    start = encode("")[0]
    state, predictions = predict(session, model,
                                 encoding, encoder_outputs,
                                 [[start]])
    cur_pairs = data_reader.beam_sample(predictions[0], beam, temperature,
                                        deterministic)
    cur_nodes = []
    for idx,prob in cur_pairs:
        cur_nodes.append(Node(idx, prob, state, None))

    for _ in xrange(stop_length):
        next_level = []
        # print('\n'.join([str(node) for node in cur_nodes]))
        # print('-'*15)
        for node in cur_nodes:
            state, predictions = predict(session, model,
                                         node.state, encoder_outputs,
                                         [[node.idx]])
            pairs = data_reader.beam_sample(predictions[0], beam, temperature,
                                            deterministic)
            next_level += [Node(idx, prob+node.prob, state, node)
                           for idx,prob in pairs]
        cur_nodes = sorted(next_level, key=lambda node: node.prob)[-beam:]

    outputs = [node.get_outputs() for node in cur_nodes]
    decoded_outputs = [decode(output) for output in outputs]
    if get_output_tokens:
        return decoded_outputs[-1]
    output_text = "[" + description + "]\n"
    output_text += decoded_outputs[-1]
    return output_text

