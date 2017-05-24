# Functional RNN cells
# LSTM implementations based on:
#   http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 
#   https://github.com/karpathy/char-rnn
# and
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/rnn_cell_impl.py

from __future__ import division

import collections

import tensorflow as tf

LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ['c', 'h'])


def make_lstm_cell(input_size, state_size, batch_size, dropout):
    """Returns lstm function and zero_state"""

    # Concatenate all the parameters into one tensor to batch operations
    Wx_shape = (input_size, 4 * state_size)
    Wh_shape = (state_size, 4 * state_size)
    W_init = tf.random_uniform_initializer(-0.08, 0.08)
    b_shape = (4 * state_size,)

    # Initialize forget biases to 1.0 and all other biases to 0
    # (order is input gate, forget gate, candidate gate, output gate)
    init_b = ([0 for _ in range(state_size)] +
                        [1 for _ in range(state_size)] +
                        [0 for _ in range(2 * state_size)])
    b_init = tf.constant_initializer(init_b)

    # Define the lstm function
    def cell(input_x, state):
        Wx = tf.get_variable("Wx", shape=Wx_shape, initializer=W_init)
        Wh = tf.get_variable("Wh", shape=Wh_shape, initializer=W_init)
        b = tf.get_variable("b", shape=b_shape, initializer=b_init)

        (c_prev, h_prev) = state

        # Do all the linear combinations in one batch then split
        x_sum = tf.matmul(input_x, Wx)
        h_sum = tf.matmul(h_prev, Wh)
        all_sums = x_sum + h_sum + b

        s1, s2, s3, s4 = tf.split(all_sums, 4, axis=1)

        # i = input gate, f = forget gate, cn = candidate gate, o = output gate
        i = tf.sigmoid(s1)
        f = tf.sigmoid(s2)
        cn = tf.tanh(s3)
        o = tf.sigmoid(s4)

        c_new = f * c_prev + i * cn
        h_new = o * tf.tanh(c_new)

        # Only apply dropout to the non-recurrent layer (i.e. h_out)
        output = tf.nn.dropout(h_new, keep_prob=dropout)

        new_state = LSTMStateTuple(c_new, h_new)
        return output, new_state

    # Define the zero state
    init_c = tf.zeros((batch_size, state_size), dtype=tf.float32)
    init_h = tf.zeros((batch_size, state_size), dtype=tf.float32)
    zero_state = LSTMStateTuple(init_c, init_h)

    return cell, zero_state


def make_rnn_cell(input_size, state_size, batch_size, dropout):
    def cell(input_x, state):
        H = tf.get_variable("H", shape=(state_size, state_size))
        I = tf.get_variable("I", shape=(input_size, state_size))
        b = tf.get_variable("b", shape=(state_size,),
            initializer=tf.constant_initializer(0.0))

        c_prev, h_prev = state
        h_new = tf.matmul(h_prev, H) + tf.matmul(input_x, I) + b
        output = tf.nn.dropout(h_new)

        return output, LSTMStateTuple(c_prev, h_new)

    zero_state = tf.zeros((batch_size, state_size), dtype=tf.float32)

    return cell, zero_state


def make_stacked_rnn_cell(init_cells):
    """Input is a list of (cell, zero_state) pairs"""
    cells = [cell for cell, _ in init_cells]
    zero_states = [state for _, state in init_cells]

    def cell(input_x, state):
        cur_input = input_x
        new_states = []
        for i, cell in enumerate(cells):
            with tf.variable_scope("cell_%d" % i) as scope:
                cur_state = state[i]
                cur_input, new_state = cell(cur_input, cur_state)
                new_states.append(new_state)
        return cur_input, tuple(new_states)

    zero_state = tuple(zero_states)

    return cell, zero_state


def make_tf_lstm_cell(batch_size, input_size, hidden_size,
                      num_layers, dropout):
    def lstm_cell(hidden_size):
        return tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.LSTMCell(hidden_size), output_keep_prob=dropout)
    cells = [lstm_cell(hidden_size) for _ in range(num_layers)]
    cell = tf.contrib.rnn.MultiRNNCell(cells)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    return cell, init_state

