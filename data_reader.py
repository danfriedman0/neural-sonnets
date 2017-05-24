# Read input file for lstm.py
# https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import json
import random

import numpy as np

from nltk import word_tokenize

UNK = "[UNK]"
NONE = "[0]"
START = "[START]"
STOP = "[STOP]"

# Input to the model is a 3-tuple:
#   (description, caption_x, caption_y)
#   caption_x is the caption, caption_y is the caption right-shifted by one
#       (i.e. the labels)
# First need to tokenize and encode everything, do that in one step
# Then need to zero pad to get fixed length
# Then make batches


def load_data(fn):
    # Data is a list of (description, [captions]) pairs
    with open(fn, 'r') as f:
        data = json.load(f)
    return data


def get_tokenizer(token_type="chars"):
    if token_type == "chars" or token_type == "char-cnn":
        tokenize = lambda s: list(s)
        join = lambda toks: ''.join(toks)
    elif token_type == "tokens":
        tokenize = lambda s: s.split(' ')
        join = lambda toks: ' '.join(toks)
    elif token_type == "words":
        pat = re.compile('(\w+\'?\w+|\w|[^ \w\'])')
        tokenize = lambda s: re.findall(pat, s)
        join = lambda toks: ' '.join(toks)
    elif token_type == "words_lower":
        pat = re.compile('(\w+\'?\w+|\w|[^ \w\'])')
        tokenize = lambda s: [w.lower() for w in re.findall(pat, s)]
        join = lambda toks: ' '.join(toks)
    elif token_type == "glove":
        tokenize = lambda s: s.lower().split(' ')
        join = lambda toks: ' '.join(toks)
    else:
        raise ValueError("Invalid token type: {}".format(token_type))
    return tokenize, join


def make_encoder(data, token_type="words", min_count=0):
    tokenize, join = get_tokenizer(token_type)
    cnt = collections.Counter()
    for d, cs in data:
        cnt.update(tokenize(d))
        for c in cs:
            cnt.update(tokenize(c))
    tokens = [t for t in cnt if cnt[t] > min_count]
    id_to_word = dict(enumerate(sorted(tokens), start=1))
    id_to_word[0] = NONE
    id_to_word[len(id_to_word)] = UNK
    id_to_word[len(id_to_word)] = START
    word_to_id = {w:i for i,w in id_to_word.iteritems()}
    vocab_size = len(word_to_id)
    print("Vocab size: {}".format(vocab_size))

    def encode(seq):
        words = tokenize(seq)
        ids = [word_to_id[START]]
        for word in words:
            if word not in word_to_id:
                word = UNK
            ids.append(word_to_id[word])
        return ids

    def decode(ids):
        words = []
        for id_ in ids:
            if id_ not in id_to_word:
                id_ = 0
            words.append(id_to_word[id_])
        return join(words)

    return encode, decode, vocab_size


def load_glove_vectors(fn):
    """
    Return vectors and word_to_id
    L \in M_{40003 x 50}(R) (add rows for special chars)
    """
    L = np.zeros((400004, 50), dtype=np.float32)
    L[1,:] = np.random.rand(50)
    L[2,:] = np.random.rand(50)
    L[3,:] = np.random.rand(50)
    word_to_id = {
        NONE: 0,
        UNK: 1,
        START: 2,
        STOP: 3
    }

    with open(fn, 'r') as f:
        for i,line in enumerate(f, start=3):
            if i == 3:
                continue
            delim = line.index(' ')
            word = line[:delim]
            embed = np.fromstring(line[delim+1:], dtype=np.float32, sep=' ')
            word_to_id[word] = i
            L[i,:] = embed

    return L, word_to_id


def glove_encoder(fn='/data/corpora/word_embeddings/glove/glove.6B.50d.txt'):
    tokenize, join = get_tokenizer("glove")
    L, word_to_id = load_glove_vectors(fn)
    id_to_word = {i:w for w,i in word_to_id.iteritems()}

    def encode(seq):
        words = tokenize(seq)
        ids = [word_to_id[START]]
        for word in words:
            if word not in word_to_id:
                word = UNK
            ids.append(word_to_id[word])
        ids.append(word_to_id[STOP])
        return ids

    def decode(ids):
        words = []
        for id_ in ids:
            if id_ not in id_to_word:
                words.append(UNK)
            else:
                words.append(id_to_word[id_])
        return join(words)

    return encode, decode, L.shape[0], L



def encode_data(data, encode, max_len=25):
    encoded_data = []
    for description,captions in data:
        d = encode(description)
        cs = [encode(caption) for caption in captions]
        if max_len is not None:
            cs = [c for c in cs if len(c) < max_len]
        encoded_data.append((d, cs))
    return encoded_data


def pad(sequence, padded_len, pad_side):
    pad_size = max(0,padded_len - len(sequence))
    if pad_side == 'left':
        pad_width = (pad_size, 0)
    else:
        pad_width = (0, pad_size)
    seq = np.array(sequence, dtype=np.int32)
    padded_seq = np.pad(seq, pad_width, mode='constant')
    return padded_seq


def pad_data(data, d_len, c_len):
    padded_data = []
    for description,captions in data:
        padded_description = pad(description, d_len, 'left')
        padded_captions = [pad(caption, c_len+1, 'right')
                           for caption in captions]
        padded_data.append((padded_description, padded_captions))
    return padded_data


def get_training_pairs(data):
    pairs = []
    descriptions = []
    for i,(description,captions) in enumerate(data):
        pairs += [(i,caption) for caption in captions]
        descriptions.append(description)
    random.shuffle(pairs)
    ids, captions = zip(*pairs)
    return np.array(descriptions), ids, np.array(captions)


def batch_producer(d_data, ids, c_data, batch_size):
    """
    Yields
        d: descriptions (batch_size, d_len)
        c_x: caption inputs (batch_size, c_len)
        c_y: caption labels (batch_size, c_len)
    """
    data_len = c_data.shape[0]
    num_batches = data_len // batch_size
    if num_batches == 0:
        raise ValueError("num_batches == 0, decrease batch_size")
    for i in xrange(num_batches):
        d_ids = ids[batch_size*i:batch_size*(i+1)]
        d = np.take(d_data, d_ids, axis=0)
        c = c_data[batch_size*i:batch_size*(i+1), :]
        c_x = c[:, 0:-1]
        c_y = c[:, 1:]
        yield (d, c_x, c_y)


def get_producer(data, batch_size, d_len, c_len):
    padded_data = pad_data(data, d_len, c_len)
    d_data, ids, c_data = get_training_pairs(padded_data)
    def producer():
        return batch_producer(d_data, ids, c_data, batch_size)
    return producer, c_data.shape[0] // batch_size


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

def beam_sample(a, beam, temperature=1.0, deterministic=False):
    # helper function to sample an index from a probability array
    # from https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    if deterministic:
        sorted_ids = np.argsort(a)
        top_ids = sorted_ids[-beam:]
    else:
        top_ids = np.random.choice(np.arange(a.shape[0]),
                    size=beam, replace=False, p=a)
    pairs = [(i, np.log(a[i])) for i in top_ids]
    return pairs

