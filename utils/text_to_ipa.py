# -*- coding: utf-8 -*-

# Use espeak to convert English text to IPA phonemes
import codecs
import re
import string
import cPickle as pickle
import sys
import os

from subprocess import check_output

# out = check_output(["espeak","--ipa","-q",line]).decode('utf-8')

def abs_path(rel_path):
    return os.path.abspath(rel_path)

def write_file_by_chunks(fn_in='data/sonnets.txt',
                         fn_out='data/sonnets_ipa.txt'):
    with codecs.open(abs_path(fn_in), 'r', encoding='utf-8') as f_in:
        text = f_in.read()

    punctuation = set(string.punctuation)
    punctuation.add('\n')
    punctuation.remove("'")
    
    i = 0
    j = 0
    out = []

    pad = lambda p: p if p == '\n' else ' ' + p + ' '

    while j < len(text):
        if j % 1000 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('{}/{}'.format(j, len(text)-1))
            sys.stdout.flush()
        if text[j] in punctuation:
            if i == j:
                out.append(pad(text[j]))
                i += 1
                j += 1
            else:
                s = text[i:j]
                ipa = check_output(["espeak","--ipa","-q",s]).decode('utf-8')
                out.append(ipa.strip())
                out.append(pad(text[j]))
                j += 1
                i = j
        else:
            j += 1

    final = ''.join(out)
    with codecs.open(abs_path(fn_out), 'w', encoding='utf-8') as f_out:
        f_out.write(final)
    print('\n')


def build_dict(infile, outfile):
    word_to_ipa = {}
    punc_set = set(string.punctuation)
    punc_set.add('\n')
    punc_set.add('iv')

    toks = []
    with codecs.open(infile, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.lower()
            _toks = line.split(' ')
            toks += _toks
    words = list(set([tok for tok in toks if tok not in punc_set]))

    print "{} tokens, {} unique words".format(len(toks), len(words))
    print "Building dict..."

    word_to_ipa = {}
    for i, word in enumerate(words):
        ipa = check_output(["espeak", "--ipa", "-q", word]).decode("utf-8")
        word_to_ipa[word] = ipa.strip()
        if i % 100 == 0:
            print "{}/{}".format(i, len(words))

    with open(outfile, "wb") as f_out:
        pickle.dump(word_to_ipa, f_out, pickle.HIGHEST_PROTOCOL)

    print "Done."


if __name__ == "__main__":
    write_file_by_chunks()
