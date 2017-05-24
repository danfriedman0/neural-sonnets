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





def write_file_by_words(infile, outfile):
    with codecs.open(infile, "r", encoding="utf-8") as f_in:
        lines = f_in.readlines()

    word_to_ipa = {}

    pat = re.compile('(\w+\'?\w+|\w|[^ \w\'])')
    with codecs.open(outfile, "w", encoding="utf-8") as f_out:
        for i, line in enumerate(lines):
            words = re.findall(pat, line)
            phones = []
            for word in words:
                word = word.lower()
                if word in word_to_ipa:
                    phones.append(word_to_ipa[word])
                elif len(word) == 1 and word[0] not in string.letters:
                    word_to_ipa[word] = word
                    phones.append(word)
                else:
                    ipa = check_output(["espeak","--ipa","-q",word]).decode('utf-8')
                    ipa = ipa.strip()
                    word_to_ipa[word] = ipa
                    phones.append(ipa)
            line_out = " ".join(phones)
            f_out.write(line_out)
            if i % 1000 == 0:
                print("{}/{}".format(i, len(lines)))

    with open("/Users/danfriedman/Box Sync/My Box Files/9 senior spring/gen/lstm/data/_word_to_ipa.pkl", "wb") as f:
        pickle.dump(word_to_ipa, f, pickle.HIGHEST_PROTOCOL)



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


def tokenize_file(infile, outfile):
    pat = re.compile('(\w+\'?\w+|\w|[^ \w\'])')
    with codecs.open(infile, "r", encoding="utf-8") as f_in:
        s = f_in.read()

    with codecs.open(outfile, "w", encoding="utf-8") as f_out:
        toks = re.findall(pat, s)
        f_out.write(' '.join(toks))


def test():
    infile = ''
    outfile = ''
    with open(infile, "rb") as f:
        word_to_ipa = pickle.load(f)

    pat = re.compile('(\w+\'?\w+|\w|[^ \w\'])')
    with codecs.open(outfile, "r", encoding="utf-8") as f:
        s = f.read()
        words = re.findall(pat, s)

    cnt = 0
    not_found = set()
    for i, word in enumerate(words):
        if word.lower() in word_to_ipa:
            cnt += 1
        else:
            not_found.add(word)

    print("{}/{}".format(cnt, len(words)))
    print(not_found)

    for tok in not_found:
        word_to_ipa[tok] = tok

    with open(infile, "wb") as f:
        word_to_ipa = pickle.dump(word_to_ipa, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    write_file_by_chunks()
