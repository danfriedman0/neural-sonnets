# -*- coding: utf-8 -*-
# quick scripts for processing text
import re

def clean_couplets():
    f_in = 'data/couplets.txt'
    f_out = 'data/_couplets.txt'
    with open(f_in, 'r') as f:
        poems = f.read().split('\n*\n')
    out = []
    for poem in poems:
        stanzas = poem.split('\n\n')
        stanzas = [stanza.strip() for stanza in stanzas]
        stanzas = [stanza for stanza in stanzas if len(stanza) > 0]
        clean = '\n\n'.join(stanzas)
        out.append(clean)
    with open(f_out, 'w') as f:
        f.write('\n\n\n'.join(out))


def tokenize():
    f_in = 'data/couplets.txt'
    f_out = 'data/_couplets.txt'
    with open(f_in, 'r') as f:
        lines = f.readlines()
    pat = re.compile('(\w+\'?\w+|\w|[^ \w\'\n])')
    tokenize = lambda s: re.findall(pat, s)
    with open(f_out, 'w') as f:
        for line in lines:
            if line == '\n':
                f.write(line)
            else:
                tokens = tokenize(line)
                f.write(' '.join(tokens) + '\n')


def get_rhymes_from_couplets():
    f_in = 'data/couplets.txt'
    with open(f_in, 'r') as f:
        poems = f.read().split('\n\n\n')
    print('{} poems'.format(len(poems)))
    for poem in poems:
        stanzas = poem.split('\n\n')
        print('  {} stanzas'.format(len(stanzas)))
        for stanza in stanzas:
            lines = stanza.split('\n')
            print('    {} lines'.format(len(lines)))
            if len(lines) % 2 != 0 and len(lines) != 3:
                return stanza



if __name__ == '__main__':
    tokenize()
