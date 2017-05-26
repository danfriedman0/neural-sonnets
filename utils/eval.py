# -*- coding: utf-8 -*-
# utilities for evaluation sonnets
from __future__ import division
import string
from collections import Counter


PUNCTUATION = set(string.punctuation)


def get_rhyme_word(line):
    for tok in reversed(line):
        if tok in PUNCTUATION:
            continue
        return tok
    return None


def get_rhyme(line):
    for tok in reversed(line):
        if tok in PUNCTUATION:
            continue
        if 'ˌ' in tok:
            return tok.split('ˌ')[-1]
        elif 'ˈ' in tok:
            return tok.split('ˈ')[-1]
        else:
            return tok
    return None


def tokenize(raw_sonnet):
    lines = raw_sonnet.split('\n')
    tokens = [line.strip().split(' ') for line in lines]
    return tokens


def get_all_rhymes(sonnets):
    all_rhymes = [[get_rhyme(line) for line in tokenize(sonnet)]
              for sonnet in sonnets]
    return all_rhymes


def get_strict_pattern(rhymes):
    pattern = []
    d = {}
    for rhyme in rhymes:
        if rhyme not in d:
            d[rhyme] = chr(ord('A') + len(d))
        pattern.append(d[rhyme])
    return ''.join(pattern)


def edit_distance(a, b):
    m = len(a); n = len(b)
    dp = [[0 for _ in xrange(n+1)] for _ in xrange(m+1)]
    for i in xrange(m+1):
        for j in xrange(n+1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = 1
            elif a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j-1],
                                   dp[i][j-1],
                                   dp[i][j-1])
    return dp[m][n]


def check_shakespearean(rhymes):
    key_rhymes = [rhymes[i] for i in [0,1,4,5,8,9,12]]
    d = {}
    for a in key_rhymes:
        if a not in d:
            d[a] = chr(ord('A') + len(d))
    out = []
    for rhyme in rhymes:
        if rhyme not in d:
            d[rhyme] = chr(ord('A') + len(d))
        out.append(d[rhyme])
    pat = ''.join(out)
    return pat, edit_distance(pat, 'ABABCDCDEFEFGG')


def check_petrarchan(rhymes):
    key_rhymes = [rhymes[i] for i in [0,1,8,9]]
    d = {}
    for a in key_rhymes:
        if a not in d:
            d[a] = chr(ord('A') + len(d))
    out = []
    for rhyme in rhymes:
        if rhyme not in d:
            d[rhyme] = chr(ord('A') + len(d))
        out.append(d[rhyme])
    pat = ''.join(out)
    octave = pat[:8]
    sestet = pat[8:]
    octave_score = edit_distance(octave, 'ABBAABBA')
    possibilities = ['CDECDE', 'CDCDCD', 'CDDCDD', 'CDDECE', 'CDCDCD']
    sestet_score = min([edit_distance(sestet, p) for p in possibilities])
    return pat, octave_score + sestet_score


def get_patterns_and_counts(sonnets):
    all_rhymes = get_all_rhymes(sonnets)
    patterns = [get_strict_pattern(rhymes) for rhymes in all_rhymes]
    counts = Counter(patterns)
    print('Found {} unique patterns in {} sonnets'.format(
          len(counts), len(sonnets)))
    return counts


def get_scores(sonnets, scheme):
    all_rhymes = get_all_rhymes(sonnets)
    if scheme == 'shakespearean' or scheme == 'S':
        checks = [check_shakespearean(rhymes) for rhymes in all_rhymes]
    elif scheme == 'petrarchan' or scheme == 'P':
        checks = [check_petrarchan(rhymes) for rhymes in all_rhymes]
    patterns, scores = zip(*checks)
    counts = Counter(patterns)
    print('Found {} unique patterns in {} sonnets'.format(
          len(counts), len(sonnets)))
    print('Avg score: {}'.format(sum(scores)/len(scores)))
    return checks, counts


def classify(sonnets):
    all_rhymes = get_all_rhymes(sonnets)
    res = []
    for rhymes in all_rhymes:
        s_pat, s_score = check_shakespearean(rhymes)
        p_pat, p_score = check_petrarchan(rhymes)
        res.append((s_score, p_score))
    return res


