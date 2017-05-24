# -*- coding: utf-8 -*-

# Use espeak to convert English text to IPA phonemes
import argparse
import codecs
import re
import string
import cPickle as pickle

from subprocess import check_output


parser = argparse.ArgumentParser()
parser.add_argument("infile", help="infile")
parser.add_argument("outfile", help="outfile")
args = parser.parse_args()

def write_file():
	lines = codecs.open(args.infile, "r", encoding="utf-8").readlines()
	outfile = codecs.open(args.outfile, "w", encoding="utf-8")

	punc_set = set([',','.','?','!',';',':'])

	for i,line in enumerate(lines):
		punc = [c for c in line if c in punc_set]
		out = check_output(["espeak","--ipa","-q",line]).decode('utf-8')
		chunks = [s.lstrip() for s in out.split('\n')]
		phones = []
		ptr = 0
		while ptr < len(punc) and ptr < len(chunks):
			phones.append(chunks[ptr])
			phones.append(punc[ptr])
			ptr += 1
		if ptr < len(chunks):
			phones += chunks[ptr:]

		outfile.write(" ".join(phones) + "\n")
		if i % 100 == 0:
			print("{}/{}".format(i,len(lines)))

	outfile.close()


def write_file_by_words():
	with codecs.open(args.infile, "r", encoding="utf-8") as f_in:
		lines = f_in.readlines()

	word_to_ipa = {}

	pat = re.compile('(\w+\'?\w+|\w|[^ \w\'])')
	with codecs.open(args.outfile, "w", encoding="utf-8") as f_out:
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



def build_dict():
	word_to_ipa = {}
	punc_set = set(string.punctuation)
	punc_set.add('\n')
	punc_set.add('iv')

	toks = []
	with codecs.open(args.infile, "r", encoding="utf-8") as f_in:
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

	with open(args.outfile, "wb") as f_out:
		pickle.dump(word_to_ipa, f_out, pickle.HIGHEST_PROTOCOL)

	print "Done."


def tokenize_file():
	pat = re.compile('(\w+\'?\w+|\w|[^ \w\'])')
	with codecs.open(args.infile, "r", encoding="utf-8") as f_in:
		s = f_in.read()

	with codecs.open(args.outfile, "w", encoding="utf-8") as f_out:
		toks = re.findall(pat, s)
		f_out.write(' '.join(toks))


def test():
	with open(args.infile, "rb") as f:
		word_to_ipa = pickle.load(f)

	pat = re.compile('(\w+\'?\w+|\w|[^ \w\'])')
	with codecs.open(args.outfile, "r", encoding="utf-8") as f:
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

	with open(args.infile, "wb") as f:
		word_to_ipa = pickle.dump(word_to_ipa, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
	#write_file_by_words()
