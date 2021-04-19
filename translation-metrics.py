#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from rouge import FilesRouge
# from SARI import *


def tokenise(reference_corpus,generated_corpus,reftokens,hyptokens):
	'''
	reference_corpus = filepath to reference corpus
	generated corpus = filepath to generated corpus (either simplified or translated)
	reftokens = pre-processing tokens from the reference transcription to be passed to BLEU
	hyptokens = pre-processing tokens from the hypothesis transcription to be passed to BLEU
	'''
	with open(reference_corpus,'r') as refs, open(generated_corpus,'r') as hyps:
		for line in refs:
			reftokens.append(nltk.word_tokenize(line))
		for line in hyps:
			hyptokens.append(nltk.word_tokenize(line))
	return (reftokens,hyptokens)

def bleu(reftokens,hyptokens,outmetrics):
	'''
	reftokens = pre-processing tokens from the reference transcription to be passed to BLEU
	hyptokens = pre-processing tokens from the hypothesis transcription to be passed to BLEU
	outmetrics = dictionary to write scores to
	'''
	score_list_one = np.zeros(shape=(len(hyptokens)))
	score_list_two = np.zeros(shape=(len(hyptokens)))
	score_list_three = np.zeros(shape=(len(hyptokens)))
	score_list_four = np.zeros(shape=(len(hyptokens)))
	score_list_ten = np.zeros(shape=(len(hyptokens)))
	cnt = 0
	for refs,hyps in zip(reftokens,hyptokens):
		score_one = sentence_bleu([refs],hyps,weights=(1,0,0,0))
		score_list_one[cnt] = score_one
		score_two = sentence_bleu([refs],hyps,weights=(0.5,0.5,0,0))
		score_list_two[cnt] = score_two
		score_three = sentence_bleu([refs],hyps,weights=(0.33,0.33,0.33,0.33))
		score_list_three[cnt] = score_three
		score_four = sentence_bleu([refs],hyps,weights=(0.25,0.25,0.25,0.25))
		score_list_four[cnt] = score_four
		score_ten = sentence_bleu([refs],hyps,weights=(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1))
		score_list_ten[cnt] = score_ten
		cnt += 1
#	print('\n\n\n 1-GRAM = ',score_list_one.mean(),'\n\n')
	outmetrics['BLEU-1'] = score_list_one.mean()
	outmetrics['BLEU-2'] = score_list_two.mean()
	outmetrics['BLEU-3'] = score_list_three.mean()
	outmetrics['BLEU-4'] = score_list_four.mean()
	outmetrics['BLEU-10'] = score_list_ten.mean()

def ter():
	# could import utils_metrics.py from current dir and implement the main function?
	pass

def rouge(reference_corpus,generated_corpus):
	fr = FilesRouge()
	scores = fr.get_scores(reference_corpus,generated_corpus)
	print(scores,'\n\n\n')

def sari():
	'''
	Import SARI.py functionality to here, append score to dict
	'''
	pass

def tojson(output_file,outmetrics):
	'''
	outmetrics = dictionary containing scores from metrics
	output_file = JSON where scores will be written from dict
	'''
	with open (output_file,'w') as out:
		json.dump(outmetrics,out,indent=4)
	print('\n\n\nWritten to json!\n')

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-r',"--ref",help="Path to input text file containing reference text from your model")
	parser.add_argument('-g',"--gen",help="Path to input text file containing generated text from your model")
	parser.add_argument('-rg',"--rouge",help="If specified, outputs ROUGE scores. Run by specifying '--rouge=True'")
	parser.add_argument('-o',"--out",help="Path to output file")
	args = parser.parse_args()

	reference_corpus = args.ref
	generated_corpus = args.gen
	output_file = args.out
	reftokens = []
	hyptokens = []
	outmetrics = dict.fromkeys(['BLEU-1','BLEU-2','BLEU-3','BLEU-4','BLEU-10','TER','SARI'])
	tokenise(reference_corpus,generated_corpus,reftokens,hyptokens)
	bleu(reftokens,hyptokens,outmetrics)
	if args.rouge:
		rouge(reference_corpus,generated_corpus)
	tojson(output_file,outmetrics)

if __name__ == '__main__':
	main()