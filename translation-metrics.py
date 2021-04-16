#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
#import numpy
from rouge import FilesRouge
# from SARI import *


def tokenise(reference_corpus,generated_corpus,reftokens,hyptokens):
	'''
	reference_corpus = filepath to reference corpus
	generated corpus = filepath to generated corpus (either simplified or translated)
	reftokens = pre-processing tokens from the reference transcription to be passed to BLEU
	hyptokens = pre-processing tokens from the hypothesis transcription to be passed to BLEU
	'''
	with open(reference_corpus,'r',encoding='latin-1') as refs, open(generated_corpus,'r',encoding='latin-1') as hyps:
		for line in refs:
			reftokens.append(nltk.word_tokenize(line))
		for line in hyps:
			hyptokens.append(nltk.word_tokenize(line))
	return (reftokens,hyptokens)

def bleu(reftokens,hyptokens,outmetrics): # re-evaluate with sentence BLEU (iterate for each sentence - don use append function!)
	'''
	reftokens = pre-processing tokens from the reference transcription to be passed to BLEU
	hyptokens = pre-processing tokens from the hypothesis transcription to be passed to BLEU
	outmetrics = dictionary to write scores to
	'''
	cc = SmoothingFunction()
	outmetrics['BLEU-1'] = corpus_bleu(reftokens,hyptokens,weights=(1,0,0,0))
	outmetrics['BLEU-2'] = corpus_bleu(reftokens,hyptokens,weights=(0.5,0.5,0,0),smoothing_function=cc.method3)
	outmetrics['BLEU-3'] = corpus_bleu(reftokens,hyptokens,weights=(0.33,0.33,0.33,0),smoothing_function=cc.method3)
	outmetrics['BLEU-4'] = corpus_bleu(reftokens,hyptokens,weights=(0.25,0.25,0.25,0.25),smoothing_function=cc.method3)
	print('\n\n\n',outmetrics,'\n\n\n') # this dict is easily encoded to json
	return outmetrics

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
	print('Written to json!\n')

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('-r',"--ref",help="Path to input text file containing reference text from your model")
	parser.add_argument('-g',"--gen",help="Path to input text file containing generated text from your model")
	parser.add_argument('-rg',"--rouge",help="If specified, outputs ROUGE scores for translation. Run by specifying '--rouge=True'")
	parser.add_argument('-o',"--out",help="Path to output file")
	args = parser.parse_args()

	reference_corpus = args.ref
	generated_corpus = args.gen
	output_file = args.out
	reftokens = []
	hyptokens = []
	outmetrics = dict.fromkeys(['BLEU-1','BLEU-2','BLEU-3','BLEU-4','TER','SARI'])
	tokenise(reference_corpus,generated_corpus,reftokens,hyptokens)
	bleu(reftokens,hyptokens,outmetrics)
	if args.rouge:
		rouge(reference_corpus,generated_corpus)
	tojson(output_file,outmetrics)

if __name__ == '__main__':
	main()