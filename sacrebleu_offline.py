#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:26:38 2022

@author: upf
"""

# import sacrebleu
from sacrebleu.metrics import BLEU, CHRF, TER # ensure v2.0.0

silver = '/home/upf/Documents/resources/corpora/UPM-LSE/BD/EXPERIMENTS/no_embdg/pred_3800_no.txt'
# silver = '/home/upf/Documents/resources/corpora/UPM-LSE/BD/EXPERIMENTS/aug/frases_test_1_augmented.txt'
gold ='/home/upf/Documents/resources/corpora/UPM-LSE/BD/EXPERIMENTS/signos/signos_test_1.txt'

gen = []
refs = [[]]

with open(silver, 'r') as g:
    for line in g:
        gen.append(line.upper())

with open(gold, 'r') as r:
    for line in r:
        refs[0].append(line.upper())

# print(sacrebleu.corpus_bleu(gen,refs))

print('\n')
bleu = BLEU()
print(bleu.corpus_score(gen, refs))
ter = TER()
print(ter.corpus_score(gen, refs))
chrf = CHRF()
print(chrf.corpus_score(gen, refs))
