# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 10:07:51 2021

@author: SNT
"""
from nltk.metrics import distance

def wer(decode, target):
    """Computes the Word Error Rate (WER).

    WER is defined as the edit distance between the two provided sentences after
    tokenizing to words.

    Args:
      decode: string of the decoded output.
      target: a string for the ground truth label.

    Returns:
      A float number for the WER of the current decode-target pair.
    """
    # Map each word to a new char.
    words = set(decode.split() + target.split())
    word2char = dict(zip(words, range(len(words))))

    new_decode = [chr(word2char[w]) for w in decode.split()]
    new_target = [chr(word2char[w]) for w in target.split()]

    return distance.edit_distance(''.join(new_decode), ''.join(new_target)) 