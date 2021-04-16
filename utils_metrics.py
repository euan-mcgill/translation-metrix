# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.
This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math


def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.
  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.
  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
  """Computes BLEU score of translated segments against one or more references.
  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.
  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  ratio = float(translation_length) / reference_length

  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return (bleu, precisions, bp, ratio, translation_length, reference_length)



#%%############################################################################
'''                                  TER                                    '''
###############################################################################


from functools import lru_cache
from itertools import product

@lru_cache(maxsize=4095)
def ld(s, t):
    """
    Levenshtein distance memoized implementation from Rosetta code:
    https://rosettacode.org/wiki/Levenshtein_distance#Python
    """
    if not s: return len(t)
    if not t: return len(s)
    if s[0] == t[0]: return ld(s[1:], t[1:])
    l1 = ld(s, t[1:])      # Deletion.
    l2 = ld(s[1:], t)      # Insertion.
    l3 = ld(s[1:], t[1:])  # Substitution.
    return 1 + min(l1, l2, l3)

def find_shifts(hyp, ref):
    """Find possible shifts in hypothesis."""
    hyp_len, ref_len = len(hyp), len(ref)
    for i, j in product(range(hyp_len), range(ref_len)):
        if i == j: # Skip words in the same position.
            continue
        # When word matches.
        if hyp[i] == ref[j]: 
            # Find the longest matching phrase from this position
            l = 0
            for l, (h, r) in enumerate(zip(hyp[i:], ref[j:])):
                if h != r:
                    break
                l += 1
            # Compute the shifted hypothesis.
            shifted_hyp = hyp[:i] + hyp[i+l:]
            shifted_hyp[j:j] = hyp[i:i+l]
            yield shifted_hyp
            
def shift(hyp, ref):
    original = ld(tuple(hyp), tuple(ref))
    # Find the lowest possible shift and it distance.
    scores = []
    for shifted_hyp in find_shifts(hyp, ref):
        shifted_dist = ld(tuple(shifted_hyp), tuple(ref))
        scores.append((original - shifted_dist, shifted_hyp))
    # Return original hypothesis if shift is not better.
    return sorted(scores)[-1] if scores else (0, hyp)
        
def ter(hyp, ref):
    # Initialize no. of edits, e.
    e = 0
    while True:
        # Find shift, s, that most reduces min-edit-distance(h', r)
        delta, s = shift(hyp, ref)
        # until no shifts that reduce edit distance remain
        if delta <= 0:
            break
        # if shift reduces edit distance, then
        # h' <- apply s to h'
        hyp = s
        # e <- e + 1
        e += 1
    # e <- e + min-edit-distance(h', r)
    e += ld(tuple(hyp), tuple(ref))
    return e / len(ref)


from nltk.translate.bleu_score import sentence_bleu

def compute_scores(ref, hyp):
    splitted_ref = [r.split(' ') for r in ref]
    splitted_hyp = [h.split(' ') for h in hyp]
    
    blue_score = list(map(lambda x, y: sentence_bleu([x], y), splitted_ref,
                           splitted_hyp))
    
    ter_score = list(map(lambda x, y: ter(x, y), splitted_ref,
                           splitted_hyp))

    return blue_score, ter_score












