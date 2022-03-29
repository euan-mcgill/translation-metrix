# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:46:31 2021

@author: SNT
"""
import pandas as pd

import sys
import numpy as np
sys.path.insert(0, "./utils/metrics/")
sys.path.insert(1, './utils')
from nmt_eval import compute_metrics

REF_FILENAME = '/home/upf/Documents/resources/corpora/UPM-LSE/BD/EXPERIMENTS/signos/signos_test_1.txt'
WORKING_DIR = '/home/upf/Documents/resources/corpora/UPM-LSE/BD/EXPERIMENTS/'
EVAL_EPOCHS = [200,400,600,800,1000,1200,1400,1600,1800,2000,2200,2400,2600,
               2800,3000,3200,3600,3800,4000]
label = REF_FILENAME.split('/')[-1].split('.')[0]
#%%############################################################################
'''                         ACCOMODATING ES-EN DATA                         '''
###############################################################################

with open(REF_FILENAME, 'r', encoding = 'utf-8') as f:
    readdata = f.read().split('\n')
    ref_data = [l.upper() for l in readdata]
    

#%%############################################################################
'''                            COMPUTING METRICS                            '''
###############################################################################    
compute_scores = []

for it in EVAL_EPOCHS:
# edit next line to change directory and / or 
    with open(WORKING_DIR+'/generated_text/'+'{}_{}.txt'.format(it, label), 'r', encoding = 'utf-8') as f:
        hyp_text = f.read()
        if "কে"  in hyp_text:
            hyp_text = hyp_text.replace("কে", ' ' ).strip()
        hyp_text = [s.strip() for s in hyp_text.split('\n')]

    compute_scores.append(compute_metrics(ref_data, hyp_text))

#%%


compute_scores_dict =  dict(zip(compute_scores[0].keys(), [[] for k in compute_scores[0].keys()]))
compute_scores_dict['Epochs'] = EVAL_EPOCHS

for s in compute_scores:
    for k in s.keys():
        compute_scores_dict[k].append(s[k])

results = pd.DataFrame.from_dict(compute_scores_dict)
results.to_excel(WORKING_DIR+'/metrics_'+label+'_'+WORKING_DIR+'.xlsx', index = False)
