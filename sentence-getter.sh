#!/usr/bin/env bash
# -*- coding: UTF-8 -*-

echo 'complex' 
sed $1'q;d' gencorpora/en_compl_corpus.test 
echo 'simple'
sed $1'q;d' gencorpora/en_simpl_corpus.test 
echo '5 epochs'
sed $1'q;d' gencorpora/simp5.txt
echo '10 epochs'
sed $1'q;d' gencorpora/simp10.txt
echo '15 epochs'
sed $1'q;d' gencorpora/simp15.txt
echo '20 epochs'
sed $1'q;d' gencorpora/simp20.txt
