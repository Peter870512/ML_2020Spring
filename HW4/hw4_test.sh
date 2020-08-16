#!/bin/bash
wget https://github.com/Peter870512/ML_2020Spring/releases/download/1.0.0/hw4_best_ckpt.model
wget https://github.com/Peter870512/ML_2020Spring/releases/download/1.0.0/w2v_all.model
wget https://github.com/Peter870512/ML_2020Spring/releases/download/1.0.0/w2v_all.model.trainables.syn1neg.npy
wget https://github.com/Peter870512/ML_2020Spring/releases/download/1.0.0/w2v_all.model.wv.vectors.npy
python3 hw4_testing.py $1 $2