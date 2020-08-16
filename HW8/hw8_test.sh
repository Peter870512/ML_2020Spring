#!/bin/bash
mkdir ckpt
cat model_* > model_12000.ckpt
mv model_12000.ckpt ./ckpt
python3 main_test.py $1 $2