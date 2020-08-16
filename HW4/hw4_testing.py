import os
import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from util import load_testing_data
from preprocess import Preprocess
from model import LSTM_Net
from data import TwitterDataset
from test import testing
import sys

path_prefix = './'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
w2v_path = os.path.join(path_prefix, 'w2v_all.model')
#testing_data = os.path.join(path_prefix, 'testing_data.txt')
testing_data = sys.argv[1]
model_dir = path_prefix

sen_len = 40
fix_embedding = True 
batch_size = 128

# 開始測試模型並做預測
print("loading testing data ...")
test_x = load_testing_data(testing_data)
preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
model = LSTM_Net(embedding, embedding_dim=300, hidden_dim=500, num_layers=3, sen_len=sen_len, dropout=0.4, fix_embedding=fix_embedding)
model = model.to(device) # device為 "cuda"，model 使用 GPU 來訓練（餵進去的 inputs 也需要是 cuda tensor）
test_x = preprocess.sentence_word2idx()
test_dataset = TwitterDataset(X=test_x, y=None)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8)
print('\nload model ...')
model = torch.load(os.path.join(model_dir, 'hw4_best_ckpt.model'))
outputs = testing(batch_size, test_loader, model, device)

# 寫到 csv 檔案供上傳 Kaggle
tmp = pd.DataFrame({"id":[str(i) for i in range(len(test_x))],"label":outputs})
print("save csv ...")
save_path = sys.argv[2]
#tmp.to_csv(os.path.join(path_prefix, 'predict.csv'), index=False)
tmp.to_csv(sys.argv[2], index=False)
print("Finish Predicting")