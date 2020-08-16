import numpy as np
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from model import fcn_autoencoder, conv_autoencoder, VAE, loss_vae

task = 'ae'
if task == 'ae':
    train = np.load(sys.argv[1], allow_pickle=True)
    x = train
    num_epochs = 200
    batch_size = 128
    learning_rate = 1e-4

    #{'fcn', 'cnn', 'vae'} 
    if 'baseline' in sys.argv[2]:
        model_type = 'cnn'
    elif 'best' in sys.argv[2]:
        model_type = 'vae'

    x = train
    if model_type == 'fcn' :
        x = x.reshape(len(x), -1)
        
    data = torch.tensor(x, dtype=torch.float)
    train_dataset = TensorDataset(data)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

    model_classes = {'fcn':fcn_autoencoder(), 'cnn':conv_autoencoder(), 'vae':VAE()}
    model = model_classes[model_type].cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    best_loss = np.inf
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in train_dataloader:
            if model_type == 'cnn' or model_type == 'vae':
                img = data[0].transpose(3, 1).cuda()
            else:
                img = data[0].cuda()
            # ===================forward=====================
            output = model(img)
            if model_type == 'vae':
                loss = loss_vae(output[0], img, output[1], output[2], criterion)
            else:
                loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # ===================save====================
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model, 'best_model_{}.pth'.format(model_type))
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, total_loss))

