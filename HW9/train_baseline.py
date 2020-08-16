import torch
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import sys

from util import same_seeds
from model import baseline_AE
from data import Image_Dataset
from data import preprocess

#trainX = np.load('trainX.npy')
trainX = np.load(sys.argv[1])
trainX_preprocessed = preprocess(trainX)
img_dataset = Image_Dataset(trainX_preprocessed)

same_seeds(0)

model = baseline_AE().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

model.train()
n_epoch = 100

# 準備 dataloader, model, loss criterion 和 optimizer
img_dataloader = DataLoader(img_dataset, batch_size=60, shuffle=True) # batch_size=64


# 主要的訓練過程
for epoch in range(n_epoch):
    epoch_loss = 0
    for data in img_dataloader:
        img = data
        img = img.cuda()

        output1, output = model(img)
        loss = criterion(output, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), './checkpoints/checkpoint_{}.pth'.format(epoch+1))        
    print('epoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, epoch_loss))

# 訓練完成後儲存 model
#torch.save(model.state_dict(), './checkpoints/last_checkpoint.pth')
torch.save(model.state_dict(), sys.argv[2])