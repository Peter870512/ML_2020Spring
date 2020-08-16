import numpy as np
import sys
from model import conv_autoencoder, VAE
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

# sys.argv[1]: test.npy
# sys.argv[2]: AE model
# sys.argv[3]: prediction.csv

batch_size = 128
test = np.load(sys.argv[1], allow_pickle=True)
task = 'ae'
if 'baseline' in sys.argv[2]:
    model_type = 'cnn'
elif 'best' in sys.argv[2]:
    model_type = 'vae'
if task == 'ae':
    if model_type == 'fcn' :
        y = test.reshape(len(test), -1)
    else:
        y = test
        
    data = torch.tensor(y, dtype=torch.float)
    test_dataset = TensorDataset(data)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

    model = torch.load(sys.argv[2], map_location='cuda')

    model.eval()
    reconstructed = list()
    
    for i, data in enumerate(test_dataloader): 
        if model_type == 'cnn' or model_type == 'vae':
            img = data[0].transpose(3, 1).cuda()
        else:
            img = data[0].cuda()
        output = model(img)
        if model_type == 'cnn':
            output = output.transpose(3, 1)
        elif model_type == 'vae':
            output = output[0].transpose(3, 1)
        reconstructed.append(output.cpu().detach().numpy())

    reconstructed = np.concatenate(reconstructed, axis=0)
    anomality = np.sqrt(np.sum(np.square(reconstructed - y).reshape(y.shape[0], -1), axis=1))
    y_pred = anomality
    
    output_path = sys.argv[3]
    with open(output_path, 'w') as f:
        f.write('id,anomaly\n')
        for i in range(len(y_pred)):
            f.write('{},{}\n'.format(i+1, y_pred[i]))
    print('Finish Writing!')

