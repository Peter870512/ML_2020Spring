import torch
import pickle
import numpy as np
import sys

from data import get_dataloader
from model import StudentNet

def decode8(fname):
    params = pickle.load(open(fname, 'rb'))
    custom_dict = {}
    for (name, param) in params.items():
        if type(param) == tuple:
            min_val, max_val, param = param
            param = np.float64(param)
            param = (param / 255 * (max_val - min_val)) + min_val
            param = torch.tensor(param)
        else:
            param = torch.tensor(param)

        custom_dict[name] = param

    return custom_dict

data_path = sys.argv[1]
test_dataloader = get_dataloader(data_path, 'testing', batch_size=32)

net = StudentNet().cuda()

param = decode8('hw7_best.pkl')
net.load_state_dict(param)

net.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        inputs, _ = data
        test_pred = net(inputs.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

#with open("predict.csv", 'w') as f:
with open(sys.argv[2], 'w') as f:
    f.write('Id,label\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))

print("Finish Writing")