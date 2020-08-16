import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import copy
import sys
import time

device = torch.device("cuda")

# 實作一個繼承 torch.utils.data.Dataset 的 Class 來讀取圖片
class Adverdataset(Dataset):
    def __init__(self, root, label, transforms):
        # 圖片所在的資料夾
        self.root = root
        # 由 main function 傳入的 label
        self.label = torch.from_numpy(label).long()
        # 由 Attacker 傳入的 transforms 將輸入的圖片轉換成符合預訓練模型的形式
        self.transforms = transforms
        # 圖片檔案名稱的 list
        self.fnames = []

        for i in range(200):
            self.fnames.append("{:03d}".format(i))

    def __getitem__(self, idx):
        # 利用路徑讀取圖片
        img = Image.open(os.path.join(self.root, self.fnames[idx] + '.png'))
        # 將輸入的圖片轉換成符合預訓練模型的形式
        img = self.transforms(img)
        # 圖片相對應的 label
        label = self.label[idx]
        return img, label
    
    def __len__(self):
        # 由於已知這次的資料總共有 200 張圖片 所以回傳 200
        return 200

class Best_Attacker:
    def __init__(self, img_dir, label):
        self.model = models.densenet121(pretrained = True)
        self.model.cuda()
        self.model.eval()
        self.iters = 10
        self.alpha = 2/255
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # 把圖片 normalize 到 0~1 之間 mean 0 variance 1
        self.normalize = transforms.Normalize(self.mean, self.std, inplace=False)
        transform = transforms.Compose([                
                        transforms.Resize((224, 224), interpolation=3),
                        transforms.ToTensor(),
                        self.normalize
                    ])
        # 利用 Adverdataset 這個 class 讀取資料
        self.dataset = Adverdataset(img_dir, label, transform)
        
        self.loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size = 1,
                shuffle = False)

    def L_infinity(self, origin, new, epsilon):
        difference = origin - new
        max_diff = torch.max(torch.abs(difference))
        return max_diff
    
    def attack(self, epsilon):
        # 存下一些成功攻擊後的圖片 以便之後顯示
        adv_examples = []
        wrong, fail, success = 0, 0, 0
        img = []
        count = 1
        for (data, target) in self.loader:
            data, target = data.to(device), target.to(device)
            data_raw = copy.deepcopy(data)
            # 將圖片丟入 model 進行測試 得出相對應的 class
            output = self.model(data)
            init_pred = output.max(1, keepdim=True)[1]

            # 如果 class 錯誤 就不進行攻擊
            if init_pred.item() != target.item():
                wrong += 1
                data_raw = data_raw * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                data_raw = data_raw.squeeze().detach().cpu().numpy()
                img.append(data_raw)
                continue
            
            # 如果 class 正確 就開始計算 gradient 進行攻擊
            for i in range(self.iters):
                data.requires_grad = True
                output = self.model(data)
                loss = F.nll_loss(output, target)
                data_grad = torch.autograd.grad(loss, data)[0]
                data = data + self.alpha * data_grad.sign()
                eta = torch.clamp(data - data_raw, min=-epsilon, max=epsilon)
                data = (data_raw + eta).detach()                         
                #data = torch.clamp(data_raw + eta, min=0, max=1).detach()  
            perturbed_data = data

            # 再將加入 noise 的圖片丟入 model 進行測試 得出相對應的 class        
            output = self.model(perturbed_data)
            final_pred = output.max(1, keepdim=True)[1]
            #print('Finish :',count)
            count += 1
            if final_pred.item() == target.item():
                #print("fail")
                # 辨識結果還是正確 攻擊失敗
                adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                adv_ex = adv_ex.squeeze().detach().cpu().numpy()
                img.append(adv_ex)
                fail += 1
            else:
                # 辨識結果失敗 攻擊成功
                success += 1
                #print('success')
                adv_ex = perturbed_data * torch.tensor(self.std, device = device).view(3, 1, 1) + torch.tensor(self.mean, device = device).view(3, 1, 1)
                adv_ex = adv_ex.squeeze().detach().cpu().numpy()
                img.append(adv_ex)

        final_acc = (fail / (wrong + success + fail))
        print("Epsilon: {}\tTest Accuracy = {} / {} = {}\n".format(epsilon, fail, len(self.loader), final_acc))
        return img, final_acc

if __name__ == '__main__':
    t1 = time.time()
    # 讀入圖片相對應的 label
    df = pd.read_csv(sys.argv[1] + "/labels.csv")
    df = df.loc[:, 'TrueLabel'].to_numpy()
    label_name = pd.read_csv(sys.argv[1] + "/categories.csv")
    label_name = label_name.loc[:, 'CategoryName'].to_numpy()
    # new 一個 Attacker class
    attacker = Best_Attacker(sys.argv[1] + "/images", df)
    # 要嘗試的 epsilon
    epsilons = [0.1]

    accuracies, examples = [], []

    # 進行攻擊 並存起正確率和攻擊成功的圖片
    for eps in epsilons:
        #ex, acc = attacker.attack(eps)
        img, acc = attacker.attack(eps)
        accuracies.append(acc)
        #examples.append(ex)
    
    number = 0
    save_path = sys.argv[2]
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path = save_path + '/'
    for ex in img:
        ex = 255 * np.array(ex).astype(np.float32)
        ex = np.swapaxes(ex,0,1)
        ex = np.swapaxes(ex,1,2)
        #print(ex.shape)
        ex[:,:,[0,2]] = ex[:,:,[2,0]] # chage the color channel
        if len(str(number)) == 1:
            name = '00' + str(number)
            cv2.imwrite(save_path+str(name)+'.png', ex)
        elif len(str(number)) == 2:
            name = '0' + str(number)
            cv2.imwrite(save_path+str(name)+'.png', ex)
        else:
            cv2.imwrite(save_path+str(number)+'.png', ex)
        number += 1
    t2 = time.time()
    print("Execute time :" + str(t2-t1))
