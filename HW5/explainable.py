import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.segmentation import slic
from lime import lime_image
import seaborn as sns
from sklearn.metrics import confusion_matrix

out_dir = sys.argv[2]
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)
out_dir += '/'
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        #input 維度 [3, 256, 256]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # [32, 256, 256] 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [32, 128, 128]

            nn.Conv2d(32, 64, 3, 1, 1), # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),        # [512, 4, 4]
                        
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

class FoodDataset(Dataset):
    def __init__(self, paths, labels, mode):

        self.paths = paths
        self.labels = labels
        train_transform = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.RandomHorizontalFlip(),   #隨機將圖片水平翻轉
            transforms.RandomVerticalFlip(),     #隨機將圖片垂直翻轉
            transforms.RandomRotation(20),       #隨機旋轉圖片
            transforms.ToTensor(),               #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)
        ])
        evalTransform = transforms.Compose([
            transforms.Resize(size=(256, 256)),                                    
            transforms.ToTensor(),
        ])
        self.transform = trainTransform if mode == 'train' else evalTransform
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        X = Image.open(self.paths[index])
        X = self.transform(X)
        Y = self.labels[index]
        return X, Y

    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
          image, label = self.__getitem__(index)
          images.append(image)
          labels.append(label)
        return torch.stack(images), torch.tensor(labels)

args = {
      'dataset_dir': sys.argv[1] + '/'
}
args = argparse.Namespace(**args)

def get_paths_labels(path):
    imgnames = os.listdir(path)
    imgnames.sort()
    imgpaths = []
    labels = []
    for name in imgnames:
        imgpaths.append(os.path.join(path, name))
        labels.append(int(name.split('_')[0]))
    return imgpaths, labels

model = Classifier().cuda()
model.load_state_dict(torch.load('hw5_best.pkl'))

train_paths, train_labels = get_paths_labels(os.path.join(args.dataset_dir, 'training'))
train_set = FoodDataset(train_paths, train_labels, mode='eval')


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())

def compute_saliency_maps(x, y, model):
    model.eval()
    x = x.cuda()

    # 因為我們要計算 loss 對 input image 的微分，原本 input x 只是一個 tensor，預設不需要 gradient
    # 這邊我們明確的告知 pytorch 這個 input x 需要gradient，這樣我們執行 backward 後 x.grad 才會有微分的值
    x.requires_grad_()
    
    y_pred = model(x)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.cuda())
    loss.backward()

    saliencies = x.grad.abs().detach().cpu()
    # saliencies: (batches, channels, height, weight)
    # 因為接下來我們要對每張圖片畫 saliency map，每張圖片的 gradient scale 很可能有巨大落差
    # 可能第一張圖片的 gradient 在 100 ~ 1000，但第二張圖片的 gradient 在 0.001 ~ 0.0001
    # 如果我們用同樣的色階去畫每一張 saliency 的話，第一張可能就全部都很亮，第二張就全部都很暗，
    # 如此就看不到有意義的結果，我們想看的是「單一張 saliency 內部的大小關係」，
    # 所以這邊我們要對每張 saliency 各自做 normalize。手法有很多種，這邊只採用最簡單的
    saliencies = torch.stack([normalize(item) for item in saliencies])
    return saliencies

img_indices = [83, 3211, 4954, 8598]
images, labels = train_set.getbatch(img_indices)
saliencies = compute_saliency_maps(images, labels, model)

cnt = 1
for i in range(len(images)):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(images[i].permute(1, 2, 0).numpy())
    ax2.imshow(saliencies[i].permute(1, 2, 0).numpy())
    plt.savefig(out_dir + '1_' + str(cnt) + '.png')
    plt.close()
    cnt += 1
    # 在 pytorch 的世界，image tensor 各 dimension 的意義通常為 (channels, height, width)
    # 但在 matplolib 的世界，想要把一個 tensor 畫出來，形狀必須為 (height, width, channels)
    # 這邊 img.permute(1, 2, 0)，代表轉換後的 tensor

print("Finish Part 1！！！")

# Fillter  Explaination  

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

layer_activations = None

def filter_explaination(x, model, cnnid, filterid, iteration=100, lr=1):
  # x: 要用來觀察哪些位置可以 activate 被指定 filter 的圖片們
  # cnnid, filterid: 想要指定第幾層 cnn 中第幾個 filter
  model.eval()

  def hook(model, input, output):
    global layer_activations
    layer_activations = output
  
  hook_handle = model.cnn[cnnid].register_forward_hook(hook)
  # 這一行是在告訴 pytorch，當 forward 「過了」第 cnnid 層 cnn 後，要先呼叫 hook 這個我們定義的 function 後才可以繼續 forward 下一層 cnn
  # 因此上面的 hook function 中，我們就會把該層的 output，也就是 activation map 記錄下來，這樣 forward 完整個 model 後我們就不只有 loss
  # 也有某層 cnn 的 activation map
  # 注意：到這行為止，都還沒有發生任何 forward。我們只是先告訴 pytorch 等下真的要 forward 時該多做什麼事
  # 注意：hook_handle 可以先跳過不用懂，等下看到後面就有說明了

  # Filter activation: 我們先觀察 x 經過被指定 filter 的 activation map
  model(x.cuda())
  # 這行才是正式執行 forward，因為我們只在意 activation map，所以這邊不需要把 loss 存起來
  filter_activations = layer_activations[:, filterid, :, :].detach().cpu()
  
  # 根據 function argument 指定的 filterid 把特定 filter 的 activation map 取出來
  # 因為目前這個 activation map 我們只是要把他畫出來，所以可以直接 detach from graph 並存成 cpu tensor
  
  # Filter visualization: 接著我們要找出可以最大程度 activate 該 filter 的圖片
  x = x.cuda()
  # 從一張 random noise 的圖片開始找 (也可以從一張 dataset image 開始找)
  x.requires_grad_()
  # 我們要對 input image 算偏微分
  optimizer = Adam([x], lr=lr)
  # 利用偏微分和 optimizer，逐步修改 input image 來讓 filter activation 越來越大
  for iter in range(iteration):
    optimizer.zero_grad()
    model(x)
    
    objective = -layer_activations[:, filterid, :, :].sum()
    # 與上一個作業不同的是，我們並不想知道 image 的微量變化會怎樣影響 final loss
    # 我們想知道的是，image 的微量變化會怎樣影響 activation 的程度
    # 因此 objective 是 filter activation 的加總，然後加負號代表我們想要做 maximization
    
    objective.backward()
    # 計算 filter activation 對 input image 的偏微分
    optimizer.step()
  filter_visualization = x.detach().cpu().squeeze()[0]

  hook_handle.remove()
  # 很重要：一旦對 model register hook，該 hook 就一直存在。如果之後繼續 register 更多 hook
  # 那 model 一次 forward 要做的事情就越來越多，甚至其行為模式會超出你預期 (因為你忘記哪邊有用不到的 hook 了)
  # 因此事情做完了之後，就把這個 hook 拿掉，下次想要再做事時再 register 就好了。

  return filter_activations, filter_visualization

img_indices = [83, 3211, 4954, 8598]
images, labels = train_set.getbatch(img_indices)
filterids = [10, 15]
cnt = 1
for f_id in filterids:
    plt.figure() 
    filter_activations, filter_visualization = filter_explaination(images, model, cnnid=10, filterid=f_id, iteration=100, lr=0.1)

    # filter visualization
    plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
    plt.savefig(out_dir + '2_' + str(cnt) +'.png')

    cnt += 1
    # filter activations
    fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
    for i, img in enumerate(images):
        axs[0][i].imshow(img.permute(1, 2, 0))
    for i, img in enumerate(filter_activations):
        axs[1][i].imshow(normalize(img))
    plt.savefig(out_dir + '2_' + str(cnt) +'.png')
    cnt += 1

plt.figure() 
filter_activations, filter_visualization = filter_explaination(images, model, cnnid=5, filterid=20, iteration=100, lr=0.1)

# filter visualization
plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
plt.savefig(out_dir + '2_' + str(cnt) +'.png')

cnt += 1
# filter activations
fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for i, img in enumerate(images):
    axs[0][i].imshow(img.permute(1, 2, 0))
for i, img in enumerate(filter_activations):
    axs[1][i].imshow(normalize(img))
plt.savefig(out_dir + '2_' + str(cnt) +'.png')

print("Finish part 2")

# LIME  

def predict(input):
    # input: numpy array, (batches, height, width, channels)                                                                                                                                                     
    
    model.eval()                                                                                                                                                             
    input = torch.FloatTensor(input).permute(0, 3, 1, 2)                                                                                                            
    # 需要先將 input 轉成 pytorch tensor，且符合 pytorch 習慣的 dimension 定義
    # 也就是 (batches, channels, height, width)

    output = model(input.cuda())                                                                                                                                             
    return output.detach().cpu().numpy()                                                                                                                              
                                                                                                                                                                             
def segmentation(input):
    # 利用 skimage 提供的 segmentation 將圖片分成 100 塊                                                                                                                                      
    return slic(input, n_segments=100, compactness=1, sigma=1)                                                                                                              
                                                                                                                                                                             

img_indices = [83, 1703, 3211, 4218, 4707, 5466, 6791, 7231, 7511, 8598, 1544]
images, labels = train_set.getbatch(img_indices)                                                                                                                                                                
np.random.seed(16)                                                                                                                                                       
# 讓實驗 reproducible

cnt = 1
for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs[0].imshow(image)                                                                                                                                             
    x = image.astype(np.double)

    explainer = lime_image.LimeImageExplainer()                                                                                                                              
    explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation, top_labels=11)
    # 基本上只要提供給 lime explainer 兩個關鍵的 function，事情就結束了
    # classifier_fn 定義圖片如何經過 model 得到 prediction
    # segmentation_fn 定義如何把圖片做 segmentation
    lime_img, mask = explaination.get_image_and_mask(                                                                                                                         
                                label=label.item(),                                                                                                                           
                                positive_only=False,                                                                                                                         
                                hide_rest=False,                                                                                                                             
                                num_features=11,                                                                                                                              
                                min_weight=0.05                                                                                                                              
                            )
    
    axs[1].imshow(lime_img)
    fig.savefig(out_dir + '3_' + str(cnt) + '.png')
    cnt += 1
plt.close()
print("Finish part 3")

# Deep dream
def dream_reg(x, model, label, iteration=100, lr=1): 
  x = x.cuda()
  origin_x = x.clone()
  x.requires_grad_()
  optimizer = Adam([x], lr=lr)
  loss = nn.CrossEntropyLoss()
  label = label.unsqueeze(0)
  #label = label.repeat(11,1)
  #label = label.squeeze()
  for iter in range(iteration):
    optimizer.zero_grad()
    #model.zero_grad()
    train_pred = model(x)
    regularization_loss = torch.sum(torch.abs(x-origin_x))
    img_loss = loss(train_pred, label.cuda()) + 0.001 * regularization_loss
    #img_loss = -train_pred[0].norm()
    img_loss.backward()
    #avg_grad = np.abs(x.grad.data.cpu().numpy()).mean()
    #norm_lr = lr / avg_grad
    #x.data += norm_lr * x.grad.data
    optimizer.step()
  #image_visualization = x.detach().cpu().squeeze()
  image_visualization = (x - origin_x).detach().cpu().squeeze()
  return image_visualization

def dream(x, model, label, iteration=100, lr=1): 
  x = x.cuda()
  origin_x = x.clone()
  x.requires_grad_()
  optimizer = Adam([x], lr=lr)
  loss = nn.CrossEntropyLoss()
  label = label.unsqueeze(0)
  for iter in range(iteration):
    optimizer.zero_grad()
    train_pred = model(x)
    img_loss = loss(train_pred, label.cuda())
    img_loss.backward()
    optimizer.step()
  image_visualization = (x - origin_x).detach().cpu().squeeze()
  return image_visualization

'''
dream_img = dream(images[1].unsqueeze(0), model, labels[1], iteration=100, lr=0.1)
plt.imshow(normalize(dream_img.permute(1, 2, 0)))
dream_img = dream_img.permute(1, 2, 0)
dream_img = Image.fromarray(np.uint8(dream_img*255))
img = images[0].permute(1, 2, 0)
img = Image.fromarray(np.uint8(img*255))
image = Image.blend(img, dream_img, 0.5)
#plt.imshow(image)
plt.close()
'''
for i in range(11):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    dream_img_reg = dream_reg(images[i].unsqueeze(0), model, labels[i], iteration=100, lr=0.1)
    ax2.imshow(normalize(dream_img_reg.permute(1, 2, 0).numpy()))
    plt.grid(False)
    dream_img = dream(images[i].unsqueeze(0), model, labels[i], iteration=100, lr=0.1)
    ax1.imshow(normalize(dream_img.permute(1, 2, 0).numpy()))
    plt.grid(False)
    plt.savefig(out_dir + '4_' + str(i) + '.png')
    plt.close()

print("Finish part 4")

# Draw the confusion matrix
C2 = np.load('matrix.npy')
C2 = C2.astype(np.float32)
for i in range(11):
    row = C2[i,:].copy()
    total = sum(row)
    C2[i,:] = C2[i,:] / total
C2 = np.round(C2,decimals=2)
sns.set()
f,ax=plt.subplots(figsize = (10, 7))
sns.heatmap(C2,annot=True,ax=ax,vmin=0, vmax=1, cmap = 'Blues')
ax.set_title('confusion matrix') 
ax.set_xlabel('predict') 
ax.set_ylabel('true') 
f.savefig(out_dir + 'matrix.png')
plt.close()
print("Finish confusion matrix")