import torch
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import AE
from data import preprocess
from data import Image_Dataset


def inference(X, model, batch_size=256):
    X = preprocess(X)
    dataset = Image_Dataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    latents = []
    for i, x in enumerate(dataloader):
        x = torch.FloatTensor(x)
        vec, img = model(x.cuda())
        if i == 0:
            latents = vec.view(img.size()[0], -1).cpu().detach().numpy()
        else:
            latents = np.concatenate((latents, vec.view(img.size()[0], -1).cpu().detach().numpy()), axis = 0)
    print('Latents Shape:', latents.shape)
    return latents

def predict(latents):
    
    # First Dimension Reduction
    transformer = KernelPCA(n_components=100, kernel='rbf', n_jobs=-1)
    kpca = transformer.fit_transform(latents)
    print('First Reduction Shape:', kpca.shape)
    
    # # Second Dimesnion Reduction
    X_embedded = TSNE(n_components=2, random_state=0).fit_transform(kpca)
    #X_embedded = TSNE(n_components=2, random_state=0).fit_transform(latents)
    print('Second Reduction Shape:', X_embedded.shape)

    # Clustering
    pred = MiniBatchKMeans(n_clusters=2, random_state=0).fit(X_embedded)
    pred = [int(i) for i in pred.labels_]
    pred = np.array(pred)
    return pred, X_embedded

def invert(pred):
    return np.abs(1-pred)

def save_prediction(pred, out_csv='prediction.csv'):
    with open(out_csv, 'w') as f:
        f.write('id,label\n')
        for i, p in enumerate(pred):
            f.write(f'{i},{p}\n')
    print(f'Save prediction to {out_csv}.')

# load model
model = AE().cuda()
#model.load_state_dict(torch.load('./checkpoints/last_checkpoint.pth'))
model.load_state_dict(torch.load(sys.argv[2]))
model.eval()

# 準備 data
#trainX = np.load('trainX.npy')
trainX = np.load(sys.argv[1])
print("Finish Load")
# 預測答案
latents = inference(X=trainX, model=model)
pred, X_embedded = predict(latents)
print("Finish predict")
# 將預測結果存檔，上傳 kaggle
save_prediction(pred, sys.argv[3])

# 由於是 unsupervised 的二分類問題，我們只在乎有沒有成功將圖片分成兩群
# 如果上面的檔案上傳 kaggle 後正確率不足 0.5，只要將 label 反過來就行了
#save_prediction(invert(pred), 'prediction_invert.csv')
#save_prediction(invert(pred), sys.argv[3])
