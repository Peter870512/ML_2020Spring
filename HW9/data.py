import numpy as np

def preprocess(image_list):
    """ Normalize Image and Permute (N,H,W,C) to (N,C,H,W)
    Args:
      image_list: List of images (8500, 32, 32, 3)
    Returns:
      image_list: List of images (8500, 3, 32, 32)
    """
    def list_normalize(image_list):
        list_size = image_list.shape[0]
        new_image_list = np.zeros((list_size, 3, 32, 32))
        for i in range(3):
            list_channel = image_list[:, i, :, :].reshape((list_size, -1))
            mean = np.mean(list_channel, axis=1)
            mean = mean[:,np.newaxis]
            mean = np.repeat(mean, 32*32, axis=1)
            mean = mean.reshape(list_size, 32, 32)

            std = np.std(list_channel, axis=1)
            std = std[:,np.newaxis]
            std = np.repeat(std, 32*32, axis=1)            
            std = std.reshape(list_size, 32, 32)
            new_image_list[:, i, :, :] = (image_list[:, i, :, :] - mean) / std
        return new_image_list

    
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    new_image_list = list_normalize(image_list)
    
    image_list = (image_list / 255.0) * 2 - 1
    image_list = image_list.astype(np.float32)
    
    new_image_list = new_image_list.astype(np.float32)
    return image_list
    #return new_image_list

from torch.utils.data import Dataset

class Image_Dataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        images = self.image_list[idx]
        return images

