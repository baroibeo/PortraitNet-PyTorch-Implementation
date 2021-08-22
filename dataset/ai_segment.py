import os
import cv2
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from augmentation import Transform

class AISegment(Dataset):
    def __init__(self,img_root,mask_root,train=False):
        self.img_root = img_root
        self.img_files = os.listdir(img_root)
        self.mask_root = mask_root
        self.train = train
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self,idx):
        img_filename = self.img_files[idx]
        img = cv2.imread(os.path.join(self.img_root,img_filename))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_root,img_filename))
        transform = Transform(self.train)
        img,mask = transform(img,mask)
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(mask)
        # plt.show()
        # print(img[0][0])
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask).long()
        return img,mask

def test():
    img_root = "/home/hieunn/datasets/Dataset_For_Portrait_Segmentation/Sub_AISegment/xtrain/"
    mask_root = "/home/hieunn/datasets/Dataset_For_Portrait_Segmentation/Sub_AISegment/ytrain/"
    ds = AISegment(img_root,mask_root,train=False)
    print(ds[0][0].shape,ds[0][1].shape)
    # print(ds[0][1])
    # print(ds[0][0])

if __name__ == '__main__':
    test()