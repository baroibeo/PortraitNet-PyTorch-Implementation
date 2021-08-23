import os
import cv2
import numpy as np
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
        print(img_filename)
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
        mask_boundary = cv2.Canny(np.uint8(mask),0.3,0.5)
        mask_boundary = mask_boundary/255.0
        mask_boundary[mask_boundary>1] = 1 
        mask_boundary[mask_boundary<0] = 0
        mask = torch.from_numpy(mask).long()
        mask_boundary = torch.from_numpy(mask_boundary).long()
        return img,mask,mask_boundary
    
    @staticmethod
    def visualize(img,mask,mask_boundary):
        if torch.is_tensor(img):
            img = img.detach().numpy()
        
        if torch.is_tensor(mask):
            mask = mask.detach().numpy()
        
        if torch.is_tensor(mask_boundary):
            mask_boundary = mask_boundary.detach().numpy()
        
        fig,axarr = plt.subplots(1,3)
        fig.tight_layout(pad=3.0)
        axarr[0].imshow(img)
        axarr[1].imshow(mask)
        axarr[2].imshow(mask_boundary)
        
        axarr[1].title.set_text("Mask with shape : {}".format(mask.shape))
        axarr[1].title.set_size(8)

        axarr[2].title.set_text("Mask boundary with shape : {}".format(mask_boundary.shape))
        axarr[2].title.set_size(8)
        plt.show()

def test():
    img_root = "/home/hieunn/datasets/Dataset_For_Portrait_Segmentation/Sub_AISegment/xtrain/"
    mask_root = "/home/hieunn/datasets/Dataset_For_Portrait_Segmentation/Sub_AISegment/ytrain/"
    ds = AISegment(img_root,mask_root,train=True)
    # print(ds[0][1])
    # print(ds[0][0])
    img,mask,mask_boundary = ds[200]
    print(mask.shape)
    print(mask_boundary.shape)
    print(torch.max(mask))
    print(torch.max(mask_boundary))
    print(img)
    AISegment.visualize(img,mask,mask_boundary)

if __name__ == '__main__':
    test()