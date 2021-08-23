import numpy as np
import cv2
from config import Config
import random

class Transform():
    def __init__(self,train=False):
        self.train = train
    
    def __call__(self,img,mask):
        pass

    def resize(self,img,mask):
        img = cv2.resize(img,(Config.IMG_NEW_WIDTH.value,Config.IMG_NEW_HEIGHT.value),cv2.INTER_LINEAR)
        mask = cv2.resize(mask,(Config.IMG_NEW_WIDTH.value,Config.IMG_NEW_HEIGHT.value),cv2.INTER_LINEAR)
        return img,mask
    
    def normalize(self,img,mask):
        img = np.array(img,np.float32)
        img = (img/255.0-Config.MEAN.value)/Config.STD.value
        if np.max(mask) == 255.0:
            mask = mask/255.0
        return img,mask
    
    def randomBlur(self,img):
        print("RANDOMBLUR")
        if random.random() <= 0.5:
            img = cv2.blur(img,(3,3))
        return img
    
    def randomHSV(self,img):
        hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        h,s,v = cv2.split(hsv)
        k = random.uniform(0.5,1.5)
        if random.random() <= 0.5:
            print("H")
            h = h*k
            h = np.clip(h,0,255).astype(hsv.dtype)
        
        if random.random() <= 0.5:
            print("V")
            v = v*k
            v = np.clip(v,0,255).astype(hsv.dtype)
        
        if random.random() <= 0.5:
            print("S")
            s = s*k
            s = np.clip(s,0,255).astype(hsv.dtype)
        
        hsv = cv2.merge((h,s,v))
        img = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        return img
    
    def randomHorizontalFlip(self,img,mask):
        if random.random() <= 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        
        return img,mask
    
    def randomVerticalFlip(self,img,mask):
        if random.random() <= 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)
        
        return img,mask
    