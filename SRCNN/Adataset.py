#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import Necessary Packages
import os
import random 
import numpy as np
from PIL import Image
from copy import deepcopy
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import cv2


# In[82]:

class DIV2KDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.path = ['/content/drive/My Drive/Image Super Resolution/data/Train/LR/', 
                        '/content/drive/My Drive/Image Super Resolution/data/Valid/LR/', 
                        '/content/drive/My Drive/Image Super Resolution/data/Test/LR/',
                        '/content/drive/My Drive/Image Super Resolution/data/Train/HR/', 
                        '/content/drive/My Drive/Image Super Resolution/data/Valid/HR/', 
                        '/content/drive/My Drive/Image Super Resolution/data/Test/HR/']
        
        self.data = deepcopy(sorted(os.listdir(self.path[self.mode])))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        LR_Input = cv2.imread(os.path.join(self.path[self.mode], self.data[index]))
        LR_Input = cv2.cvtColor(LR_Input, cv2.COLOR_BGR2YCrCb)
        HR_truth = cv2.imread(os.path.join(self.path[(self.mode)+3], self.data[index]))
        HRYCrCb = cv2.cvtColor(HR_truth, cv2.COLOR_BGR2YCrCb)
        HRYCrCb = HRYCrCb.astype(np.float32)
        width = int(HR_truth.shape[1])
        height = int(HR_truth.shape[0])
        dim = (width, height)
        LR_Input = cv2.resize(LR_Input, dim , interpolation = cv2.INTER_CUBIC)

        LR_Input = LR_Input.astype(np.float32)
        HR_truth = HR_truth.astype(np.float32)
        # HRYCrCb = HRYCrCb.astype(np.float32)

        LRY = LR_Input[:, :, 0]
        HRY = HRYCrCb[:, :, 0]

        LRY *= 1.0/255.0
        HRY *= 1.0/255.0

        LRY = LRY.astype(np.float32)
        HRY = HRY.astype(np.float32)

        LRY = F.to_tensor(LRY)
        HRY = F.to_tensor(HRY)
        LR_Input = F.to_tensor(LR_Input)
        HR_truth = F.to_tensor(HR_truth)
        
        return LRY, HRY, LR_Input, HR_truth
