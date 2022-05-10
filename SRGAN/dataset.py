#!/usr/bin/env python
# Import Necessary Packages
import os
import random 
import numpy as np
from PIL import Image
from copy import deepcopy
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import cv2


class DIV2KDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        # self.path = ['/content/drive/My Drive/Image Super Resolution/data/Train/LR/',
        #              '/content/drive/My Drive/Image Super Resolution/data/Valid/LR/',
        #              '/content/drive/My Drive/Image Super Resolution/data/Test/LR/',
        #              '/content/drive/My Drive/Image Super Resolution/data/Train/HR/',
        #              '/content/drive/My Drive/Image Super Resolution/data/Valid/HR/',
        #              '/content/drive/My Drive/Image Super Resolution/data/Test/HR/']

        self.path = ['/content/drive/My Drive/Image Super Resolution/Rdata/Train/LR/',
                     '/content/drive/My Drive/Image Super Resolution/Rdata/Valid/LR/',
                     '/content/drive/My Drive/Image Super Resolution/Rdata/Test/LR/',
                     '/content/drive/My Drive/Image Super Resolution/Rdata/Train/HR/',
                     '/content/drive/My Drive/Image Super Resolution/Rdata/Valid/HR/',
                     '/content/drive/My Drive/Image Super Resolution/Rdata/Test/HR/']
        
        mean_hr = [105.57533, 113.14066, 115.96609]
        std_hr = [71.6331, 66.5753, 69.58413]

        mean_lr = [105.58418, 113.14932, 115.97493]
        std_lr = [68.996025, 63.638702, 66.573044]

        self.data = deepcopy(sorted(os.listdir(self.path[self.mode])))
        self.img_transform_lr =  transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean=mean_lr, std=std_lr)])
        self.img_transform_hr =  transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean=mean_hr, std=std_hr)])
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        LR_input = cv2.imread(os.path.join(self.path[self.mode], self.data[index]))
        HR_truth = cv2.imread(os.path.join(self.path[(self.mode)+3], self.data[index]))
        
        #LR_input = Image.open(os.path.join(self.path[self.mode], self.data[index])).convert('RGB')
        #HR_truth = Image.open(os.path.join(self.path[self.mode+3], self.data[index])).convert('RGB')
   
        #LR_input = np.asarray(LR_input)
        #HR_truth = np.asarray(HR_truth)

        LR_input = deepcopy(LR_input.astype(np.float32))
        HR_truth = deepcopy(HR_truth.astype(np.float32))
        
       	LR_input *= 1.0/255.0
        HR_truth *= 1.0/255.0

        #HR_truth = HR_truth.transpose(2, 0, 1).astype(np.float32)
        #LR_input = LR_input.transpose(2, 0, 1).astype(np.float32)
        
        #LR_input = Image.fromarray(LR_input, 'RGB')
        #HR_truth = Image.fromarray(HR_truth, 'RGB')

        LR_tensor = F.to_tensor(LR_input)
        HR_tensor = F.to_tensor(HR_truth)
        
        #LR_tensor = self.img_transform_hr(LR_input)
        #HR_tensor = self.img_transform_hr(HR_truth)

        return LR_tensor, HR_tensor