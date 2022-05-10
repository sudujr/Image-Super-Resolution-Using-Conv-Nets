#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torchvision.models import vgg19


# In[2]:


def PixelL1Loss():
    ''' Pixel L1 Loss / MAE Loss'''
    return nn.L1Loss()


# In[6]:


def PixelL2Loss():
    ''' Pixel L2 Loss / MSE Loss'''
    return nn.MSELoss()


# In[5]:


class ContentLoss(nn.Module):
    """ Perceptual Loss Using VGG 19"""
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.vgg = vgg19(pretrained=True)
        self.model = nn.Sequential(*list(self.vgg.features)[:31])

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.mse = nn.MSELoss()
        
    def forward(self, predicted, groundTruth):
        return self.mse(self.model(predicted), self.model(groundTruth))
        


# In[ ]:




