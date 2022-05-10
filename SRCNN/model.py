#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


# In[2]:


class SRCNN(nn.Module):
    ''' SRCN Pytorch Model'''
    
    def __init__(self, c, f1, f2):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(c, f1, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(f2, c, kernel_size=5, stride=1, padding = 2)
        self.init_weights()
        
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)
        
    def forward(self, X):
        out = F.relu(self.conv1(X))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        return out
    

        


# In[3]:


# model = SRCNN()
# model = model.cuda()
# summary(model, input_size=(1, 512, 512))


# In[ ]:





# In[4]:


class ConvBlock(nn.Module):
    ''' Convolution Layer with Pre Batch Normalization & Relu Preactivation'''
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.activation = nn.ReLU(True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding = 1, bias=False)
        
    def forward(self, x):
        x = self.bn(x)
        x = self.activation(x)
        x = self.conv(x)
        return x


# In[5]:


class ResidualFunction(nn.Module):
    ''' Convolution Layer with Relu Preactivation'''
    
    def __init__(self, channels):
        
        super(ResidualFunction, self).__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)
        
    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.conv2(h)
        return h
        


# In[6]:


class RecursiveBlocks(nn.Module):
    
    def __init__(self,in_channels, out_channels,  U):
        super(RecursiveBlocks, self).__init__()
        self.U = U
        self.conv = ConvBlock(in_channels, out_channels)
        self.res_unit = ResidualFunction(out_channels)
        
    def forward(self, x):
        h = self.conv(x)
        out = h
        while(self.U > 0):
            out = self.res_unit(out)
            out = torch.add(out, h)
            self.U -= 1
        
        return out
        


# In[7]:


class DRRN(nn.Module):
    def __init__(self, in_channels, out_channels, U):
        super(DRRN, self).__init__()
        self.rb = RecursiveBlocks(in_channels, out_channels, U)
        self.conv = ConvBlock(out_channels, in_channels)
        
    def forward(self, x):
        out = x
        out = self.rb(x)
        out = self.conv(out)
        out = torch.add(out, x)
        return out
        


# In[8]:


# model = DRRN(1, 2, 1)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# summary(model, input_size=(1, 512, 512))


# In[ ]:




