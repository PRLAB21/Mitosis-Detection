# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 00:09:52 2020

@author: Aalia Sohail Khan
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# nn.Conv2d(22, 22, kernel_size=(3, 1),  stride=(1, 1),  padding=(1, 0), bias=False)
def conv_bn_block(in_f, out_f, k_size=(3,3), s_size=2, dilation_rate=1,*args, **kwargs): # send k_size in tuple (3,1) or (3,3)
    return  nn.Sequential(
             nn.Conv2d(in_f, out_f, kernel_size=k_size, stride=s_size, padding=((k_size[0]-1)//2, (k_size[1]-1)//2), dilation=dilation_rate),
             nn.BatchNorm2d(out_f)
         )


def pooling(pool_type="Avg", k_size=3, s_size=2, *args, **kwargs):
    if pool_type == "Avg":
        return nn.AvgPool2d(kernel_size=k_size, stride=s_size, padding=((k_size-1)//2,(k_size-1)//2))
    if pool_type == "Max":
        return nn.MaxPool2d(kernel_size=k_size, stride=s_size, padding=((k_size-1)//2,(k_size-1)//2))


class conv_block(nn.Module):
    def __init__(self,block_idx, k_size=(3,3), s_size=1, dilation_rate=1, activation='lrelu',*args, **kwargs):
        super(conv_block,self).__init__()
        activation_fun = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()]
                ])
        self.conv =nn.Sequential(*[conv_bn_block(block_idx, block_idx,k_size,s_size, dilation_rate,*args, **kwargs) for i in range(3)],nn.LeakyReLU())
        # self.activation = activation_fun[activation]

    def forward(self, x):
        return self.conv(x)


class asymetric_block(nn.Module): 
    def __init__(self,block_idx, k_size=(1,3), s_size=1, dilation_rate=1, activation='lrelu',*args, **kwargs):
        super(asymetric_block,self).__init__()
        self.conv1x1  = nn.Conv2d(block_idx, block_idx//2, kernel_size=(1,1), stride=1, padding=(0,0), dilation=1)
        self.conv_asym  = nn.Sequential(
        nn.Conv2d(block_idx//2, block_idx//2, kernel_size=k_size, stride=(1,1), padding=((k_size[0]-1)//2, (k_size[1]-1)//2), dilation=dilation_rate),
        nn.BatchNorm2d(block_idx//2),
        nn.Conv2d(block_idx//2, block_idx//2, kernel_size=k_size, stride=(1,1), padding=((k_size[0]-1)//2, (k_size[1]-1)//2), dilation=dilation_rate),
        nn.LeakyReLU(),
        nn.BatchNorm2d(block_idx//2),
        nn.Conv2d(block_idx//2, block_idx//2, kernel_size=k_size, stride=(1,1), padding=((k_size[0]-1)//2, (k_size[1]-1)//2), dilation=dilation_rate),
        nn.BatchNorm2d(block_idx//2),
        nn.Conv2d(block_idx//2, block_idx//2, kernel_size=k_size, stride=(1,1), padding=((k_size[0]-1)//2, (k_size[1]-1)//2), dilation=dilation_rate),
        nn.LeakyReLU(),
        nn.BatchNorm2d(block_idx//2))

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.conv_asym(x)
        return x




class ASTMNet_dense_block(nn.Module):
    def __init__(self):
        super(ASTMNet_dense_block,self).__init__()
        self.decode = nn.Sequential(
            nn.Dropout(p = 0.5),                          
            nn.Linear(256,85), #1500
            nn.Dropout(p = 0.5), 
            nn.BatchNorm1d(85),
            nn.LeakyReLU(),
            nn.Linear(85,2))
            
    def forward(self, x):   
        return  self.decode(x)  


class ASTMNet(nn.Module): 
    def __init__(self, in_f=3, layers=[32, 64,128,256]):
        super(ASTMNet,self).__init__()
        #self.layers = [*layers]  
        self.input = conv_bn_block(in_f,layers[0],(3,3),2,1)
        
       
        self.Block1_a = conv_block(layers[0], k_size=(3,3),s_size=1,dilation_rate=1,activation='lrelu')
        self.Block1_left = asymetric_block(layers[0], k_size=(1,5), s_size=1, dilation_rate=1, activation='lrelu')
        self.Block1_right = asymetric_block(layers[0], k_size=(5,1), s_size=1, dilation_rate=1, activation='lrelu')
        self.Block1_b = nn.Sequential(conv_block(layers[0], (3,3),1,1,'lrelu'), pooling("Avg",3,2))
        self.c1 = conv_bn_block(layers[0],layers[1],k_size=(1,1), s_size=1, dilation_rate=1) 
        
        self.Block2_a = conv_block(layers[1], k_size=(3,3),s_size=1,dilation_rate=1,activation='lrelu')
        self.Block2_left = asymetric_block(layers[1], k_size=(1,3), s_size=1, dilation_rate=1, activation='lrelu')
        self.Block2_right = asymetric_block(layers[1], k_size=(3,1), s_size=1, dilation_rate=1, activation='lrelu')
        self.Block2_b = nn.Sequential(conv_block(layers[1], (3,3),1,1,'lrelu'), pooling("Avg",3,2))
        self.c2 = conv_bn_block(layers[1],layers[2],k_size=(1,1), s_size=1, dilation_rate=1)  
        
        self.Block3_a = conv_block(layers[2], k_size=(3,3),s_size=1,dilation_rate=1,activation='lrelu')
        self.Block3_ab = nn.Sequential(*[conv_block(layers[2], (3,3),1,1,'lrelu') for  i in range(3)])
        self.Block3_b = nn.Sequential(conv_block(layers[2], (3,3),1,1,'lrelu'), pooling("Avg",3,2))
        self.c3 = conv_bn_block(layers[2],layers[3],k_size=(1,1), s_size=1, dilation_rate=1)  
        
        self.B4 = nn.Sequential(*[conv_block(layers[3], (3,3),1,1,'lrelu') for  i in range(3)])
        self.global_avgpool= nn.AdaptiveAvgPool2d((1,1))
        self.classifier = ASTMNet_dense_block()

    def forward(self,x):
        x = F.leaky_relu(self.input(x))
       
        x = self.Block1_a(x)
       
        x_l = self.Block1_left(x)
        
        x_r = self.Block1_right(x)
        x = torch.cat((x_l,x_r), dim=1)
        x = self.Block1_b(x)
        x = self.c1(x)
        
        
        x = self.Block2_a(x)
        x_l = self.Block2_left(x)
        x_r = self.Block2_right(x)
        x = torch.cat((x_l,x_r), dim=1)
        x = self.Block2_b(x)
        x = self.c2(x)
        
        x = self.Block3_a(x)
        x = self.Block3_ab(x)
        x = self.Block3_b(x)
        x = self.c3(x)
        
        x = self.B4(x)
        
        x = self.global_avgpool(x)
        x= x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
