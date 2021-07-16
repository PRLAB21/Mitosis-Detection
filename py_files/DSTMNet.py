
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    


class block_A(nn.Module):
    def __init__(self,block_idx, k_size=(3,3), s_size=1, padding= (1,1), dilation_rate=1, activation='lrelu'):
        super(block_A,self).__init__()
        in_f = block_idx
        out_f = block_idx
        activation_fun = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()]
                ])        
        self.conv_path_A_1 = nn.Sequential(
        nn.Conv2d(in_f, out_f, kernel_size=k_size, stride=(1,1), padding=(1,1), dilation=1),
        nn.BatchNorm2d(out_f),
        nn.Conv2d(out_f, out_f, kernel_size=k_size, stride=(1,1), padding=(1,1), dilation=1),
        nn.BatchNorm2d(out_f),
        nn.LeakyReLU()
        )
        
        self.conv_path_A_2 = nn.Sequential(
        nn.Conv2d(out_f, out_f, kernel_size=k_size, stride=(1,1), padding=(1,1), dilation=1),
        nn.BatchNorm2d(out_f),    
        nn.Conv2d(out_f, out_f, kernel_size=k_size, stride=(1,1), padding=(1,1), dilation=1),
        nn.BatchNorm2d(out_f),
        nn.LeakyReLU()
        )
        
        self.conv_path_A_3 = nn.Sequential(
        nn.Conv2d(out_f, out_f, kernel_size=k_size, stride=(1,1), padding=(1,1), dilation=1),
        nn.BatchNorm2d(out_f),
        nn.Conv2d(out_f, out_f, kernel_size=k_size, stride=(1,1), padding=(1,1), dilation=1),
        nn.BatchNorm2d(out_f),
        nn.LeakyReLU()
        )
        
    def forward(self, x):
        x= self.conv_path_A_1(x)
        x= self.conv_path_A_2(x)
        x= self.conv_path_A_3(x)
        return x

class block_B(nn.Module):
    def __init__(self,block_idx, k_size=(3,3), s_size=1, dilation_rate=1, activation='lrelu'):
        super(block_B,self).__init__()
        in_f = block_idx
        out_f = block_idx
        activation_fun = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()]
                ])
        
        self.conv_path_B_1 = nn.Sequential(
        nn.Conv2d(in_f, out_f, kernel_size=k_size, stride=(1,1), padding=(1,1), dilation=1),
        nn.BatchNorm2d(out_f),
        nn.Conv2d(out_f, out_f, kernel_size=k_size, stride=(1,1), padding=(1,1), dilation=1),
        nn.BatchNorm2d(out_f),
        activation_fun[activation]
        )
        self.conv_path_B_2 = nn.Sequential(
        nn.Conv2d(out_f, out_f, kernel_size=k_size, stride=(1,1), padding=(2,2), dilation=2),
        nn.BatchNorm2d(out_f),
        nn.Conv2d(out_f, out_f, kernel_size=k_size, stride=(1,1), padding=(1,1), dilation=1),
        nn.BatchNorm2d(out_f),
        activation_fun[activation]
        )
        self.conv_path_B_3 = nn.Sequential(        
        nn.Conv2d(out_f, out_f, kernel_size=k_size, stride=(1,1), padding=(4,4), dilation=4),
        nn.BatchNorm2d(out_f),
        nn.Conv2d(out_f, out_f, kernel_size=k_size, stride=(1,1), padding=(1,1), dilation=1),
        nn.BatchNorm2d(out_f),
        activation_fun[activation]
        )
        
    def forward(self, x):
        x = self.conv_path_B_1(x)
        x = self.conv_path_B_2(x)
        x = self.conv_path_B_3(x)
        return x

class sub_block(nn.Module):
    def __init__(self,block_idx1,block_idx2, k_size=(3,3), s_size=1, dilation_rate=1, activation='lrelu'):
        super(sub_block,self).__init__()
        activation_fun = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()]
                ])
        self.conv =nn.Sequential(
            conv_bn_block(block_idx1, block_idx2,k_size,s_size, dilation_rate) ,
            conv_bn_block(block_idx1, block_idx2,k_size,s_size, dilation_rate) ,
            conv_bn_block(block_idx1, block_idx2,k_size,s_size, dilation_rate),
            activation_fun[activation]
            )
    def forward(self, x):
        return self.conv(x)
    
class DSTMNet(nn.Module): 
    def __init__(self, in_f=3, layers=[20, 40, 80, 160, 320]):
        super(DSTMNet,self).__init__()
        self.input = conv_bn_block(in_f,layers[0],(3,3),2,1) 
        self.c0 = conv_bn_block(layers[0],layers[1], k_size=(1,1), s_size=1, dilation_rate=1)  
        
        self.Block1_1 = sub_block(layers[1], layers[1], k_size=(3,3), s_size=1,  dilation_rate=1, activation='lrelu') 
        self.Block1_A = block_A(layers[1], k_size=(3,3))
        self.Block1_B = block_B(layers[1], k_size=(3,3))
        self.Block1_2 = sub_block(layers[1], layers[1], k_size=(3,3), s_size=1,  dilation_rate=1, activation='lrelu')  
        self.Block1_3 = pooling("Avg",3,2)
        self.c1 = conv_bn_block(layers[1],layers[2], k_size=(1,1), s_size=1, dilation_rate=1)  
        
        self.Block2_1 = sub_block(layers[2], layers[2], k_size=(3,3), s_size=1,  dilation_rate=1, activation='lrelu') 
        self.Block2_A = block_A(layers[2], k_size=(3,3))
        self.Block2_B = block_B(layers[2], k_size=(3,3))
        self.Block2_2 = sub_block(layers[2], layers[2], k_size=(3,3), s_size=1,  dilation_rate=1, activation='lrelu')  
        self.Block2_3 = pooling("Avg",3,2)
        self.c2 = conv_bn_block(layers[2],layers[3], k_size=(1,1), s_size=1, dilation_rate=1)  
        
        self.Block3_1 = sub_block(layers[3], layers[3], k_size=(3,3), s_size=1,  dilation_rate=1, activation='lrelu') 
        self.Block3_A = block_A(layers[3], k_size=(3,3))
        self.Block3_B = block_B(layers[3], k_size=(3,3))
        self.Block3_2 = sub_block(layers[3], layers[3], k_size=(3,3), s_size=1,  dilation_rate=1, activation='lrelu')  
        self.Block3_3 = pooling("Avg",3,2)
        self.c3 = conv_bn_block(layers[3],layers[4], k_size=(1,1), s_size=1, dilation_rate=1)  
    
        self.B4_1 = sub_block(layers[4], layers[4], k_size=(3,3), s_size=1,  dilation_rate=1, activation='lrelu')
        self.B4_2 = sub_block(layers[4], layers[4], k_size=(3,3), s_size=1,  dilation_rate=1, activation='lrelu')
        self.B4_3 = sub_block(layers[4], layers[4], k_size=(3,3), s_size=1,  dilation_rate=1, activation='lrelu')
        
        self.global_avgpool= nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
                        nn.Dropout(p = 0.5),                          
                        nn.Linear(320,85), 
                        nn.Dropout(p = 0.5), 
                        nn.BatchNorm1d(85),
                        nn.LeakyReLU(),
                        nn.Linear(85,2))

    def forward(self,x):
        x = F.leaky_relu(self.input(x))
        x = self.c0(x)  
        
        x = self.Block1_1(x)  
        x_A= self.Block1_A(x) 
        x_B= self.Block1_B(x) 
        x = x_A+x_B
        x= self.Block1_2 (x)  
        x = self.Block1_3(x) 
        x = self.c1(x)  
        
        x = self.Block2_1(x) 
        x_A = self.Block2_A(x) 
        x_B = self.Block2_B(x)
        x = x_A+x_B
        x = self.Block2_2(x) 
        x = self.Block2_3(x) 
        x = self.c2(x) 
        
        x = self.Block3_1(x) 
        x_A = self.Block3_A(x) 
        x_B = self.Block3_B(x) 
        x = x_A+x_B
        x = self.Block3_2(x) 
        x = self.Block3_3(x) 
        x = self.c3(x) 

        x = self.B4_1(x) 
        x = self.B4_2(x)
        x = self.B4_3(x)
        
        x = self.global_avgpool(x) 
        x= x.view(x.size(0),-1)
        x = self.classifier(x) 
        return x
