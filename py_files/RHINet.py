
import torch 
import torch.nn as nn
import torch.nn.functional as F


def conv_bn_block(in_f, out_f, k_size=3, s_size=2, dilation_rate=1):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, kernel_size=k_size, stride=s_size, padding=k_size//2, dilation=dilation_rate),
        nn.BatchNorm2d(out_f)
        )

def pooling(pool_type="Avg", k_size=3, s_size=2):
    if pool_type == "Avg":
        return nn.AvgPool2d(kernel_size=k_size, stride=s_size, padding=(k_size//2,k_size//2))
    if pool_type == "Max":
        return nn.MaxPool2d(kernel_size=k_size, stride=s_size, padding=(k_size//2,k_size//2))



class intermediate_block(nn.Module):
    def __init__(self,block_idx, k_size=3, s_size_1=1, s_size_2=2, dilation_rate=1, activation_type='LeakyReLU'):
        super(intermediate_block,self).__init__()
        activation_fun = nn.ModuleDict([
                ['LeakyReLU', nn.LeakyReLU()],
                ['ReLU', nn.ReLU()]
                ])
        self.conv_1 =nn.Sequential(
            conv_bn_block(block_idx, block_idx, k_size, s_size=s_size_1, dilation_rate=1),
            conv_bn_block(block_idx, block_idx, k_size, s_size=s_size_1, dilation_rate=1),
            conv_bn_block(block_idx, block_idx, k_size, s_size=s_size_1, dilation_rate=1),
            nn.LeakyReLU(),
            pooling(pool_type = "Avg",k_size=2, s_size=1)
            )
        
        self.conv_2 =nn.Sequential(
            conv_bn_block(block_idx, block_idx, k_size, s_size=s_size_1, dilation_rate=1),
            conv_bn_block(block_idx, block_idx, k_size, s_size=s_size_1, dilation_rate=1),
            conv_bn_block(block_idx, block_idx, k_size, s_size=s_size_1, dilation_rate=1),
            nn.LeakyReLU())
        
        self.conv_3 =nn.Sequential(
            conv_bn_block(block_idx, block_idx, k_size, s_size=s_size_1, dilation_rate=1),
            conv_bn_block(block_idx, block_idx, k_size, s_size=s_size_1, dilation_rate=1),
            pooling(pool_type = "Avg",k_size=2, s_size=1),
            conv_bn_block(block_idx, block_idx, k_size, s_size=s_size_2, dilation_rate=1),
            nn.LeakyReLU()
            )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x


class conv_block(nn.Module):
    def __init__(self,block_idx, k_size=3, s_size=1, dilation_rate=1, activation_type='LeakyReLU'):
        super(conv_block,self).__init__()
        activation_fun = nn.ModuleDict([
                ['LeakyReLU', nn.LeakyReLU()],
                ['LeakyReLU', nn.LeakyReLU()]
                ])
        self.conv =nn.Sequential(*[conv_bn_block(block_idx, block_idx,k_size,s_size, dilation_rate) for i in range(3)],nn.LeakyReLU())
        # self.activation = activation_fun[activation]
    
    def forward(self, x):
        return self.conv(x)


class RHINet(nn.Module):
    def __init__(self, in_f=3, layers=[20, 40, 80, 160, 320]): 
        super(RHINet,self).__init__()
        
        self.input = conv_bn_block(in_f, out_f=layers[0], k_size=3,s_size=1,dilation_rate=1) 
        self.pool_0 = pooling(pool_type = "Avg",k_size=3, s_size=1)
        self.layer_0 = conv_bn_block(in_f=layers[0], out_f=layers[0], k_size=3,s_size=2,dilation_rate=1)
        self.c0 = conv_bn_block(in_f=layers[0], out_f=layers[1], k_size=3, s_size=1,dilation_rate=1) 
                   
        self.B1 = intermediate_block(layers[1], k_size=3, s_size_1=1, s_size_2=2, dilation_rate=1, activation_type='LeakyReLU')
        self.c1 = conv_bn_block(layers[1], layers[2], k_size=1, s_size=1, dilation_rate=1) 
        
        self.B2 = intermediate_block(layers[2], k_size=3, s_size_1=1, s_size_2=2, dilation_rate=1, activation_type='LeakyReLU')
        self.c2 = conv_bn_block(layers[2],layers[3], k_size=1, s_size=1, dilation_rate=1)   #15
        
        self.B3 = intermediate_block(layers[3], k_size=3, s_size_1=1, s_size_2=1, dilation_rate=1, activation_type='LeakyReLU')
        self.c3 = conv_bn_block(layers[3],layers[4], k_size=1, s_size=1,dilation_rate=1)
        
        self.B4 = nn.Sequential(conv_block(layers[4], k_size=3, s_size=1, dilation_rate=1, activation_type='LeakyReLU'),
                                conv_block(layers[4], k_size=3, s_size=1, dilation_rate=1, activation_type='LeakyReLU'),
                                conv_block(layers[4], k_size=3, s_size=1, dilation_rate=1, activation_type='LeakyReLU')
                                )        
        self.global_avgpool= nn.AdaptiveMaxPool2d((1,1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),                          
            nn.Linear(320,80), 
            nn.Dropout(p = 0.5), 
            nn.BatchNorm1d(80),
            nn.LeakyReLU(),
            nn.Linear(80,2))
        
    def forward(self,x):
        x = self.input(x)
        x = self.pool_0(x)
        x = self.layer_0(x)
        x = self.c0(x)
                   
        x = self.B1(x)
        x = self.c1(x)
        
        x = self.B2(x)
        x = self.c2(x)
        
        x = self.B3(x)
        x= self.c3(x)
        
        x = self.B4(x)      
        x= self.global_avgpool(x)
        
        x= x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
