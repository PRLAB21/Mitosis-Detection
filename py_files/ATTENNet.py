
##### cnn model list based ######
import torch 
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
    
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 3 #7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale




class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class Basic_Conv_operation(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, expansion=1, padding=1, dilation=1, groups=1, activation=False, activation_type='lrelu', bn=True, bias=False):
        super(Basic_Conv_operation, self).__init__()
        activation_fun = nn.ModuleDict([
          ['lrelu', nn.LeakyReLU()],
          ['relu', nn.ReLU()]
          ])
        self.expansion = expansion
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_c,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = activation_fun[activation_type] if activation else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Basic_pool_operation(nn.Module):
    def __init__(self,  kernel_size, stride=2, padding=1, pool_type = "Avg"):
        super(Basic_pool_operation, self).__init__()
        if pool_type == "Avg":
            self.pooling = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=(kernel_size//2,kernel_size//2))
        if pool_type == "Max":
            self.pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=(kernel_size//2,kernel_size//2))

    def forward(self, x):
        x = self.pooling(x)
        return x
    

def conv_sub_block(idx, kernel_size=3, stride=1, expansion=1,activation=False):
    sub_layers =[]
    for i in range(3):
        if i!=2:
            sub_layers.append(Basic_Conv_operation(idx, idx, kernel_size=3, stride=1, expansion=1,activation=False))
        if i==2:
            sub_layers.append(Basic_Conv_operation(idx, idx, kernel_size=3, stride=1, expansion=1,activation=True))
    return nn.Sequential(*sub_layers)



class conv_full_block_att2(nn.Module):
    def __init__(self,  channels, kernel_size=3, stride=1, expansion=1,activation=False):
        super(conv_full_block_att2, self).__init__()
        self.conv_block1=nn.Sequential(*conv_sub_block(channels, kernel_size=3, stride=1))
        self.attent_block1 = CBAM(channels)
        self.conv_block2=nn.Sequential(*conv_sub_block(channels, kernel_size=3, stride=1))
        self.attent_block2 = CBAM(channels)
        self.conv_block3=nn.Sequential(*conv_sub_block(channels, kernel_size=3, stride=1), Basic_pool_operation(kernel_size=3, stride=2))

    def forward (self, x):
        x = self.conv_block1(x)
        x = self.attent_block1(x)
        x = self.conv_block2(x)
        x = self.attent_block2(x)
        x = self.conv_block3(x)
        return x 


class conv_full_block_att_nopool(nn.Module):
    def __init__(self,  channels, kernel_size=3, stride=1, expansion=1,activation=False):
        super(conv_full_block_att_nopool, self).__init__()
        self.conv_block1=nn.Sequential(*conv_sub_block(channels, kernel_size=3, stride=1))
        self.attent_block1 = CBAM(channels)
        self.conv_block2=nn.Sequential(*conv_sub_block(channels, kernel_size=3, stride=1))
        self.attent_block2 = CBAM(channels)
        self.conv_block3=nn.Sequential(*conv_sub_block(channels, kernel_size=3, stride=1))

    def forward (self, x):
        x = self.conv_block1(x)
        x = self.attent_block1(x)
        x = self.conv_block2(x)
        x = self.attent_block2(x)
        x = self.conv_block3(x)
        return x 

class ATTENNet_dense_block(nn.Module):
    def __init__(self):
        super(ATTENNet_dense_block,self).__init__()
        self.decode = nn.Sequential(
        nn.Dropout(p = 0.5),                          
        nn.Linear(320,85), #128 1500
        nn.Dropout(p = 0.5), 
        nn.BatchNorm1d(85),
        nn.LeakyReLU(),
        nn.Linear(85,2))

    def forward(self, x):   
        return  self.decode(x)


class ATTENNet(nn.Module):  
    def __init__(self, in_c=3, block_channels=[20,40,80,160, 320]):
        super(ATTENNet,self).__init__()
        self.input = Basic_Conv_operation(in_c, block_channels[0], kernel_size=3, stride=2, expansion=1,activation=True)
        self.c0 = Basic_Conv_operation(block_channels[0], block_channels[1], kernel_size=1, stride=1,expansion =1,activation=False)
        self.block1 = conv_full_block_att2(block_channels[1], kernel_size=3, stride=1) 
        self.c1 = Basic_Conv_operation(block_channels[1], block_channels[2], kernel_size=1, stride=1,expansion =1,activation=False)
        self.block2 = conv_full_block_att2(block_channels[2], kernel_size=3, stride=1)
        self.c2 = Basic_Conv_operation(block_channels[2], block_channels[3], kernel_size=1, stride=1,expansion =1,activation=False)
        self.block3 = conv_full_block_att2(block_channels[3], kernel_size=3, stride=1)
        self.c3 = Basic_Conv_operation(block_channels[3], block_channels[4], kernel_size=1, stride=1,expansion =1,activation=False)
        self.block4 = conv_full_block_att_nopool(block_channels[4], kernel_size=3, stride=1)
        self.global_avgpool= nn.AdaptiveAvgPool2d((1,1))
        self.classifier = ATTENNet_dense_block()
    
    def forward (self, x):
        x =  self.input(x)
        # print ("input", x)
        x =  self.c0(x)
        # print ("c0", x)
        x = self.block1(x)
        # print ("block1", x)
        x = self.c1(x)
        # print ("c1", x)
        x = self.block2(x)
        # print ("block2", x)
        x = self.c2(x)
        # print ("c2", x)
        x = self.block3(x)
        x = self.c3(x)
        x = self.block4(x)
        # print ("block3", x)
        x = self.global_avgpool(x)
        # print ("block3", x)
        x= x.view(x.size(0),-1)
        x = self.classifier(x)
        return x
