import torch.nn as nn
import torch
from torchvision import models
import PIL.Image as Image
from torch.nn import functional as F
import matplotlib.pyplot as plt
from utils import save_net,load_net

model_path = 'vgg16_bn-6c64b313.pth'

class MFSnet(nn.Module):
    def __init__(self, load_weights=False):
        super(MFSnet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 'CB64', 64]
        self.frontend_feat1 = ['M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,'M']
        self.frontend_feat2 = ['A']
        self.frontend_feat3 = [64]
        self.frontend_feat4 = ['CA']
        self.backend_feat  = [512, 512, 256,128,64]

        self.frontend = make_layers(self.frontend_feat)
        self.frontend1 = make_layers(self.frontend_feat1,in_channels = 64)
        self.frontend2 = make_layers(self.frontend_feat2,in_channels = 64)
        self.frontend3 = make_layers(self.frontend_feat3,in_channels = 512)
        self.frontend4 = make_layers2(self.frontend_feat4)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            mod = models.vgg16_bn(pretrained = False)
            mod.load_state_dict(torch.load(model_path))
            self._initialize_weights()
            #for i in range(len(self.frontend.state_dict().items())):
                #print(i)
                #z=(6,7,8,9,10,13,14,15,16)
                #if i not in z:
                   # list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]


    def forward(self,x):    
        a = self.frontend(x)
        a = self.frontend2(a)
        b = self.frontend2(a)
        a = self.frontend1(a)
        b = self.frontend1(b)
        x = self.frontend(x)
        x0 = x
        x = self.frontend1(x)
        b = F.upsample_bilinear(b, size=[26,26])
        b1 = F.softmax(b,dim=2)
        b2 = F.softmax(b,dim=3)
        b3 = torch.add(b1,b2)
        a1 = F.softmax(a,dim=2)
        a2 = F.softmax(a,dim=3)
        a3 = torch.add(a1,a2)
        ab0 = torch.add(a3,b3)
        y = a3/ab0
        z = b3/ab0
        b = b*z
        a = a*y
        b4 = self.frontend4(b)
        b = torch.add(b4,b)
        a4 = self.frontend4(a)
        a = torch.add(a4,a)
        ab = torch.add(a,b)
        ab = F.upsample_bilinear(ab, size=[53,53])
        ab1 = F.softmax(ab,dim=2)
        ab2 = F.softmax(ab,dim=3)
        ab3 = torch.add(ab1,ab2)
        x1 = F.softmax(x,dim=2)
        x2 = F.softmax(x,dim=3)
        x3 = torch.add(x1,x2)
        abx = torch.add(ab3,x3)
        y = ab3/abx
        z = x3/abx
        x = x*z
        ab = ab*y
        x4 = self.frontend4(x)
        x = torch.add(x4,x)
        ab4 = self.frontend4(ab)
        ab = torch.add(ab4,ab)
        x = torch.add(ab,x)
        x = F.upsample_bilinear(x, size=[848,848])
        x = self.frontend3(x)
        x = torch.add(x0,x)
        x = self.frontend1(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
    
def make_layers(cfg, in_channels = 3,batch_norm=True,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        elif v == 'BN512':
            layers += [ nn.BatchNorm2d(512)]
        elif v == 'BN64':
            layers += [ nn.BatchNorm2d(64)]
        elif v=='CB64':
            layers += [CBAM(64)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding= d_rate, dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  

def make_layers1(cfg, in_channels = 3,batch_norm=True,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding= d_rate, dilation = d_rate)
            if batch_norm:
                layers += [nn.BatchNorm2d(v)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)              

def make_layers2(cfg, in_channels = 3,batch_norm=True,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'A':
            layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
        elif v == 'BN512':
            layers += [ nn.BatchNorm2d(512)]
        elif v == 'BN64':
            layers += [ nn.BatchNorm2d(64)]
        elif v=='CB64':
            layers += [CBAM(64)]
        elif v=='CA':
            layers += [CA(512)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding= d_rate, dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  

class CA(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CA, self).__init__()
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.inplanes=128
    def forward(self, x):
        out = self.sigmoid(self.f2(self.relu(self.f1(x))))
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.inplanes=128
    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # (特征图的大小-算子的size+2*padding)/步长+1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        #2*h*w
        x = self.conv(x)
        #1*h*w
        return self.sigmoid(x)
    
class CBAM(nn.Module):
    def __init__(self, c1, ratio=16, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        return out

if __name__ == '__main__':
    x = torch.randn(size=(1,3,848,848))
    # x = torch.randn(size=(1,64,224))
    # model = Bottlrneck(64,64,256,True)
    model = MFSnet()

    output = model(x)
    print(f'输入尺寸为:{x.shape}')
    print(f'输出尺寸为:{output.shape}')
    #print(model)
           


