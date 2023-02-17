import os
import time
import shutil
import numpy as np
import tensorflow as tf
import torch.nn.functional as F
import torch.nn as nn
import torch
from functools import partial

#modelのchannel数リスト
def get_inplanes():
    return [64,128,256,512]

class Block(nn.Module):
    """
    ResNet 10,18,34用のBLOCK
    2層の畳み込み層、バッチ正規化、relu 残差接続
    """
    expansion=1
    def __init__(self,in_features,out_features,stride=1,downsample=None):
        super().__init__()
        self.conv1=nn.Conv3d(in_features,out_features,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm3d(out_features)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv3d(out_features,out_features,kernel_size=3,padding=1,bias=False)
        self.bn2=nn.BatchNorm3d(out_features)
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        res=x
        out=self.relu(self.bn1(self.conv1(x)))
        
        out=self.bn2(self.conv2(out))
        
        out+=res
        out=self.relu(out)
        
        return out
    
class BottleNeck(nn.Module):
    """
    ResNet 50,101,152,200用のBLOCK
    conv1*1*1 -> conv3*3*3 ->conv1*1*1 バッチ正規化、relu 残差接続
    """
    expansion=4
    
    def __init__(self,in_features,features,stride=1,downsample=None):
        """
        in_features:
        features:
        
        """
        super().__init__()
        self.conv1=nn.Conv3d(in_features,features,kernel_size=1,stride=stride,bias=False)
        self.bn1=nn.BatchNorm3d(features)
        self.relu=nn.ReLU(inplace=True)
        
        self.conv2=nn.Conv3d(features,features,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm3d(features)
        self.conv3=nn.Conv3d(features,features*self.expansion,kernel_size=1,stride=stride,bias=False)
        self.bn3=nn.BatchNorm3d(features*self.expansion)
        self.downsample=downsample
        self.stride=stride
        
    def forward(self,x):
        residual=x
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.relu(self.bn2(selfl.conv2(out)))
        
        out=self.conv3(out)
        out=self.bn3(out)
        
        if self.downsample is not None:
            residual=self.downsample(x)
        out+=residual
        out=self.relu(out)
        
        return out
        

class ResNet_3D(nn.Module):
    def __init__(self,block,
                 layer_nums,
                 block_in_channels,
                 conv_size=7,
                 conv_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_facter=1.0,
                 n_classes=10):
        """
        block : 3d_residual_block
        layer_nums : [1,1,1,1] 等、各residual blockが何個か
        block_in_channels: 各blockの入力channels [64,128,256,512]
        
        """
        super().__init__()
        # もし入力channelをおおおきくするなら
        block_in_channels=[int(x*widen_facter) for x in block_in_channels]
        self.in_feature=block_in_channels[0]
        self.conv1=nn.Conv3d(3,self.in_feature,
                             kernel_size=(conv_size,7,7),
                             stride=(conv_stride,2,2),
                             padding=(conv_size//2,3,3),
                             bias=False)
        self.bn1=nn.BatchNorm3d(self.in_feature)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool3d(kernel_size=3,stride=2,padding=1)
        
        # Layer1
        self.layer1=self._makelayer(block,block_in_channels[0],layer_nums[0],shortcut_type)
        
        # Layer2
        self.layer2=self._makelayer(block,
                                    block_in_channels[1],
                                   layer_nums[1],
                                   shortcut_type,
                                   stride=2)
        self.layer3=self._makelayer(block,
                                   block_in_channels[2],
                                   layer_nums[2],
                                   shortcut_type,
                                   stride=2)
        self.layer4=self._makelayer(block,
                                   block_in_channels[3],
                                   layer_nums[3],
                                   shortcut_type,
                                   stride=2)
        #global avg pooling + linear層
        #新規モデルならいらないはず
        self.avgpool=nn.AdaptiveAvgPool3d((1,1,1))
        self.fc=nn.Linear(block_in_channels[3]*block.expansion,n_classes)
        
        
        #各重みの初期化方法
        for m in self.modules():
            #conv3dはHeの初期化
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                       mode='fan_out',
                                       nonlinearity='relu')
            #Batchnorm3dは重み1 , bias 0で初期化
            elif isinstance(m,nn.BatchNorm3d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
        
    def _downsample_basic_block(self,x,features,stride):
        out=F.avg_pool3d(x,kernel_size=1,stride=stride)
        zero_pads=torch.zeros(out.size(0),features-out.size(1),out.size(2),out.size(3),out.size(4))
        if isinstance(out.data,torch.cuda.FloatTensor):
            zero_pads=zero_pads.cuda()
        
        #channel (dim=1)  で  padding ?
        out=torch.cat([out.data,zero_pads],dim=1)
        return out
    
    def _makelayer(self,block,features,block_nums,shortcut_type,stride=1):
        """
        block : residual_block  ,  feature: 各レイヤーチャネル
        block_nums : 各レイヤー何ブロックか　　
        """
        downsample=None
        
        
        if stride!=1 or self.in_feature != features*block.expansion:
            if shortcut_type=='A':
                downsample=partial(self.downsample_basic_block,
                                  planes=features*block.expansion,
                                  stride=stride)
            else:
                downsample=nn.Sequential(nn.Conv3d(self.in_feature,features*block.expansion,kernel_size=1,stride=stride,bias=False),
                                        nn.BatchNorm3d(features*block.expansion))
             
        layers=[]
        # 最初は self.in_feature -> channel
        layers.append(
            block(in_features=self.in_feature,
                out_features=features,
                stride=stride,
                downsample=downsample))
        
        #self.in_featurをlayer作成するたび更新
        self.in_feature=features*block.expansion
        #作成するresidual_blockを何個接続するか
        for i in range(1,block_nums):
            layers.append(block(self.in_feature,features))
        
        return nn.Sequential(*layers)
        
    
    def forward(self,x):
        x=self.relu(self.bn1(self.conv1(x)))
        if not self.no_max_pool:
            x=self.maxpool(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=self.avgpool(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x
    # *wargs : 複数の引数をタプルで受け取る　**kwargs : 複数のキーワード引数を辞書として受け取る
def generate_model(model_depth,**kwargs):
    #3Dres-netのblockの数
    assert model_depth in [10,18,34,50,101,152,200]
    block_in_channels =[64, 128, 256, 512]
    if model_depth ==10:
        model=ResNet_3D(Block,[1,1,1,1],block_in_channels,**kwargs)
    elif model_depth ==18:
        model=ResNet_3D(Block,[2,2,2,2],block_in_channels,**kwargs)
    elif model_depth == 34:
        model = ResNet(Block, [3, 4, 6, 3],block_in_channels, **kwargs)
    
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3],block_in_channels, **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3],block_in_channels, **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3],block_in_channels, **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3],block_in_channels, **kwargs)
    
    return model