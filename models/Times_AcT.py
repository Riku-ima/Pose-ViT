import math

import torch
import torch.nn as nn
import torch.nn.functional as F 
from collections import OrderedDict
from torchinfo import summary
#import Dataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader,random_split
from einops import rearrange,reduce,repeat

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)


class MHSA(nn.Module):
    def __init__(self,emb_dim:int=128,head:int=2,dropout:float=0.,with_qkv=True):
        """
        with_qkv :
        -> True なら、MHSAのために、3倍のdimに入力をq,k,vに埋め込む
        """
        
        super(MHSA,self).__init__()
        self.num_heads=head
        self.emb_dim=emb_dim
        self.head_dim=emb_dim//self.num_heads
        self.sqrt_d=emb_dim**0.5
        self.with_qkv=with_qkv
        
        #埋め込み
        if with_qkv:
            self.w_qkv=nn.Linear(self.emb_dim,self.emb_dim*3,bias=False)
            self.dropout=nn.Dropout(dropout)
        
        self.w_o=nn.Sequential(
            nn.Linear(emb_dim,emb_dim),
            nn.Dropout(dropout))
        
    def forward(self,z:torch.Tensor)->torch.Tensor:
        B,N,D=z.shape
        if self.with_qkv:
            #->(B,N,3,num_heads,head_dim)
        
            z=self.w_qkv(z).reshape(B,N,3,self.num_heads,self.head_dim)
            # ->(3,B,num_heads,N,head_dim)
            z=z.permute(2,0,3,1,4)
            q,k,v=z[0],z[1],z[2]
        #埋め込みが不要なら、ただqkv複製
        else:
            z=z.reshape(B,N,self.num_heads,self.head_dim)
            #->(B.num_heads ,N ,head_dim )
            z=z.permute(0,2,1,3)
            q,k,v=z,z,z
        
        attn=(q@k.transpose(-2,-1))/self.sqrt_d
        attn=F.softmax(attn,dim=-1)
        attn_weight=self.dropout(attn)
        # (B,heads,N,Dh) -> (B,N,heads,Dh) -> (B,N,D)
        out=(attn_weight@v).transpose(1,2).reshape(B,N,D)
        
        if self.with_qkv:
            out=self.w_o(out)
        return out
    
class Block(nn.Module):
    def __init__(self,emb_dim:int=384,num_heads:int=2,mlp_ratio=4,
                 drop:float=0.,attn_drop:float=0.,drop_path=0.1,
                attention_type='divided_space_time'):
        
        """
        mlp_ratio : nn.Linearでemb_dimの何倍に埋め込むか
        attn_drop : Attn用のdropout率
        drop      : Block用のdropout率
        """
        super(Block,self).__init__()
        self.attention_type=attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])
        
        #spational 用 Attn 
        self.drop_path=DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm_layer=nn.LayerNorm(emb_dim)
        self.attn=MHSA(emb_dim=emb_dim,head=num_heads,dropout=attn_drop)
        
        #temporal
        if self.attention_type=='divided_space_time':
            self.temporal_norm=nn.LayerNorm(emb_dim)
            self.temporal_attn=MHSA(emb_dim=emb_dim,head=num_heads,dropout=attn_drop)
            self.temporal_fc=nn.Linear(emb_dim,emb_dim)
    
        self.mlp=nn.Sequential(
            nn.Linear(emb_dim,int(emb_dim*mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(emb_dim*mlp_ratio),emb_dim),
            nn.Dropout(drop)
            )
    
    def forward(self,z,T:int=16):
        """
        input : z -> (B,N,D)  ,T frames ,P keypoints
        --------------------------------------------------
        B: batch_num , 
        N (H*W*frame)+1   今回は (keypoints × T frames )+1
        D;emb_dim  
        """
        B=z.size(0)
        P=z.size(1)//T
        if self.attention_type in ['space_only','joint_space_time']:
            z=z+self.drop_path(self.attn(self.norm_layer(z)))
            z=z+self.drop_path(self.mlp(self.norm_layer(z)))
            return z
        
        elif self.attention_type=='divided_space_time':
            ###  Temporal  ###
            #同一のパッチの箇所で、時間でattn  ,cls_tokenは処理しない!!!
            #zt (B,t*h*w,d)-> (B*h*w,t,d)
            zt=z[:,1:,:]
            zt = rearrange(zt, 'b (p t) d -> (b p) t d',b=B,p=P,t=T)
            res_temporal=self.drop_path(self.temporal_attn(self.temporal_norm(zt)))
            #(B*P,T,D) -> (B,P*t,d)
            res_temporal=rearrange(res_temporal,'(b p) t d -> b (p t) d',b=B,p=P,t=T)
            
            res_temporal=self.temporal_fc(res_temporal)
            #cls_token以外に接続
            zt=z[:,1:,:]+res_temporal
            ###  Spatial  ###
            init_cls_token =z[:,0,:].unsqueeze(1)
            #各フレーム分クラストークンを複製
            cls_token=init_cls_token.repeat(1,T,1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            zs=zt
            zs=rearrange(zs,'b (p t) m -> (b t) p m',b=B,p=P,t=T)
            zs=torch.cat((cls_token,zs),1)
            res_spatial=self.drop_path(self.attn(self.norm_layer(zs)))     
            
            cls_token=res_spatial[:,0,:]
            cls_token=rearrange(cls_token,'(b t) m -> b t m',b=B,t=T)
            #cls_tokenについて各frameの平均をとる
            cls_token=torch.mean(cls_token,1,True)
            
            res_spatial=res_spatial[:,1:,:]
            res_spatial=rearrange(res_spatial,'(b t) p m -> b (t p) m',b=B,t=T,p=P)

            res=res_spatial
            z=zt
            
            #  temporal_out(init_cls_token) + temporal_spatial_out(cls_token)
            z=torch.cat((init_cls_token,z),1)+torch.cat((cls_token,res),1)
            z=z+self.drop_path(self.mlp(self.norm_layer(z)))
            return z
        
class PatchEmbLayer(nn.Module):
    def __init__(self,in_features:int=2,emb_dim:int=8,t:int=16,p:int=18):
        """
        in_features : 各keypoints座標2次元
        emb_dim : 各keypoint(x,y)を何次元に埋め込むか、要検討
        
        """
        super().__init__()
        self.emb_dim=emb_dim
        self.emb_layer=nn.Linear(in_features,emb_dim)
        
        self.num_patch=t*p  #T*P
        self.cls_token=nn.Parameter(torch.randn(1,1,emb_dim))
        self.pos_emb=nn.Parameter(
            torch.randn(1,self.num_patch+1,emb_dim)
            )
    def forward(self,z:torch.Tensor) ->torch.Tensor:
        """
        入力z || (B,T,P) B:batch , T :frames , P :keypoints
        """
        B,T,P=z.size(0),z.size(1),z.size(2)
        z=rearrange(z,'b t (k d) -> b t k d',b=B,t=T,d=2)
        #->b t p emb_dim
        z=self.emb_layer(z)
        z=rearrange(z,'b t p e -> b (t p) e',b=B,t=T,p=P//2)
        cls_token=self.cls_token.repeat(repeats=(z.size(0),1,1))
        z_0=torch.cat((cls_token,z),1)
        z_0=z_0+self.pos_emb
        return z_0

class Times_AcT(nn.Module):
    def __init__(self,num_classes:int=10,emb_dim:int=8,num_blocks:int=7,
                 head_num:int=2,mlp_ratio:int=4,drop:float=0.,attn_drop:float=0.):
        super().__init__()
        self.input_emb_layer=PatchEmbLayer(emb_dim=emb_dim)
        self.vit_encoder=nn.Sequential(
            *[Block(emb_dim=emb_dim,num_heads=head_num,mlp_ratio=mlp_ratio,drop=drop,attn_drop=attn_drop)
             for _ in range(num_blocks)] )
        
        self.mlp_head=nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim,num_classes))
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        input_x=self.input_emb_layer(x)
        out=self.vit_encoder(input_x)
        cls_token=out[:,0]
        pred=self.mlp_head(cls_token)
        return pred
    
def generate_model(emb_dim:int=128):
    
    model=Times_AcT(emb_dim=emb_dim)
    return model