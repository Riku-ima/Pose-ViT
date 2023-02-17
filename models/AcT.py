import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from einops import rearrange,reduce,repeat

class VitembLayer(nn.Module):
    def __init__(self,num_patch:int=16,input_dim:int=18*2,emb_dim:int=128):
        super(VitembLayer,self).__init__()
        self.emb_dim=emb_dim
        #各frameをパッチに....
        self.num_patch=num_patch
        self.input_dim=input_dim
        
        self.patch_emb_layer=nn.Linear(self.input_dim,self.emb_dim)
        self.cls_token=nn.Parameter(torch.randn(1,1,emb_dim))
        self.pos_emb=nn.Parameter(
            torch.randn(1,self.num_patch+1,emb_dim))
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        """
        in : x (B,N,T,P) N人　T frame , P keypoints
        """
        #x -> (B*N,T,P)
        if x.dim()==4:
            x=x.reshape(-1,self.num_patch,self.input_dim)
        z_0=self.patch_emb_layer(x)
        cls_token=self.cls_token.repeat(repeats=(x.size(0),1,1))
        z_0=torch.cat([cls_token,z_0],dim=1)
        
        # Add positional emb
        z_0=z_0+self.pos_emb
        return z_0


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
    
class VitEncoderBlock(nn.Module):
    def __init__(self,emb_dim:int=128,head:int=2,hidden_dim:int=128*2,dropout:float=0):
        super(VitEncoderBlock,self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.ln1=nn.LayerNorm(emb_dim)
        self.mhsa=MHSA(emb_dim=emb_dim,head=head,dropout=dropout)
        self.ln2=nn.LayerNorm(emb_dim)
        self.mlp=nn.Sequential(
            nn.Linear(emb_dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,emb_dim),
            nn.Dropout(dropout))
    
    def forward(self,z:torch.Tensor)->torch.Tensor:
        out1=self.mhsa(self.ln1(z))+z
        out2=self.mlp(self.ln2(out1))+out1    
        return out2


class PoseViT(nn.Module):
    def __init__(self,num_classes:int=10,emb_dim:int=128,num_patch:int=16,num_blocks:int=7,head_num:int=2,hidden_dim:int=128*2,dropout:float=0.):
        """
        num_patch : 今verでは動画のframe: 16
        num_blocks : 何個のencodorblokc か
        """
        super(PoseViT,self).__init__()
        
        self.input_emb_layer=VitembLayer(num_patch=num_patch,emb_dim=emb_dim)
        self.vit_encoder=nn.Sequential(
            *[VitEncoderBlock(emb_dim,head_num,hidden_dim,dropout)
            for _ in range(num_blocks)]
        )
        self.mlp_head=nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim,num_classes)
        )
    
    def forward(self,x:torch.Tensor) ->torch.Tensor:
        input_x=self.input_emb_layer(x)
        out=self.vit_encoder(input_x)
        cls_token=out[:,0]
        #(B,D) -> (B,cls)
        pred=self.mlp_head(cls_token)
        return pred

def generate_model(model_size,**kwargs):
    assert model_size in['micro','S','M','L']
    if model_size=='micro':
        model=PoseViT(emb_dim=64,head_num=1,hidden_dim=256,**kwargs)
    if model_size=='S':
        model=PoseViT(emb_dim=128,head_num=2,hidden_dim=256,**kwargs)
    if model_size=='M':
        model=PoseViT(emb_dim=192,head_num=3,hidden_dim=256,**kwargs)
    if model_size=='L':
        model=PoseViT(emb_dim=256,head_num=4,hidden_dim=512,**kwargs)
    return model