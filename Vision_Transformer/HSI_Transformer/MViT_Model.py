import torch
import torch.nn as nn
import numpy as np
from CBAM import CBAM

class SSA(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.ss_attention = nn.Sequential(
            
            nn.Conv2d(self.in_channels,self.in_channels,kernel_size=3),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels,self.in_channels,kernel_size=3),
            nn.BatchNorm2d(self.in_channels),
            CBAM(self.in_channels),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(self.in_channels,self.out_channels,kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.ss_attention(x)
        return x

class MViT_Embeddings(nn.Module):
    
    def __init__(self, in_channels, embed_dim, image_size, seq_patch_size, dropout):
        super().__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.seq_patch_size = seq_patch_size
        self.patch_embedding = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.seq_patch_size,
            stride=self.seq_patch_size
        )
        
        self.num_patches = (self.image_size // self.seq_patch_size) ** 2
        self.num_positions = self.num_patches
        
        self.cls_token = nn.Parameter(torch.randn(size=(1,1,self.embed_dim)),requires_grad=True)
        self.position_embedding = nn.Embedding(self.num_positions+1, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions+1).expand((1,-1)),
            persistent=False
        )
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        
        patch_embeds = self.patch_embedding(pixel_values)
        
        embeddings = patch_embeds.flatten(2)
        
        embeddings = embeddings.transpose(1,2)
        
        cls_token = self.cls_token.expand(pixel_values.shape[0],-1,-1)
        
        embeddings_with_cls_token = torch.cat([cls_token,embeddings],dim=1)
        
        embeddings = embeddings_with_cls_token + self.position_embedding(self.position_ids)
        
        return embeddings
    
class MViT(nn.Module):
    
    def __init__(self, image_size, num_classes, seq_patch_size, embed_dim, num_encoders, num_heads, dropout, in_channels,out_channels_for_ssa, activation):
        super().__init__()
        
        self.spectral_spatial_attn = SSA(in_channels,out_channels_for_ssa)
        temp = torch.randn(2,in_channels,image_size,image_size)
        final_image_size = self.spectral_spatial_attn(temp).shape[2]
        self.embeddings = MViT_Embeddings(out_channels_for_ssa, embed_dim, final_image_size, seq_patch_size, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,nhead=num_heads,dropout=dropout,activation=activation, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer,num_layers=num_encoders)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim,eps=1e-6),
            nn.Linear(in_features=embed_dim,out_features=num_classes)
        )
        
    def forward(self,pixel_values):
        ss_attention_output = self.spectral_spatial_attn(pixel_values)
        hidden_states = self.embeddings(ss_attention_output)
        last_hidden_states = self.encoder(hidden_states)
        logits = self.mlp_head(last_hidden_states[:,0,:])
        return logits
        
    
# vit = MViT(11,76,3,1024,3,8,0,256,128,"gelu")
# x = torch.randn(2,256,11,11)
# # c = SSA(256,128)
# # print(c(x).shape)
# print(vit(x).shape)
