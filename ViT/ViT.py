'''Combines all the modules to create the vision transformer architecture.
'''


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
import einops


from .patch_embedding import PatchEmbedding
from .positional_encoder import PositionalEncoder
from .transformer_encoder import TransformerEncoderNetwork
from .mlp_head import MLPHead


class VisionTransformer(nn.Module):

    def __init__(self, image_size, patch_size, patch_embedding_dim, image_depth, device, init_std=0.02, num_classes, **kwargs):
        '''Vision transformer by making use of all the other modules in this directory.
        '''

        super(VisionTransformer, self).__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_embedding_dim = patch_embedding_dim #the constant embedding's dimension throughout the network.
        self.image_depth = image_depth
        self.num_classes = num_classes
        self.device = device
        
        self.num_patches = self.image_depth * self.patch_size**2

        
        self.patch_embed_module = PatchEmbedding(patch_size=self.patch_size,
                                                 image_depth=self.image_depth,
                                                 patch_embedding=self.patch_embedding_dim,
                                                 device=self.device)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.patch_embedding_dim)) #to be concatenated with the output of the linear projection.
        #+1 in token_length param is to account for the CLS token later.
        positional_encoder_module = PositionalEncoder(token_length=self.num_patches+1, output_dim=self.patch_embedding_dim, n=10000, device=self.device)
        self.positional_embedding_tensor = positional_encoder_module() #generate the positional embeddings here once to be summed with the projected patches with CLS token later.

        self.transformer_blocks = TransformerEncoderBlock(device=self.device,
                                                          input_dim=self.patch_embedding_dim,
                                                          **kwargs).to(self.device)


        self.mlp_head = MLPHead(patch_embedding_dim=self.patch_embedding_dim,
                                num-classes=self.num_classes).to(self.device)

    
    def forward(self, x):
        
        #linear projection on the patches.
        x = self.patch_embed_module(x)
        
        #repeat the cls token based on the batch size of x.
        batched_cls_token = einops.repeat(self.cls_token, '() n e -> b n e', b=x.size(0))
        
        #concatenate the cls token with the linearly projected patches.
        x = torch.cat([batched_cls_token, x], dim=1)

        #repeat the positional encoding tensor to the batch size of x.
        batched_pos_encoded_tensors = einops.repeat(self.positional_embedding_tensor.unsqueeze(0), '() p e -> b p e', b=x.size(0))

        #add the positional encoding tensor to x.
        x = x + batched_pos_encoded_tensors

        x = self.transformer_blocks(x)
        x = self.mlp_head(x)

        return x










        
                 
