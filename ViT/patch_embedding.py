'''Module to produce linear projection of the patched images (tokens).
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    '''The idea of patch embedding is to divide a given image into same-size grids and project each grids into another dimension linearly.
       In this implementation, we will be using Conv2d to linearly project each patch without overlapping.
    '''

    def __init__(self, patch_size, image_depth, embedding_dim, device):
        
        super(PatchEmbedding, self).__init__()

        self.patch_projection = nn.Conv2d(image_depth,
                                          embedding_dim,
                                          kernel_size=patch_size,
                                          stride=patch_size).to(device)

    def forward(self, x):
        
        
        x = self.patch_projection(x) #The output will be [B, embedding_dim, height * number_patches, width*num_patches][
        x = x.flatten(2) #this will flatten the height and width patches at the last 2 dimensions.
        x = x.transpose(1,2) #we want the tensor to be [B, num_patches, embedding_dim]


        



        
        
        
        
        



