'''Module to produce patch embeddings from the given image dataset.
'''

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn

from .ImagePatchMLP import ImagePatchMLP
from .PositionalEncoder import PositionalEncoder

class PatchEmbedding(nn.Module):
    '''Responsible for dividing the given image batch into patches and produce an embedding from them.
    '''


    def __init__(self, image_height=224, image_width=224, image_channel=1, patch_size=16, stride=16, linear_out_dimension=16**2):
        '''Param init.
        '''
        super(PatchEmbedding, self).__init__()

        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.patch_size = patch_size
        self.stride = stride
        self.linear_out_dimension = linear_out_dimension
        
        #initialize the linear projection module.
        self.patch_linear_module = ImagePatchMLP(flattened_img_dimension=patch_size**2, output_dimension=self.linear_out_dimension)

        #initialize the unfold function.
        self.unfolding_func = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size )
    
    def linear_projection_patches(self, patched_image_tensors):
        '''Linearly projects the image patches.
        '''

        return self.patch_linear_module(patched_image_tensors)
        
    def cls_token_concat(self, linear_projected_tensors):
        '''Receives the image patches that has been linearly projected and appends a learnable parameter tensor at the end of the input tensor.
            
        Input:
            patched_image_tensors -- A tensor of shape [batch size, total number of patches, a single flattened image dimension].
        '''
        batch_size = linear_projected_tensors.size(0)
        cls_token = torch.ones(batch_size, 1, self.linear_out_dimension)

        cls_concat_tensors = torch.cat([linear_projected_tensors, cls_token], dim=1)

        return cls_concat_tensors


    def get_non_overlapping_patches(self, inp):
        '''Break the batch image tensors to N non-overlapping patches.
        '''
        
        patched_image_tensors = self.unfolding_func(inp) #this will return a tensor of shape [batch size, a single flattened image patch dimension, total number of patches]
        patched_image_tensors = patched_image_tensors.permute(0, 2, 1) #to keep things consistent with the paper, we permute the dimensions to  [batch size, total number of patches, a single flattened image patch dimension]. Also, the linear projection happens to the image dimensions, not the number of patches. So this makes more sense.


        return patched_image_tensors 


    
    def __call__(self, batched_tensor_images):
        '''Given a batched images in a tensor, perform the Patch Embedding and return the result.
        
        Input:
            batched_tensor_images --  A tensor of shape [batch size, image channel, image height, image width]
        '''

        patched_image_tensors = self.get_non_overlapping_patches(inp=batched_tensor_images)
        linear_projected_tensors = self.linear_projection_patches(patched_image_tensors=patched_image_tensors)
        
        cls_token_concat_tensors = self.cls_token_concat(linear_projected_tensors=linear_projected_tensors)

        positional_encoding_module = PositionalEncoder(token_length=cls_token_concat_tensors.size(1), output_dim=cls_token_concat_tensors.size(2), n=1000)
        positional_encoding_tensor = positional_encoding_module() #tensor of size [num_patches+1, flattened image patch dimension]

        #in order to perform element-wise addition to the projected tensor with the CLS token, we're gonna have to stack up the positional encoding for every element in the batch.
        stacked_pos_enc_tensor = positional_encoding_tensor.unsqueeze(0).repeat_interleave(cls_token_concat_tensors.size(0), dim=0)

        patch_embeddings = torch.add(cls_token_concat_tensors, stacked_pos_enc_tensor)  

        return patch_embeddings 
    