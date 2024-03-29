'''Main module for Vision Transformer.
'''
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn


class VisionTransformer(nn.Sequential):
    '''ViT Architecture.
    '''

    def __init__(self):
        pass

