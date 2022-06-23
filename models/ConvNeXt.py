#
import torch
import torch.nn as nn
import torch.nn.functional as F

#
import numpy as np

# Hugging face API
from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification

# local package
from Configure import Configure


class ConvNeXt(nn.Module):

    def __init__(self, config: Configure):

        super(ConvNeXt, self).__init__()
        
        #
        self.config = config
        self.num_labels = config.model.n_classes

        # Model
        pretrained_model = config.model.pretrained_model
        self.feature_extractor = ConvNextFeatureExtractor.from_pretrained(pretrained_model)
        self.model = ConvNextForImageClassification.from_pretrained(pretrained_model)

        # Classifier
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(1000, self.num_labels)

    def feature_extract(self, imgs):
        # Change input array into list with each batch being one element

        # convert it to numpy array first
        device = torch.device('cpu')
        if imgs.device != torch.device('cpu'):
            device = torch.device(f'cuda:{self.config.use_gpu_index}')
        imgs = imgs.cpu().numpy()
        imgs = np.split(np.squeeze(np.array(imgs)), imgs.shape[0])

        # Remove unecessary dimension
        for index, array in enumerate(imgs):
            imgs[index] = np.squeeze(array)

        # Apply feature extractor, stack back into 1 tensor and then convert to tensor
        # imgs = (batch_size, 3, 224, 224)
        imgs = torch.tensor(np.stack(self.feature_extractor(imgs)['pixel_values'], axis=0))
        imgs = imgs.to(device)

        return imgs

    def forward(self, x, labels=None):

        # Feature extraction
        x = self.feature_extract(x)

        # Swin-ViT
        x = self.model(pixel_values=x)

        # x = self.dropout(x)
        logits = self.classifier(x.logits)

        return logits

'''
output of ConvNeXt
ImageClassifierOutputWithNoAttention(loss=None, logits=tensor([[-0.3647, -0.9871,  0.7566,  ...,  0.3254,  3.4399, -0.8364],
        [-0.3316,  0.9977, -0.0317,  ...,  0.8078,  0.4987,  0.1913],
        [-1.1859, -0.5237, -0.0870,  ...,  0.4682,  3.4617,  0.1552],
        ...,
        [-0.5353,  0.3237,  0.7599,  ...,  0.1663,  2.4754, -0.2040],
        [-0.0333, -0.6616,  0.2661,  ..., -0.1722, -0.6683, -0.8669],
        [-0.4033,  0.0814,  0.3842,  ...,  1.1655,  1.7428, -0.3268]],
        grad_fn=<AddmmBackward0>), hidden_states=None)
'''

'''
ConvNextConfig {
    "_name_or_path": "facebook/convnext-base-224",
    "architectures": [
        "ConvNextForImageClassification"
    ],
    "depths": [
        3,
        3,
        27,
        3
    ],
    "drop_path_rate": 0.0,
    "hidden_act": "gelu",
    "hidden_sizes": [
        128,
        256,
        512,
        1024
    ],
    "layer_norm_eps": 1e-12,
    "layer_scale_init_value": 1e-06,
    "model_type": "convnext",
    "num_channels": 3,
    "num_stages": 4,
    "patch_size": 4,
    "torch_dtype": "float32",
    "transformers_version": "4.18.0"
}

'''