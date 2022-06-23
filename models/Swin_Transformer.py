#
import torch
import torch.nn as nn
import torch.nn.functional as F

#
import numpy as np

# Hugging face API
from transformers import AutoFeatureExtractor, SwinForImageClassification   

from Configure import Configure


class Swin_Transformer(nn.Module):

    def __init__(self, config: Configure):

        super(Swin_Transformer, self).__init__()

        #
        self.config = config
        self.num_labels = config.model.n_classes

        # 
        # pretrained_model = 'microsoft/swin-base-patch4-window7-224'
        pretrained_model = config.model.pretrained_model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model)
        self.model = SwinForImageClassification.from_pretrained(pretrained_model)

        # 
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
        imgs = torch.tensor(
            np.stack(self.feature_extractor(imgs)['pixel_values'], axis=0))
        imgs = imgs.to(device)

        return imgs

    def forward(self, x, labels=None):

        # Feature extraction
        x = self.feature_extract(x)

        # Swin-Transformer
        x = self.model(pixel_values=x)
        logits = self.classifier(x.logits)

        return logits
