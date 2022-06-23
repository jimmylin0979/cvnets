#
import torch
from torchvision.datasets import ImageFolder

# Inherit from torchvision.datasets.ImageFolder
class ImageLabelDataSet(ImageFolder):

    def __init__(self, root, transform_set):

        #
        super(ImageLabelDataSet, self).__init__(
            root=root, transform=transform_set)

    # help to get images for visualizing
    def getbatch(self, indices):
        '''
            @ Params : 
                1. indices (python.list)
            @ Returns : 
                1. images (torch.tensor with shape (1, ))
                2. labels (torch.tensor with shape (1, ))
        '''
        images = []
        labels = []
        for index in indices:
            image, label = self.__getitem__(index)
            # transform_ToTensor =  transforms.Compose([
            #                         transforms.Resize((224, 224)),
            #                         transforms.ToTensor()])
            # image = transform_ToTensor(image)

            images.append(image)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)
