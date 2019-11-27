"""Dataset class template
This module provides a template for users to implement custom datasets.
The filename should be <dataset>_dataset.py
The class name should be <Dataset>Dataset
You need to implement the following functions:
    <__init__>: Initialize this dataset class.
    <__getitem__>: Return a image and its corresponding lable.
    <__len__>: Return the size of the dataset.
"""

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class TemplateDataset(Dataset):
    def __init__(self, opt):
    '''Initialize this dataset class.
       We need to specific the path of the dataset and the domain label of each image.
    '''
        self.image_list = []
        self.label_list = []
        if opt.is_train:
            trs = [transforms.Resize(opt.load_size, interpolation=Image.ANTIALIAS), transforms.RandomCrop(opt.fine_size)]
        else:
            trs = [transforms.Resize(opt.load_size, interpolation=Image.ANTIALIAS), transforms.CenterCrop(opt.fine_size)]
        if opt.is_flip:
            trs.append(transforms.RandomHorizontalFlip())
        trs.append(transforms.ToTensor())
        trs.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(trs)
        self.num_data = len(self.image_list)
        
    def __getitem__(self, index):
    '''Return a image and its corresponding lable.'''
        img = Image.open(self.image_list[index]).convert('RGB')
        img = self.transform(img)
        label = torch.FloatTensor(self.label_list[index])
        return img, label

    def __len__(self):
        '''Size of the dataset. '''
        return self.num_data

