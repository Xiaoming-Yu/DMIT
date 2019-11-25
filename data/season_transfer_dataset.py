import random
import torch
import os
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SeasonTransferDataset(Dataset):
    def __init__(self, opt):
        self.image_path = opt.dataroot
        self.is_train = opt.is_train
        self.d_num = opt.n_attribute
        print ('Start preprocessing dataset..!')
        random.seed(1234)
        self.preprocess()
        print ('Finished preprocessing dataset..!')
        if self.is_train:
            trs = [transforms.Resize(opt.load_size, interpolation=Image.ANTIALIAS), transforms.RandomCrop(opt.fine_size)]
        else:
            trs = [transforms.Resize(opt.load_size, interpolation=Image.ANTIALIAS), transforms.CenterCrop(opt.fine_size)]
        if opt.is_flip:
            trs.append(transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(trs)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.num_data = max(self.num)
        
    def preprocess(self):
        dirs = os.listdir(self.image_path)
        trainDirs = [dir for dir in dirs if 'train' in dir]
        testDirs = [dir for dir in dirs if 'test' in dir]
        assert len(trainDirs) ==  self.d_num
        trainDirs.sort()
        testDirs.sort()
        self.filenames = []
        self.num = []
        if self.is_train:
            for dir in trainDirs:
                filenames = glob("{}/{}/*.jpg".format(self.image_path,dir)) + glob("{}/{}/*.png".format(self.image_path,dir))
                filenames.sort()
                random.shuffle(filenames)
                self.filenames.append(filenames)
                self.num.append(len(filenames))
        else:
            for dir in testDirs:
                filenames = glob("{}/{}/*.jpg".format(self.image_path,dir)) + glob("{}/{}/*.png".format(self.image_path,dir))
                filenames.sort()
                self.filenames.append(filenames)
                self.num.append(len(filenames))
        self.labels=[[ int(j==i) for j in range(self.d_num)] for i in range(self.d_num)]
 
    def __getitem__(self, index):
        imgs = []
        labels = []
        for d in range(self.d_num):
            index_d = index if index < self.num[d] else random.randint(0,self.num[d]-1)
            img = Image.open(self.filenames[d][index_d]).convert('RGB')
            img = self.transform(img)
            img = self.norm(img)
            imgs.append(img)
            labels.append(torch.FloatTensor(self.labels[d]))
        return imgs, labels

    def __len__(self):
        return self.num_data

