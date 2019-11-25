import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_gan_criterion(mode):
    if mode == 'dcgan':
        criterion = GANLoss(dis_loss=nn.BCEWithLogitsLoss(),gen_loss=nn.BCEWithLogitsLoss())
    elif mode == 'lsgan':
        criterion = GANLoss(dis_loss=nn.MSELoss(),gen_loss=nn.MSELoss())
    elif mode == 'hinge':
        def hinge_dis(pre, margin):
            '''margin should not be 0'''
            logict = (margin>0).float() +  (-1. * (margin<0).float())
            return torch.mean(F.relu((margin-pre)*logict))
        def hinge_gen(pre, margin):
            return -torch.mean(pre)
        criterion = GANLoss(real_label=1,fake_label=-1,dis_loss=hinge_dis,gen_loss=hinge_gen)
    else:
        raise NotImplementedError('{} is not implementation'.format(mode))
    return criterion
    
def get_rec_loss(mode):
    if mode == 'l1':
        criterion = nn.L1Loss()
    elif mode == 'mse':
        criterion = nn.MSELoss()
    else:
        raise NotImplementedError('{} is not implementation'.format(mode))
    return criterion
    
def get_kl_loss():
    def kl_loss(mu, logvar):
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        loss = torch.sum(KLD_element).mul_(-0.5)
        return loss
    return kl_loss
    

class GANLoss(nn.Module):
    def __init__(self, real_label=1., fake_label=0.,dis_loss=None,gen_loss=None):
        super(GANLoss, self).__init__()
        self.real_label = real_label
        self.fake_label = fake_label
        self.dis_loss = dis_loss
        self.gen_loss = gen_loss
        self.real_label_tensor = None
        self.fake_label_tensor = None
    
    def get_target_tensor(self, input):
        create_label =  self.real_label_tensor is None
        if not create_label:
            if isinstance(input,list):
                for pre,tar in zip(input,self.real_label_tensor):
                    create_label = create_label or pre.numel() != tar.numel()
            else:
                create_label = create_label or input.numel() != self.real_label_tensor.numel()
        if create_label:
            if isinstance(input,list):
                self.real_label_tensor = []
                self.fake_label_tensor = []
                for pre in input:
                    self.real_label_tensor.append(torch.FloatTensor(pre.size()).fill_(self.real_label).to(pre.device))
                    self.fake_label_tensor.append(torch.FloatTensor(pre.size()).fill_(self.fake_label).to(pre.device))
            else:
                self.real_label_tensor = torch.FloatTensor(input.size()).fill_(self.real_label).to(input.device)
                self.fake_label_tensor = torch.FloatTensor(input.size()).fill_(self.fake_label).to(input.device)
        return self.real_label_tensor, self.fake_label_tensor
        
    def __call__(self, real=None, fake1=None, fake2=None, weight_real=1, weight_fake1=1, weight_fake2=1):
        err = 0.0
        if not isinstance(real,list):
            real = [real]
            fake1 = [fake1] if fake1 is not None else fake1
            fake2 = [fake2] if fake2 is not None else fake2
        real_label_tensor,fake_label_tensor = self.get_target_tensor(real)
        if fake1 is not None and fake2 is not None:
            for r,f1,f2,r_label,f_label in zip(real,fake1,fake2,real_label_tensor,fake_label_tensor):
                err += self.dis_loss(r*weight_real,r_label*weight_real) + \
                       self.dis_loss(f1*weight_fake1,f_label*weight_fake1)*0.5 + \
                       self.dis_loss(f2*weight_fake2,f_label*weight_fake2)*0.5
        elif fake1 is not None or fake2 is not None:
            fake = fake1 if fake1 is not None else fake2
            for r,f,r_label,f_label in zip(real,fake,real_label_tensor,fake_label_tensor):
                err += self.dis_loss(r*weight_real,r_label*weight_real) + \
                       self.dis_loss(f*weight_fake1,f_label*weight_fake1)
        else:
            for r,r_label in zip(real,real_label_tensor):
                err += self.gen_loss(r*weight_real,r_label*weight_real)
        return err
