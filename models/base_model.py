import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import models.modules.network as network
import util.loss as loss
import util.util as util
from collections import OrderedDict
from torchvision.utils import make_grid
from abc import  ABC, abstractmethod

################## BaseModel #############################
class BaseModel(ABC):
    def __init__(self, opt):
        if opt.gpu>=0:
            self.device = torch.device('cuda:%d'%opt.gpu)
            torch.cuda.set_device(opt.gpu)
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')
        self.opt = opt
        self.start_epoch=0
        self.current_data=None
        self.nets={}
        self.optimizers={}
        self.lr_schedulers={}
        self.build_models()
        
    def build_models(self):
        self.dec = network.get_decoder(self.opt)
        self.enc_style = network.get_style_encoder(self.opt)
        self.enc_content = network.get_content_encoder(self.opt)
        if self.opt.is_train:
            self.dis_content = network.get_content_discriminator(self.opt)
            self.dis_rand_prior = network.get_image_discriminator(self.opt)
            self.dis_rand_enc = self.dis_rand_prior
            ckpt = None
            if self.opt.continue_train:
                ckpt = torch.load('{}/resume.pth'.format(self.opt.model_dir),map_location='cpu')
                self.start_epoch = ckpt['epoch']
            self._init_net(self.dec,'dec',ckpt)
            self._init_net(self.enc_style,'enc_style',ckpt)
            self._init_net(self.enc_content,'enc_content',ckpt)
            self._init_net(self.dis_content,'dis_content',ckpt)
            self._init_net(self.dis_rand_prior,'dis_rand_prior',ckpt)
            if self.opt.use_same_dis:
                self.dis_rand_enc = self.dis_rand_prior
            else:
                self.dis_rand_enc = network.get_image_discriminator(self.opt)
                self._init_net(self.dis_rand_enc,'dis_rand_enc',ckpt)
            self.criterion_content = loss.get_gan_criterion(self.opt.content_gan_mode)
            self.criterion_image = loss.get_gan_criterion(self.opt.image_gan_mode)
            self.criterion_rec = loss.get_rec_loss(self.opt.rec_mode)
            self.criterion_kl = loss.get_kl_loss()
        else:
            self._eval_net(self.dec,'dec')
            self._eval_net(self.enc_style,'enc_style')
            self._eval_net(self.enc_content,'enc_content')
    
    def _init_net(self, net, net_name, ckpt):
        net.to(self.device)
        net_optimizer = self.define_optimizer(net)
        if ckpt is not None:
            net.load_state_dict(ckpt[net_name]['weight'])
            net_optimizer.load_state_dict(ckpt[net_name]['optimizer'])
            lr_scheduler = util.get_scheduler(net_optimizer,self.opt,ckpt['epoch'])
        else:
            net.apply(network.weights_init(self.opt.init_type))
            lr_scheduler = util.get_scheduler(net_optimizer,self.opt,-1)
        net.train()
        self.nets[net_name] = net
        self.optimizers[net_name] = net_optimizer
        self.lr_schedulers[net_name] = lr_scheduler

    
    def _eval_net(self, net, net_name):
        net.load_state_dict(torch.load('{}/{}_{}.pth'.format(self.opt.model_dir,net_name,self.opt.which_epoch),map_location='cpu'))
        net.to(self.device)
        net.eval()

    def sample_latent_code(self, size, std=1):
        code = torch.FloatTensor(size).normal_()
        return code.to(self.device) * std
        
    def define_optimizer(self, net):
        return optim.Adam([{'params': net.parameters(), 'initial_lr': self.opt.lr}],
                          lr=self.opt.lr,
                          betas=(0.5, 0.999))
    def update_lr(self):
        for _, scheduler in self.lr_schedulers.items():
            scheduler.step()
        lr = self.optimizers['dec'].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def save_generator(self, epoch):
        for net_name, net in self.nets.items():
            if 'dis' not in net_name:
                torch.save(net.state_dict(), '{}/{}_{}.pth'.format(self.opt.model_dir,net_name,epoch))
    
    def save_ckpt(self, epoch):
        ckpt = {'epoch':epoch}
        for net_name in self.nets:
            ckpt[net_name]={'weight':self.nets[net_name].state_dict(),
                            'optimizer':self.optimizers[net_name].state_dict()}
        torch.save(ckpt,'{}/resume.pth'.format(self.opt.model_dir,net_name))
    
    
    @abstractmethod
    def prepare_data(self, data):
        '''prepare data for training or inference'''
    
    @abstractmethod
    def translation(self, data):
        '''translate the input image'''
        
    def get_current_errors(self):
        return self.current_losses
        
    def get_current_visuals(self):
        with torch.no_grad():
            self.enc_content.eval()
            self.enc_style.eval()
            self.dec.eval()
            real, attr_source, index_target,_ = self.current_data
            batch_size,_,h,w = real.size()
            content = self.enc_content(real)
            attr_target = attr_source[index_target]
            style_enc, _, _ = self.enc_style(real)
            style_exc = style_enc[index_target]
            style_rand = self.sample_latent_code(style_enc.size())
            input_code = [torch.cat([attr_source,style_enc],dim=1),
                          torch.cat([attr_target,style_exc],dim=1),
                          torch.cat([attr_target,style_rand],dim=1)]
            input_code = torch.cat(input_code,dim=0)
            input_content = content.repeat(3,1,1,1)
            fakes = self.dec(input_content,input_code)
            self.dec.train()
            self.enc_style.train()
            self.enc_content.train()
            fakes = torch.split(fakes, batch_size, dim=0)
            imgs = torch.cat([real,*fakes],dim=3)
            imgs = make_grid(imgs,nrow=1)
            imgs = util.tensor2im(imgs)
            return {'real,rec,fake_enc,fake_rand':imgs}
    
    def update_dis_image(self, dis, dis_opt, real, attr_source, weight_source, fake, attr_target, weight_target):
        dis.zero_grad()
        pred_fake_input = dis(fake, attr_target)
        pred_fake_code = dis(real, attr_target)
        pred_real = dis(real, attr_source)
        errD = self.criterion_image(real=pred_real,weight_real=weight_source.view(-1,1,1,1),
                                    fake1=pred_fake_code,weight_fake1=weight_source.view(-1,1,1,1),
                                    fake2=pred_fake_input,weight_fake2=weight_target.view(-1,1,1,1))
        errD.backward()
        dis_opt.step()
        return errD
        
    def update_dis_content(self, dis, dis_opt, real, attr_source, fake, attr_target):
        dis.zero_grad()
        pred_fake = dis(fake, attr_target)
        pred_real = dis(real, attr_source)
        errD = self.criterion_content(real=pred_real,fake1=pred_fake)
        errD.backward()
        dis_opt.step()
        return errD
        
    def calculate_gen_image(self, dis, fake, attr_target, weight_target):
        pred_fake = dis(fake, attr_target)
        errG = self.criterion_image(real=pred_fake,weight_real=weight_target.view(-1,1,1,1))
        return errG
    
    def calculate_gen_content(self, dis, fake, attr_target):
        pred_fake = dis(fake, attr_target)
        errG = self.criterion_content(real=pred_fake)
        return errG
        
    def update_model(self):
        ### prepare data ###
        real, attr_source, index_target, weight_source = self.current_data
        batch_size = real.size(0)
        attr_target = attr_source[index_target]
        weight_target = weight_source[index_target]
        ### generate image ###
        content = self.enc_content(real)
        style_enc, mu, logvar = self.enc_style(real)
        style_rand_enc = style_enc[index_target]
        style_rand_prior = self.sample_latent_code(style_enc.size(),weight_target)
        if self.opt.lambda_div: 
            style_rand_prior2 = self.sample_latent_code(style_enc.size(),weight_target)
            input_content = content.repeat(4,1,1,1)
            input_code = [torch.cat([attr_source,style_enc],dim=1),
                          torch.cat([attr_target,style_rand_enc],dim=1),
                          torch.cat([attr_target,style_rand_prior],dim=1),
                          torch.cat([attr_target,style_rand_prior2],dim=1)]
            input_code = torch.cat(input_code,dim=0)
        else:
            input_content = content.repeat(3,1,1,1)
            input_code = [torch.cat([attr_source,style_enc],dim=1),
                          torch.cat([attr_target,style_rand_enc],dim=1),
                          torch.cat([attr_target,style_rand_prior],dim=1)]
            input_code = torch.cat(input_code,dim=0)
        fakes = self.dec(input_content,input_code)
        fakes =  torch.split(fakes, batch_size, dim=0)
        _, mu_rec, _  = self.enc_style(fakes[2])
        content_rec = self.enc_content(fakes[2])
        if self.opt.lambda_cyc > 0:
            cyc = self.dec(content_rec,torch.cat([attr_source,style_enc],dim=1))
        ### update discriminator ###
        errD_content = self.update_dis_content(self.dis_content, self.optimizers['dis_content'],
                        content.detach(),attr_source, content.detach(), attr_target)
        errD_rand_prior = self.update_dis_image(self.dis_rand_prior, self.optimizers['dis_rand_prior'],
                    real, attr_source, weight_source, fakes[2].detach(), attr_target, weight_target)
        errD_rand_enc = self.update_dis_image(self.dis_rand_enc, self.optimizers['dis_rand_prior'] if self.opt.use_same_dis else self.optimizers['dis_rand_enc'],
                    real, attr_source, weight_source, fakes[1].detach(), attr_target, weight_target)
        ### update generator ###
        self.enc_content.zero_grad()
        self.enc_style.zero_grad()
        self.dec.zero_grad()
        errG_rand_enc = self.calculate_gen_image(self.dis_rand_enc, fakes[1], attr_target, weight_target) 
        errG_rand_prior = self.calculate_gen_image(self.dis_rand_prior, fakes[2], attr_target, weight_target)
        errG_content = self.calculate_gen_content(self.dis_content, content, attr_target)
        errRec = self.criterion_rec(fakes[0],real) * self.opt.lambda_rec
        errKL = self.criterion_kl(mu,logvar) * self.opt.lambda_kl
        errCyc = 0 
        if self.opt.lambda_cyc > 0:
            errCyc = self.criterion_rec(cyc,real) * self.opt.lambda_cyc
        errG_total = errG_content + errG_rand_prior + errG_rand_enc +\
                     errRec + errKL + errCyc
        errG_total.backward(retain_graph=True)
        self.optimizers['dec'].step()
        self.optimizers['enc_style'].step()
        self.optimizers['enc_content'].step()
        ###encourage decoder to make use random style###
        self.dec.zero_grad()
        errStyle = torch.mean(torch.abs(mu_rec-style_rand_prior)) * self.opt.lambda_style
        errDiv = 0
        if self.opt.lambda_div > 0:
            errDiv =  torch.mean(torch.abs(style_rand_prior-style_rand_prior2)) / \
                      (torch.mean(torch.abs(fakes[2]-fakes[3]))+1e-5) * \
                       self.opt.lambda_div
        errDec_total = errStyle + errDiv
        errDec_total.backward(retain_graph=True)
        self.optimizers['dec'].step()
        ###encourage content decoder to extract domain-invariant content###
        self.enc_content.zero_grad()
        errContent = torch.mean(torch.abs(content-content_rec)) * self.opt.lambda_content
        errContent.backward()
        self.optimizers['enc_content'].step()
        ###save current losses###
        dict = []
        dict += [('D_rand_prior', errD_rand_prior.item())]
        dict += [('G_rand_prior', errG_rand_prior.item())]
        dict += [('D_rand_enc', errD_rand_enc.item())]
        dict += [('G_rand_enc', errG_rand_enc.item())]
        dict += [('D_content', errD_content.item())]
        dict += [('G_content', errG_content.item())]
        dict += [('errRec', errRec.item())]
        dict += [('errKl', errKL.item())]
        dict += [('errStyle', errStyle.item())]
        dict += [('errContent', errContent.item())]
        if self.opt.lambda_cyc>0:
            dict += [('errCyc', errCyc.item())]
        if self.opt.lambda_div>0:
            dict += [('errDiv', errDiv.item())]
        self.current_losses = OrderedDict(dict)
        
