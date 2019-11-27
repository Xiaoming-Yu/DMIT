"""Model class template
This module provides a template for users to implement custom models.
The filename should be <model>_model.py
The class name should be <Model>Model
It implements a simple multi-mapping translation baseline with a unified model.
You need to implement the following functions:
    <prepare_data>: Unpack input data and perform pre-processing steps.
    <translation>: Perform image translation for model evalution.
"""

import torch
from models.base_model import BaseModel

            
################## TemplateModel #############################
class TemplateModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        
    def prepare_data(self, data):
        '''prepare data for training or inference.
        
        Parameters:
            data            -- It should contain the image and the corresponding domain information.
        Returns:
            img             -- Input image.
            domain_source   -- Domain information of the input image, such as one-hot lable, attribute lable, or sentence embedding.
            index_target    -- Define the target mapping of the input image.
            weight_source   -- Define the domain weight to tackle the domain(class) imbalance problem.
        '''
        img, domain_source = data
        index_target =  torch.randperm(img.size(0))
        weight_source = torch.ones([img.size(0),1])
        self.current_data = [img.to(self.device),
                             domain_source.to(self.device),
                             index_target.to(self.device),
                             weight_source.to(self.device)]
        return self.current_data
        
    def translation(self, data):
        '''translate the input image'''
        with torch.no_grad():
            img, domain_source, index_target = self.prepare_data(data)
            domain_target = domain_source[index_target]
            style_enc, _, _ = self.enc_style(img)
            style_rand = self.sample_latent_code(style_enc.size())
            content = self.enc_content(img)
            enc_rec = self.dec(content,torch.cat([domain_source,style_enc],dim=1))
            rand_intra_domain = self.dec(content,torch.cat([domain_source,style_rand],dim=1))
            rand_inter_domain = self.dec(content,torch.cat([domain_target,style_rand],dim=1))
            return [('input': tensor2im(img.data)),
                    ('reconstruction': tensor2im(enc_rec.data)),
                    ('intra-domain translation': tensor2im(rand_intra_domain.data)),
                    ('inter-domain translation': tensor2im(rand_inter_domain.data))]
            