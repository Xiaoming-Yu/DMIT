import torch
import models.modules.network as network
from util.util import tensor2im
from models.base_model import BaseModel

TEST_SEQ = ['this small bird has a blue crown and white belly',
            'this small yellow bird has gray wings and a black bill',
            'a small brown bird with a brown crown has a white belly',
            'this black bird has no other colors with a short bill',
            'an orange bird with green wings and blue head',
            'a black bird with a red head',
            'this particular bird with a red head and breast and features grey wings']
            
################## SemanticImageSynthesis #############################
class SemanticImageSynthesisModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.enc_attribute = network.get_attribute_encoder('cub_text',opt)
        self.enc_attribute.to(self.device)
        self.enc_attribute.eval()
        
    def prepare_data(self,data):
        img, captions, captions_lens = data
        batch_size = img.size(0)
        captions_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)
        img = img[sorted_cap_indices].to(self.device)
        captions = captions[sorted_cap_indices].squeeze(dim=2).to(self.device)
        captions_lens = captions_lens.to(self.device)
        hidden = self.enc_attribute.init_hidden(batch_size)
        _, sent_emb = self.enc_attribute(captions, captions_lens, hidden)
        index_target = torch.tensor(range(-1,batch_size-1)).to(self.device)
        weight_source = torch.ones([batch_size,1]).to(self.device)
        self.current_data = [img, sent_emb, index_target,weight_source]
        return self.current_data
        
    def translation(self, data):
        with torch.no_grad():
            img, cap_ori, cap_len_ori = data
            assert img.size(0) == 1
            img = img.repeat(len(TEST_SEQ)+1,1,1,1)
            cap_tar, cap_len_tar = [cap_ori], [cap_len_ori]
            for seq in TEST_SEQ:
                cap, cap_len = self.opt.txt_dataset.cap2ix(seq)
                cap = torch.LongTensor(cap).unsqueeze(0)
                cap_len = torch.LongTensor([cap_len])
                cap_tar.append(cap)
                cap_len_tar.append(cap_len)
            cap_tar = torch.cat(cap_tar,dim=0)
            cap_len_tar = torch.cat(cap_len_tar,dim=0)
            img, sent_emb, _, _ = self.prepare_data([img,cap_tar,cap_len_tar])
            style_enc, _, _ = self.enc_style(img)
            content = self.enc_content(img)
            fakes = self.dec(content,torch.cat([sent_emb,style_enc],dim=1))
            results = [('input',tensor2im(img[0].data)),
                       ('rec',tensor2im(fakes[0].data))]
            for i in range(len(TEST_SEQ)):
                results.append(('seq_{}'.format(i+1),tensor2im(fakes[i+1].data)))
            return results
            