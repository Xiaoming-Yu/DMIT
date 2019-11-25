import math
import functools
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.spectral_norm  import spectral_norm
from .cbin import CBINorm2d
from .cbbn import CBBNorm2d
from .adain import AdaINorm2d

def get_decoder(opt):
    return Decoder(input_nc=opt.nef*(2**opt.n_ds_blocks),
                   output_nc=opt.output_nc,
                   n_latent=opt.n_attribute+opt.n_style,
                   n_us_blocks=opt.n_ds_blocks,
                   n_resblocks=opt.n_dec_resblocks,
                   norm_type=opt.norm_type,
                   up_type=opt.up_type,
                   use_dropout=opt.use_dropout)

def get_style_encoder(opt):
    return StyleEncoder(input_nc=opt.input_nc,
                        nef=opt.nef,
                        n_style=opt.n_style,
                        n_blocks=opt.n_style_blocks,
                        norm_type=opt.norm_type)

def get_content_encoder(opt):
    return ContentEncoder(input_nc=opt.input_nc,
                          nef=opt.nef,
                          n_ds_blocks=opt.n_ds_blocks,
                          n_resblocks=opt.n_content_resblocks,
                          norm_type=opt.norm_type)
    

def get_content_discriminator(opt):
    return StyleDiscriminator(input_nc=opt.nef*(2**opt.n_ds_blocks),
                              ndf=opt.nef*(2**opt.n_ds_blocks),
                              n_block=opt.n_content_disblocks,
                              norm_type=opt.norm_type,
                              n_latent=opt.n_attribute)

def get_image_discriminator(opt):
    return MultiStyleDiscriminator(input_nc=opt.input_nc,
                                   ndf=opt.ndf,
                                   n_block=opt.n_image_disblocks,
                                   norm_type=opt.norm_type,
                                   n_latent=opt.n_attribute)

def get_attribute_encoder(attr_type, opt):
    if attr_type == 'cub_text':
        encoder = RNN_ENCODER(opt.n_words, nhidden=opt.n_attribute)
        state_dict = torch.load('./encoder/cub_text.pth',
                            map_location=lambda storage, loc: storage)
        encoder.load_state_dict(state_dict)
        for p in encoder.parameters():
            p.requires_grad = False
        print('Load text encoder successful')
    elif attr_type == 'identity':
        encoder = lambda x: x
    else:
        raise NotImplementedError('attribute encoder [%s] is not found' % attr_type)
    return encoder
    
def get_norm_layer(layer_type='cbin', num_con=0):
    if layer_type == 'cbbn':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        c_norm_layer = functools.partial(CBBNorm2d, affine=True, num_con=num_con)
    elif layer_type == 'cbin':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        c_norm_layer = functools.partial(CBINorm2d, affine=False, num_con=num_con)
    elif layer_type == 'adain':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        c_norm_layer = functools.partial(AdaINorm2d, num_con=num_con)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % layer_type)
    return norm_layer, c_norm_layer

def get_nl_layer(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'sigmoid':
        nl_layer = nn.Sigmoid
    elif layer_type == 'tanh':
        nl_layer = nn.Tanh
    else:
        raise NotImplementedError('nl_layer layer [%s] is not found' % layer_type)
    return nl_layer    

def weights_init(init_type='xavier'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif (classname.find('Norm') == 0):
            if hasattr(m, 'weight') and m.weight is not None:
                init.constant(m.weight.data, 1.0)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
    return init_fun    
     
def conv3x3(in_dim, out_dim, norm_layer=None, nl_layer=None):
    return Conv2dBlock(in_dim, out_dim, kernel_size=3, stride=1, padding=1,
            pad_type='reflect', bias=False, norm_layer=norm_layer, nl_layer=nl_layer)                     

## The code of RNN_ENCODER is modified from AttnGAN (https://github.com/taoxugit/AttnGAN)
class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = 18
        self.ntoken = ntoken
        self.ninput = ninput
        self.drop_prob = drop_prob
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.rnn_type = 'LSTM'
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.nhidden = nhidden // self.num_directions
        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        emb = self.drop(self.encoder(captions))
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        output, hidden = self.rnn(emb, hidden)
        output = pad_packed_sequence(output, batch_first=True)[0]

        words_emb = output.transpose(1, 2)
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb

## The code of LayerNorm is modified from MUNIT (https://github.com/NVlabs/MUNIT)
class LayerNorm(nn.Module):
  def __init__(self, n_out, eps=1e-5, affine=True):
    super(LayerNorm, self).__init__()
    self.n_out = n_out
    self.affine = affine
    if self.affine:
      self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
      self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
    
  def forward(self, x):
    normalized_shape = x.size()[1:]
    if self.affine:
      return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
    else:
      return F.layer_norm(x, normalized_shape)
    
class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0,
                pad_type='reflect', bias=True, norm_layer=None, nl_layer=None):
        super(Conv2dBlock, self).__init__()
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        self.conv = spectral_norm(nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size,
                    stride=stride, padding=0, bias=bias))
        if norm_layer is not None:
            self.norm = norm_layer(out_dim)
        else:
            self.norm = lambda x: x
        
        if nl_layer is not None:
            self.activation = nl_layer()
        else:
            self.activation = lambda x: x
                     
    def forward(self, x):
        return self.activation(self.norm(self.conv(self.pad(x))))

class TrConv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0,
                    bias=True, dilation=1, norm_layer=None, nl_layer=None):
        super(TrConv2dBlock, self).__init__()
        self.trConv = spectral_norm(nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel_size,
                                    stride=stride,padding=padding, bias=bias, dilation=dilation))
        if norm_layer is not None:
            self.norm = norm_layer(out_dim)
        else:
            self.norm = lambda x: x
        
        if nl_layer is not None:
            self.activation = nl_layer()
        else:
            self.activation = lambda x: x
                     
    def forward(self, x):
        return self.activation(self.norm(self.trConv(x)))

class Upsampling2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, type='Trp', norm_layer=None, nl_layer=None):
        super(Upsampling2dBlock, self).__init__()
        if type=='transpose':
            self.upsample = TrConv2dBlock(in_dim,out_dim,kernel_size=4,stride=2,
                            padding=1,bias=False,norm_layer=norm_layer,nl_layer=nl_layer)
        elif type=='nearest':
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                Conv2dBlock(in_dim,out_dim,kernel_size=3, stride=1, padding=1,
                    pad_type='reflect', bias=False,norm_layer=norm_layer,nl_layer=nl_layer)
                )
        elif type=='pixelshuffle':
            self.upsample = nn.Sequential(
                Conv2dBlock(in_dim,out_dim*4,kernel_size=3, stride=1, padding=1,
                    pad_type='reflect', bias=False,norm_layer=norm_layer,nl_layer=nl_layer),
                nn.PixelShuffle(2)
                )
        else:
            raise NotImplementedError('Upsampling layer [%s] is not found' % type)
    def forward(self, x):
        return self.upsample(x)
 
class ResidualBlock(nn.Module):
    def __init__(self, h_dim, norm_layer=None, nl_layer=None, use_dropout=False):
        super(ResidualBlock, self).__init__()
        block = [conv3x3(h_dim,h_dim,norm_layer=norm_layer,nl_layer=nl_layer),
                 conv3x3(h_dim,h_dim,norm_layer=norm_layer)]
        if use_dropout:
            block.append(nn.Dropout(0.5))
        self.encode = nn.Sequential(*block)

    def forward(self, x):
        y = self.encode(x)
        return x+y
        
class ContentEncoder(nn.Module):
    def __init__(self, input_nc=3, nef=64, n_ds_blocks=2, n_resblocks=4, norm_type='cbin'):
        super(ContentEncoder, self).__init__()
        norm_layer, _ = get_norm_layer(layer_type=norm_type)
        nl_layer = get_nl_layer(layer_type='lrelu')
        block = [Conv2dBlock(input_nc, nef, kernel_size=7, stride=1, padding=3,
                 pad_type='reflect', bias=False, norm_layer=norm_layer, nl_layer=nl_layer)]
        input_nef = nef
        for _ in range(0,n_ds_blocks):
            output_nef = min(input_nef*2,1024)
            block.append(Conv2dBlock(input_nef,output_nef, kernel_size=4, stride=2, padding=1,
                            pad_type='reflect', bias=False, norm_layer=norm_layer, nl_layer=nl_layer))
            input_nef = output_nef
        for _ in range(n_resblocks):
            block.append(ResidualBlock(input_nef, norm_layer=norm_layer,nl_layer=nl_layer))
        self.encode =  nn.Sequential(*block)
        
    def forward(self, x):
        y = self.encode(x)
        return y

class DownResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer=None, nl_layer=None):
        super(DownResidualBlock, self).__init__()
        self.encode = nn.Sequential(
                        norm_layer(in_dim),
                        nl_layer(),
                        conv3x3(in_dim, in_dim,norm_layer=norm_layer,nl_layer=nl_layer),
                        conv3x3(in_dim, out_dim),
                        nn.AvgPool2d(kernel_size=2, stride=2))
        self.shortcut = nn.Sequential(
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Conv2dBlock(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True))
    def forward(self, x):
        y = self.encode(x)
        y = y + self.shortcut(x)
        return y
        
class StyleEncoder(nn.Module):
    def __init__(self, input_nc=3, nef=64, n_style=1, n_blocks=4, norm_type='cbin'):
        super(StyleEncoder, self).__init__()
        norm_layer, _ = get_norm_layer(layer_type=norm_type)
        nl_layer = get_nl_layer(layer_type='lrelu')
        block = [Conv2dBlock(input_nc, nef, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_nef = min(nef*n,256) 
            output_nef = min(nef*(n+1),256) 
            block.append(DownResidualBlock(input_nef, output_nef, norm_layer, nl_layer))
        block += [nl_layer(),nn.AdaptiveAvgPool2d(1)]
        self.encode =  nn.Sequential(*block)
        self.fc = spectral_norm(nn.Linear(output_nef, n_style))
        self.fcVar = spectral_norm(nn.Linear(output_nef, n_style))

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)
        
    def forward(self, x):
        x_conv = self.encode(x).squeeze()
        mu = self.fc(x_conv)
        logvar = self.fcVar(x_conv)
        latent_code = self.reparametrize(mu, logvar)
        return latent_code, mu, logvar

class ConResidualBlock(nn.Module):
    def __init__(self, h_dim, c_norm_layer=None, nl_layer=None,use_dropout=False,return_con=False):
        super(ConResidualBlock, self).__init__()
        self.return_con = return_con
        self.c1 = Conv2dBlock(h_dim,h_dim, kernel_size=3, stride=1, padding=1,pad_type='reflect', bias=False)
        self.n1 = c_norm_layer(h_dim)
        self.l1 = nl_layer()
        self.c2 = Conv2dBlock(h_dim,h_dim, kernel_size=3, stride=1, padding=1, pad_type='reflect', bias=False)
        self.n2 = c_norm_layer(h_dim)
        if use_dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = lambda x: x
    def forward(self, input):
        x, code = input
        y = self.l1(self.n1(self.c1(x),code))
        y = self.n2(self.c2(y),code)
        y = self.dropout(y)
        out = x + y
        if self.return_con:
            out = [out,code]
        return out
        
class Decoder(nn.Module):
    def __init__(self, input_nc=256, output_nc=3, n_latent=128, n_us_blocks=2, n_resblocks=6,
                    norm_type='cbin', up_type='transpose',use_dropout=False):
        super(Decoder, self).__init__()
        norm_layer, c_norm_layer = get_norm_layer(layer_type=norm_type, num_con=n_latent)
        nl_layer = get_nl_layer(layer_type='relu')
        block = []
        for i in range(n_resblocks):
            block.append(ConResidualBlock(input_nc, c_norm_layer=c_norm_layer,
                        nl_layer=nl_layer,use_dropout=use_dropout,return_con=i<(n_resblocks-1)))
        for i in range(n_us_blocks):
            block.append(Upsampling2dBlock(input_nc,input_nc//2,type=up_type,
                            norm_layer=LayerNorm,nl_layer=nl_layer))
            input_nc = input_nc//2
        block +=[Conv2dBlock(input_nc, output_nc, kernel_size=7, stride=1, padding=3,
                    pad_type='reflect', bias=True,nl_layer=nn.Tanh)]
        self.decode = nn.Sequential(*block)
        
    def forward(self, content, code):
        out = self.decode([content,code])
        return out

class ConDownResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, c_norm_layer=None, nl_layer=None, return_con=False):
        super(ConDownResidualBlock, self).__init__()
        self.return_con = return_con
        self.cnorm1 = c_norm_layer(in_dim)
        self.nl1 = nl_layer()
        self.conv1 = conv3x3(in_dim, in_dim)
        self.cnorm2 = c_norm_layer(in_dim)
        self.nl2 = nl_layer()
        self.cmp = nn.Sequential(
                        conv3x3(in_dim, out_dim),
                        nn.AvgPool2d(kernel_size=2, stride=2))
        self.shortcut = nn.Sequential(
                        nn.AvgPool2d(kernel_size=2, stride=2),
                        Conv2dBlock(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, input):
        x, code = input
        out = self.cmp(self.nl2(self.cnorm2(self.conv1(self.nl1(self.cnorm1(x,code))),code)))
        out = out + self.shortcut(x)
        if self.return_con:
            out = [out,code]
        return out
        
class StyleDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_block=3, norm_type='cbin', n_latent=2):
        super(StyleDiscriminator, self).__init__()
        norm_layer, c_norm_layer = get_norm_layer(layer_type=norm_type,num_con=n_latent)
        nl_layer = get_nl_layer(layer_type='lrelu')
        self.head = Conv2dBlock(input_nc, ndf, kernel_size=4,stride=2,padding=1,bias=True)
        dim_in=ndf
        block = []
        for i in range(1, n_block):
            dim_out = min(dim_in*2, 512)
            block += [ConDownResidualBlock(dim_in, dim_out, c_norm_layer=c_norm_layer, nl_layer=nl_layer,return_con=i<(n_block-1))]
            dim_in = dim_out
        block.append(Conv2dBlock(dim_in, 1, kernel_size=1, stride=1, padding=0,bias=True))
        self.encode = nn.Sequential(*block)
    def forward(self, x, code):
        out = self.encode([self.head(x),code])
        return out
        
class MultiStyleDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=32, n_block=3,  norm_type='cbin', n_latent=2):
        super(MultiStyleDiscriminator, self).__init__()
        self.model_1 = StyleDiscriminator(input_nc=input_nc, ndf=ndf, n_block=n_block, norm_type=norm_type,n_latent=n_latent)
        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.model_2 = StyleDiscriminator(input_nc=input_nc, ndf=ndf//2, n_block=n_block, norm_type=norm_type,n_latent=n_latent)
        
    def forward(self, x, code):
        pre1 = self.model_1(x, code)
        pre2 = self.model_2(self.down(x), code)
        return [pre1, pre2]
         