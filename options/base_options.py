import argparse
import os
from util import util
import pickle
import dateutil.tz

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        ### data configurations 
        self.parser.add_argument('--dataroot', default='./datasets/summer2winter_yosemite', help='path to images')
        self.parser.add_argument('--n_threads', default=4, type=int, help='# sthreads for loading data')
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fine_size', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--n_attribute', type=int, default=2, help='# of attribute dimensions')
        self.parser.add_argument('--n_style', type=int, default=8, help='# of style dimensions')
        self.parser.add_argument('--is_flip', action='store_true', help='flip the images for data argumentation')
        ### network parameters
        self.parser.add_argument('--up_type', type=str, default='transpose', help='transpose, nearest, or pixelshuffle')
        self.parser.add_argument('--norm_type', type=str, default='cbin', help='normalization:[cbin, cbbn, adain]')
        self.parser.add_argument('--nef', type=int, default=64, help='# of encoder filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discriminator filters in first conv layer')
        self.parser.add_argument('--n_style_blocks', type=int, default=4, help='# of style encoder blocks')
        self.parser.add_argument('--n_content_resblocks', type=int, default=4, help='# of content encoder resblocks')
        self.parser.add_argument('--n_dec_resblocks', type=int, default=6, help='# of decoder resblocks')
        self.parser.add_argument('--n_content_disblocks', type=int, default=5, help='# of content discriminator blocks')
        self.parser.add_argument('--n_image_disblocks', type=int, default=3, help='# of image discriminator blocks')
        self.parser.add_argument('--n_ds_blocks', type=int, default=2, help='# of image downsampling blocks')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout in training time')
        self.parser.add_argument('--use_same_dis', action='store_true', help='share the weights of image discriminator')
        ### model environment
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--exp_name', type=str, default='DMIT_Season_Transfer', help='the storing name in checkpoints')
        self.parser.add_argument('--model_name', type=str, default='season_transfer', help='default model name:[season_transfer, semantic_image_synthesis]')
        self.parser.add_argument('--gpu', type=int, default=-1, help='assign gpu for the model, -1 means cpu mode')
        
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.is_train = self.is_train   # train or test
        
        args = vars(self.opt)
    
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k).encode('utf-8'), str(v).encode('utf-8')))
        print('-------------- End ----------------')
        
        self.opt.expr_dir = os.path.join(self.opt.checkpoints_dir,self.opt.exp_name)
        self.opt.model_dir = os.path.join(self.opt.expr_dir,'model')
        if self.is_train:
            if not self.opt.continue_train:
                self.opt.model_dir = os.path.join(self.opt.expr_dir,'model')
                util.mkdirs(self.opt.model_dir)
                pkl_file = os.path.join(self.opt.expr_dir, 'opt.pkl')
                pickle.dump(self.opt, open(pkl_file, 'wb'))
                # save to the disk
                file_name = os.path.join(self.opt.expr_dir, 'opt_train.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k).encode('utf-8'), str(v).encode('utf-8')))
                    opt_file.write('-------------- End ----------------\n')
            else:
                file_name = os.path.join(self.opt.expr_dir, 'opt_train_continue.txt')
                with open(file_name, 'wt') as opt_file:
                    opt_file.write('------------ Options -------------\n')
                    for k, v in sorted(args.items()):
                        opt_file.write('%s: %s\n' % (str(k).encode('utf-8'), str(v).encode('utf-8')))
                    opt_file.write('-------------- End ----------------\n')
        else:
            if self.opt.results_dir=='':
                self.opt.results_dir = os.path.join(self.opt.expr_dir,'results')
            results_dir = self.opt.results_dir
            util.mkdirs(results_dir)
            self.opt.model_dir = os.path.join(self.opt.expr_dir,'model')
        return self.opt
