import os
from options.test_options import TestOptions
from data.dataloader import CreateDataLoader
from util.visualizer import save_images
from util import html, util
from models import create_model
from itertools import islice

def main():    
    opt = TestOptions().parse()
    opt.is_flip = False  
    opt.batchSize = 1
    data_loader = CreateDataLoader(opt)
    model = create_model(opt) 
    web_dir = os.path.join(opt.results_dir, 'test')
    webpage = html.HTML(web_dir, 'task {}'.format(opt.exp_name))

    for i, data in enumerate(islice(data_loader, opt.how_many)):
        print('process input image %3.3d/%3.3d' % (i, opt.how_many))
        results = model.translation(data)
        img_path = 'image%3.3i' % i
        save_images(webpage, results, img_path, None, width=opt.fine_size)
    webpage.save()

if __name__ == '__main__':
    main()