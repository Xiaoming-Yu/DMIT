import time
from options.train_options import TrainOptions
from data.dataloader import CreateDataLoader
from util.visualizer import Visualizer
from models import create_model


def main():
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset_size = len(data_loader) * opt.batch_size
    visualizer = Visualizer(opt)
    model = create_model(opt)    
    start_epoch = model.start_epoch
    total_steps = start_epoch*dataset_size
    for epoch in range(start_epoch+1, opt.niter+opt.niter_decay+1):
        epoch_start_time = time.time()
        model.update_lr()
        save_result = True
        for i, data in enumerate(data_loader):
            iter_start_time = time.time()
            total_steps += opt.batch_size
            epoch_iter = total_steps - dataset_size * (epoch - 1)
            model.prepare_data(data)
            model.update_model()
            if save_result or total_steps % opt.display_freq == 0:
                save_result = save_result or total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, ncols=1, save_result=save_result)
                save_result = False
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)
        print('epoch {} cost dime {}'.format(epoch,time.time()-epoch_start_time))
        model.save_ckpt(epoch)
        model.save_generator('latest')
        if epoch % opt.save_epoch_freq == 0:
            print('saving the generator at the end of epoch {}, iters {}'.format(epoch, total_steps))
            model.save_generator(epoch)
            
        
        
if __name__ == '__main__':
    main()