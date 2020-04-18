"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import matplotlib.pyplot as plt

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    epoch = 0
    G_real = []
    G_fake = []
    D_real = []
    D_fake = []
    D_gp = []
    G = []
    D = [] 
    losses_D = []
    losses_G = []
    epoch_list = []
    #for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    while total_iters <  opt.total_num_giters:
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()

            model.set_input(data)          # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            epoch_iter += 1 
            total_iters += 1 

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), int(total_iters/opt.display_freq), opt.display_freq, save_result)

#             if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
#                 losses = model.get_current_losses()
#                 t_comp = (time.time() - iter_start_time) / opt.batch_size
#                 visualizer.print_current_losses(epoch, total_iters, losses, t_comp, t_data)
#                 if opt.display_id > 0:
#                     visualizer.plot_current_losses(epoch, float(total_iters) / dataset_size, losses)

            if total_iters % opt.score_freq == 0:    # print generation scores and save logging information to the disk 
                scores = model.get_current_scores()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_scores(epoch, total_iters, scores)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(total_iters) / dataset_size, scores)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

            if total_iters % opt.save_giters_freq == 0: # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(total_iters)
               
            if total_iters % opt.print_freq == 0:
                loss = model.get_current_losses()
                loss_list = list(loss.items())
    #             print(loss_list)
                G_real.append(loss_list[0][1])
                G_fake.append(loss_list[1][1])
                D_real.append(loss_list[2][1])
                D_fake.append(loss_list[3][1])
                D_gp.append(loss_list[4][1])
                
                losses_G.append(loss_list[5][1])
                losses_D.append(loss_list[6][1])
                epoch_list.append(epoch)

        epoch += 1
        print('(epoch_%d) End of giters %d / %d \t Time Taken: %d sec' % (epoch, total_iters, opt.total_num_giters, time.time() - epoch_start_time))

        #print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        #model.update_learning_rate()                     # update learning rates at the end of every epoch.
    
    
    with open('./DG_losses.txt', 'w') as the_file:
        the_file.write(str(losses_D) + '\n')
        the_file.write(str(losses_G) + '\n')
        
    with open('./G_losses.txt', 'w') as the_file:
        the_file.write(str(G_real) + '\n')
        the_file.write(str(G_fake) + '\n')
        
    with open('./D_losses.txt', 'w') as the_file:
        the_file.write(str(D_real) + '\n')
        the_file.write(str(D_fake) + '\n')
        the_file.write(str(D_gp) + '\n')
    print('epoch_list',epoch_list)
    print('losses_D',losses_D)
    print('losses_G',losses_G)
    
    plt.xticks(epoch_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epoch_list, G_real, label = "G_real")
    plt.plot(epoch_list, G_fake, label = "G_fake")
    plt.legend(loc = "upper right")
    path = "./G_losses.png"
    plt.savefig(path)
    plt.show()
    
    plt.xticks(epoch_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epoch_list, D_real, label = "D_real")
    plt.plot(epoch_list, D_fake, label = "D_fake")
    plt.plot(epoch_list, D_gp, label = "D_gp")
    plt.legend(loc = "upper right")
    path = "./D_losses.png"
    plt.savefig(path)
    plt.show()


    plt.xticks(epoch_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epoch_list, losses_D, label = "losses_D")
    plt.plot(epoch_list, losses_G, label = "losses_G")
    plt.legend(loc = "upper right")
    path = "./losses.png"
    plt.savefig(path)
    plt.show()

