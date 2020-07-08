from depthaware.options.train_options import TrainOptions
from depthaware.data.data_loader import CreateDataLoader
from depthaware.models.models import create_model
from depthaware import utils as util
from depthaware.utils.visualizer import Visualizer
from tqdm import tqdm
import os
import numpy as np
import time
import torch
import cProfile

def train_loop(opt, model, dataset):
    total_steps = 0
    model.model.train()
    for i, data in tqdm(enumerate(dataset)):
        iter_start_time = time.time()
        total_steps += opt.batchSize

        ############## Forward and Backward Pass ######################
        model.forward(data)
        model.backward(total_steps, opt.nepochs * dataset.__len__() * opt.batchSize + 1)

        # print time.time()-iter_start_time
        break


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    opt = TrainOptions().parse()
    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    ioupath_path = os.path.join(opt.checkpoints_dir, opt.name, 'MIoU.txt')
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0

        try:
            best_iou = np.loadtxt(ioupath_path, dtype=float)
        except:
            best_iou = 0.
        print('Resuming from epoch %d at iteration %d, previous best IoU %f' % (start_epoch, epoch_iter, best_iou))
    else:
        start_epoch, epoch_iter = 1, 0
        best_iou = 0.

    data_loader = CreateDataLoader(opt)
    dataset, dataset_val = data_loader.load_data()
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    model = create_model(opt, dataset.dataset)
    # print (model)
    visualizer = Visualizer(opt)

    cProfile.run('train_loop(opt, model, dataset)')


