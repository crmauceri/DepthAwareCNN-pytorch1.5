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

DEBUG = False
if DEBUG:
    np.random.seed(1)
    torch.manual_seed(0)
else:
    np.random.seed(int(time.time()))

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
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    for epoch in range(start_epoch, opt.nepochs):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size

        model.model.train()
        for i, data in tqdm(enumerate(dataset), desc="Epoch {}".format(epoch)):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            ############## Forward and Backward Pass ######################
            try:
                # print("\nMemory check forward: {}, {}".format(torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()))
                #print("Image size: {}".format(data['image'].shape))
                model.forward(data)
            except RuntimeError as e:
                print("Error on forward iteration {} : {}".format(i, e))
                exit()

            try:
                # print("Memory check backward: {}, {}".format(torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()))
                model.backward(total_steps, opt.nepochs * dataset.__len__() * opt.batchSize + 1)
            except RuntimeError as e:
                print("Error on backward iteration {} : {}".format(i, e))
                exit()

            ############## update tensorboard and web images ######################
            if total_steps % opt.display_freq == 0:
                visuals = model.get_visuals(total_steps)
                visualizer.display_current_results(visuals, epoch, total_steps)

            ############## Save latest Model   ######################
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            # print time.time()-iter_start_time

        # end of epoch
        model.model.eval()
        if dataset_val!=None:
            label_trues, label_preds = [], []
            for i, data in enumerate(dataset_val):
                seggt, segpred = model.forward(data,False)
                seggt = seggt.data.cpu().numpy()
                segpred = segpred.data.cpu().numpy()

                label_trues.append(seggt)
                label_preds.append(segpred)

            metrics = util.label_accuracy_score(
                label_trues, label_preds, n_class=opt.label_nc)
            metrics = np.array(metrics)
            metrics *= 100
            print('''\
                    Validation:
                    Accuracy: {0}
                    Accuracy Class: {1}
                    Mean IU: {2}
                    FWAV Accuracy: {3}'''.format(*metrics))
            model.update_tensorboard(metrics,total_steps)
        iter_end_time = time.time()

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch+1, opt.nepochs, time.time() - epoch_start_time))
        if metrics[2]>best_iou:
            best_iou = metrics[2]
            print('saving the model at the end of epoch %d, iters %d, loss %f' % (epoch, total_steps, model.trainingavgloss))
            model.save('best')

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d, loss %f' % (epoch, total_steps, model.trainingavgloss))
            model.save('latest')
            model.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

