python depthaware/train.py \
--name cityscapes_VGGdeeplab_depthconv \
--dataset_mode cityscapes \
--flip --scale --crop --colorjitter \
--depthconv \
--dataroot ./datasets/cityscapes/leftImg8bit/train_extra/ \
--vallist ./datasets/cityscapes/leftImg8bit/val/ \
--continue_train
