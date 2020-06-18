python depthaware/train.py \
--name cityscapes_VGGdeeplab_depthconv \
--dataset_mode cityscapes \
--flip --scale --crop --colorjitter \
--depthconv \
--list ./dataset/cityscapes/leftImg8bit/train_extra/ \
--vallist ./dataset/cityscapes/leftImg8bit/val/ \
--continue_train
