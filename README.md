### Depth-aware CNN for RGB-D Segmentation [<a href="https://arxiv.org/pdf/1803.06791.pdf">Arxiv</a>]

This fork is compatible with pytorch version 1.5.1 CUDA is required.

### Installation

#### Dependancies:
 
 - <a href="http://pytorch.org/">Pytorch</a>, 
 - <a href="https://github.com/Knio/dominate">dominate</a>, 
 - <a href="https://github.com/lanpa/tensorboard-pytorch">TensorBoardX</a> 
 - tqdm
 - scipy

Code snippet creates new conda environment and installs dependancies

```bash
conda create -n depthcnn python=3.8
conda activate depthcnn
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install tqdm scipy
pip install tensorboardX
pip install dominate
pip install opencv-python
```

#### Compile CUDA code

The depth-aware convolution and depth-aware average pooling operations are under folder `models/ops/`, to build them, simply use `python setup.py install` to compile.

```bash
cd models/ops/depthconv/
python setup.py install

cd ../depthavgpooling/
pythono setup.py install

cd ../../..
```

#### Finally, install the whole module

```bash
pip install -e .
```

### Training

```bash
#!./scripts/train.sh
python train.py \
--name nyuv2_VGGdeeplab_depthconv \
--dataset_mode nyuv2 \
--flip --scale --crop --colorjitter \
--depthconv \
--list dataset/lists/nyuv2/train.lst \
--vallist dataset/lists/nyuv2/val.lst
```

### Testing 

```bash
#!./scripts/test.sh
python test.py \
--name nyuv2_VGGdeeplab_depthconv \
--dataset_mode nyuv2 \
--list dataset/lists/nyuv2/test.lst \
--depthconv \
--how_many 0
```

### Citation
If you find this work useful, please consider citing:

        @inproceedings{wang2018depthconv,
            title={Depth-aware CNN for RGB-D Segmentation},
            author={Wang, Weiyue and Neumann, Ulrich},
            booktitle={ECCV},
            year={2018}
        }

    
### Acknowledgemets

The visulization code is borrowed from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
[Here](https://github.com/laughtervv/Deeplab-Pytorch) is a pytorch implementation of [DeepLab](http://liangchiehchen.com/projects/DeepLab.html).
