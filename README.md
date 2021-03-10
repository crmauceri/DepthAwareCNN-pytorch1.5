### Depth-aware CNN for RGB-D Segmentation [<a href="https://arxiv.org/pdf/1803.06791.pdf">Arxiv</a>]

This fork is compatible with PyTorch version 1.5.1. CUDA is required.

PyTorch 1.0 has a breaking change for [how CUDA extensions are added](https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-mixed-c-cuda-extension). I reimplemented the whole CUDA kernel to make it work with 1.5.1 therefore there is no one-to-one corespondance with the functions in the original [PyTorch 0.4 implementation](https://github.com/laughtervv/DepthAwareCNN/tree/master/models/ops/depthconv/src). 

I have added [unit tests](https://github.com/crmauceri/DepthAwareCNN-pytorch1.5/blob/master/depthaware/unit_tests) to verify that my implementation calculates the correct gradients. The unit tests comparing to PyTorch's default implementations of convolutions and average pooling to my Depth Aware implemenations. By using an input with the depth channel set to all ones, both implemenations should produce the same result. **depthconv_unit_tests.py does not pass** Due to precision errors, the weight gradients in the final layers of the VGG 16 network are only the same to 3 decimal places. I consider this to be an acceptable difference. 

### Installation

#### Dependancies:
 
 - <a href="http://pytorch.org/">Pytorch</a>, 
 - <a href="https://github.com/Knio/dominate">dominate</a>, 
 - <a href="https://github.com/lanpa/tensorboard-pytorch">TensorBoardX</a> 
 - tqdm
 - scipy
 - opencv-python

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
cd depthaware/models/ops/depthconv/
python setup.py install

cd ../depthavgpooling/
python setup.py install

cd ../../../../
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
If you find this work useful, please consider citing the original paper:

        @inproceedings{wang2018depthconv,
            title={Depth-aware CNN for RGB-D Segmentation},
            author={Wang, Weiyue and Neumann, Ulrich},
            booktitle={ECCV},
            year={2018}
        }
