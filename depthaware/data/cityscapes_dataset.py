import os.path
from depthaware.data.base_dataset import *
from PIL import Image
import time


#label_weight = [0.005770029194127712, 0.012971614093310078, 0.03362765598112945, 0.1221253676849356, 0.06859890961300749, 0.15823995906267385, 0.09602253559800432, 0.12810205801896177, 0.1718342979655409, 0.2830090542974214, 0.06808788822945917, 0.28288925581409397, 0.30927228790865696, 0.6046432911319981, 0.7276073719428268, 0.6584037740058684, 1.6161287361233052, 0.4147706187681264, 0.8706942889933341, 0.8146644289372541, 0.8744887302745185, 0.25134887482271207, 0.3527236656093415, 1.9965490899244573, 3.453731279765878, 0.603116521402235, 1.6573996378194742, 21.603576890926714, 1.3738455233450662, 11.13489209800063, 7.110616094064334, 3.5123361407056404, 8.061760999036036, 1.5451820155073996, 0.9412019674579293, 9.351917523626016, 0.8485119225668366, 0.09619406694759904, 0.07387533823120886, 0.019189673545819297]

def make_dataset_frombasedir(basedir):
    """
    Cityscapes format:
    All data in similar directory structure with common base directory
    """

    # 'troisdorf_000000_000073' is corrupted
    images = [x for x in recursive_glob(rootdir=basedir, suffix='.png') if 'troisdorf_000000_000073' not in x]
    segs = []
    depths = []
    HHAs = []
    for x in images:
        img_dir, img_path = os.path.split(x)
        segs.append(os.path.join(img_dir.replace('leftImg8bit', 'gtCoarse'), img_path.replace('leftImg8bit', 'labelIds')))
        depths.append(x.replace('leftImg8bit', 'disparity'))
        HHAs.append(x.replace('leftImg8bit', 'HHA'))

    return {'images':images, 'segs':segs, 'HHAs':HHAs, 'depths':depths}

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]

class CityscapesDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        np.random.seed(int(time.time()))
        self.paths_dict = make_dataset_frombasedir(opt.list)
        self.len = len(self.paths_dict['images'])

    def __getitem__(self, index):
        #self.paths['images'][index]
        # print self.opt.scale,self.opt.flip,self.opt.crop,self.opt.colorjitter
        img = np.asarray(Image.open(self.paths_dict['images'][index]))#.astype(np.uint8)
        depth = np.asarray(Image.open(self.paths_dict['depths'][index])).astype(np.float32)/120. # 1/10 * depth

        HHA = np.asarray(Image.open(self.paths_dict['HHAs'][index]))
        seg = np.asarray(Image.open(self.paths_dict['segs'][index])).astype(np.uint8)

        params = get_params(self.opt, seg.shape)
        depth_tensor_tranformed = transform(depth, params, normalize=False,istrain=self.opt.isTrain)
        seg_tensor_tranformed = transform(seg, params, normalize=False,method='nearest',istrain=self.opt.isTrain)
        if self.opt.inputmode == 'bgr-mean':
            img_tensor_tranformed = transform(img, params, normalize=False, istrain=self.opt.isTrain, option=1)
            HHA_tensor_tranformed = transform(HHA, params, normalize=False, istrain=self.opt.isTrain, option=2)
        else:
            img_tensor_tranformed = transform(img, params, istrain=self.opt.isTrain, option=1)
            HHA_tensor_tranformed = transform(HHA, params, istrain=self.opt.isTrain, option=2)

        return {'image':img_tensor_tranformed,
                'depth':depth_tensor_tranformed,
                'seg': seg_tensor_tranformed,
                'HHA': HHA_tensor_tranformed,
                'imgpath': self.paths_dict['segs'][index]}

    def __len__(self):
        return self.len

    def name(self):
        return 'CityscapesDataset'

class CityscapesDataset_val(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        np.random.seed(8964)
        self.paths_dict = make_dataset_frombasedir(opt.vallist)
        self.len = len(self.paths_dict['images'])

    def __getitem__(self, index):
        #self.paths['images'][index]
        img = np.asarray(Image.open(self.paths_dict['images'][index]))#.astype(np.uint8)
        # print (img)
        depth = np.asarray(Image.open(self.paths_dict['depths'][index])).astype(np.float32)/120. # 1/5 * depth
        HHA = np.asarray(Image.open(self.paths_dict['HHAs'][index]))
        seg = np.asarray(Image.open(self.paths_dict['segs'][index])).astype(np.uint8)

        params = get_params(self.opt, seg.shape, test=True)
        depth_tensor_tranformed = transform(depth, params, normalize=False,istrain=self.opt.isTrain)
        seg_tensor_tranformed = transform(seg, params, normalize=False,method='nearest',istrain=self.opt.isTrain)
        # HHA_tensor_tranformed = transform(HHA, params,istrain=self.opt.isTrain)
        if self.opt.inputmode == 'bgr-mean':
            img_tensor_tranformed = transform(img, params, normalize=False, istrain=self.opt.isTrain, option=1)
            HHA_tensor_tranformed = transform(HHA, params, normalize=False, istrain=self.opt.isTrain, option=2)
        else:
            img_tensor_tranformed = transform(img, params, istrain=self.opt.isTrain, option=1)
            HHA_tensor_tranformed = transform(HHA, params, istrain=self.opt.isTrain, option=2)

        return {'image':img_tensor_tranformed,
                'depth':depth_tensor_tranformed,
                'seg': seg_tensor_tranformed,
                'HHA': HHA_tensor_tranformed,
                'imgpath': self.paths_dict['segs'][index]}

    def __len__(self):
        return self.len

    def name(self):
        return 'CityscapesDataset'


if __name__ == '__main__':
    dataset = make_dataset_frombasedir('datasets/cityscapes/leftImg8bit/')
    print("Images: {}".format(len(dataset['images'])))