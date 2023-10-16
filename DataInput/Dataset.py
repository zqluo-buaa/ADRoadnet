import torch
from torch.utils.data import Dataset
from os.path import *
from glob import *
import os
import logging
from PIL import Image
import numpy as np
from torchvision import transforms
from Utils.modules import RandomMaskingGenerator
from einops import rearrange, reduce, repeat
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2 as cv
import random


class GaussianBlur(torch.nn.Module):

    def __init__(self, kernel_size):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, img):
        image = np.array(img)
        image_blur = cv.GaussianBlur(image, self.kernel_size, 0)
        return Image.fromarray(image_blur)


class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))  # 2020 07 26 or --> and
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct / 2., noise_pct / 2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255  # 盐噪声
            img_[mask == 2] = 0  # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img


class Gaussian_noise(object):
    """增加高斯噪声
    此函数用将产生的高斯噪声加到图片上
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
    """

    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        # 将图片灰度标准化
        img_ = np.array(img).copy()
        img_ = img_ / 255.0
        # 产生高斯 noise
        noise = np.random.normal(self.mean, self.sigma, img_.shape)
        # 将噪声和图片叠加
        gaussian_out = img_ + noise
        # 将超过 1 的置 1，低于 0 的置 0
        gaussian_out = np.clip(gaussian_out, 0, 1)
        # 将图片灰度范围的恢复为 0-255
        gaussian_out = np.uint8(gaussian_out * 255)
        # 将噪声范围搞为 0-255
        # noise = np.uint8(noise*255)
        return Image.fromarray(gaussian_out).convert('RGB')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed((seed))
    # torch.backends.cudnn.deterministics = True

class BasicDataset(Dataset):
    '''this class can be used for constructing a basic class to input data consisting of image and label.
    It consists of three methods:
    __init__: where the files root path of images and labels are appointed, and defining other parameters such as transform, scale, image_set
    __len__: where the length of this dataset is defined
    __getitem__: where we construct a generator for the map from image to label'''
    def __init__(self,
                 root,
                 image_set='train',
                 scale=1,
                 transform=False):
        self.root = root
        self.scale = scale
        self.transform = transform
        self.image_set = image_set
        data_dir = ['image', 'label', 'o_label', 'edge_label']
        mode_dir = ['train', 'val', 'test', 's_train']  # train为多分支 s_train为单分支

        assert 0<scale<=1, "Scale must be between 0 and 1"
        assert (image_set in mode_dir), f"image_set must be set as train or val or test but {image_set}"

        setup_seed(42)

        if image_set == 's_train':
            mode = 'train'
        else:
            mode = image_set
        mode_dir_path = join(root, mode)

        self.images = glob(join(mode_dir_path, data_dir[0], '**'))  # return a list of files with its extension and abspath
        self.labels = glob(join(mode_dir_path, data_dir[1], '**'))
        if self.image_set=='train':
            self.o_labels = glob(join(mode_dir_path, data_dir[2], '**'))
            self.edge_labels = glob(join(mode_dir_path, data_dir[3], '**'))

        '''
        self.sat =  glob(join(mode_dir_path, data_dir[2], '**'))
        '''

        # self.images = os.path.abspath(os.listdir(join(mode_dir_path, data_dir[0])))  # return a list of files with its extension and abspath
        # self.labels = os.path.abspath(os.listdir(join(mode_dir_path, data_dir[1])))

        logging.info(f"the number of {mode} examples is {len(self.images)}")

        # print(self.images)

        assert len(self.images) == len(self.labels), \
            f'in {mode} mode, the number of image and label examples should be equal, but {len(self.images)} and {len(self.labels)}'

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        '''
        :param self: 
        :param index(Int): Index 
        :return: dict:{image [Tensor], label [Tensor]} shape = CHW and the image has been normalized as 0~1  
        '''
        image = Image.open(self.images[index]).convert('RGB')
        label = Image.open(self.labels[index]).convert('L')  # read labels as grayscale mode
        if self.image_set=='train':
            o_labels = Image.open(self.o_labels[index]).convert('L') # 0-255
            edge_labels = Image.open(self.edge_labels[index]).convert('L')  # 0-255
        # sat = Image.open(self.sat[index]).convert('L')  # read labels as grayscale mode  # 会自动拉伸归一化
        '''
        sat = cv.imread(self.sat[index], 0)
        '''

        # assert np.array(label).max() in (255, 0), f'the label should have been scaled as 0~255 but got {np.array(label).max()}.'

        if (self.image_set == 'train' or self.image_set == 's_train') and self.transform == True:  # horizontal flip and rotate 90 degree
            p = 0.5
        #     if np.random.rand(1) >= p:
        #         image = image.transpose(Image.FLIP_LEFT_RIGHT)
        #         label = label.transpose(Image.FLIP_LEFT_RIGHT)
        #         o_labels = o_labels.transpose(Image.FLIP_LEFT_RIGHT)
        #         dis_labels = dis_labels.transpose(Image.FLIP_LEFT_RIGHT)
        #
        #     if np.random.rand(1) >= p:
        #         image = image.transpose(Image.FLIP_TOP_BOTTOM)
        #         label = label.transpose(Image.FLIP_TOP_BOTTOM)
        #         o_labels = o_labels.transpose(Image.FLIP_TOP_BOTTOM)
        #         dis_labels = dis_labels.transpose(Image.FLIP_TOP_BOTTOM)

            # if np.random.rand(1) >= p:
            #     image = image.rotate(90)
            #     label = label.rotate(90)  # PIL.Image HWC
            #     o_labels = o_labels.rotate(90)
            #     dis_labels = dis_labels.rotate(90)
            if np.random.rand(1) >= 0.5:
                k = np.random.randint(0, 3)
                image = GaussianBlur((2*k+1,2*k+1))(image)
            if np.random.rand(1) >= 0.5:
                image = Gaussian_noise(0, 0.1)(image)
            # elif np.random.rand(1) >= 0.5:
            #     image = AddPepperNoise(0.9, p=0.5)(image)

        '''to tensor and scaled to 0-1'''
        image = transforms.ToTensor()(image)  # tensor CHW
        label = transforms.ToTensor()(label)
        if self.image_set == 'train':
            edge_labels = transforms.ToTensor()(edge_labels)
            o_labels = torch.from_numpy(np.array(o_labels, dtype=np.int64)).unsqueeze(0)  # CHW
            orientation_labels = torch.zeros((37, 750, 750), dtype=torch.float32)
            # print(torch.max(o_labels), torch.min(o_labels))
            orientation_labels.scatter_(dim=0, index=o_labels, src=torch.ones(o_labels.shape, dtype=torch.float32))

        # resize 至相同尺寸
        newsize = (512, 512)
        trans_size = transforms.Resize(newsize)
        image = trans_size(image)
        label =  trans_size(label)
        if self.image_set == 'train':
            orientation_labels = trans_size(orientation_labels)
            edge_labels =trans_size(edge_labels)
            tmp_label = torch.cat([label, edge_labels, orientation_labels], dim=0)

        # shape_label = shape_label.resize(newsize)


            # print(torch.max(dis_labels))
        #
        # label = torch.cat((label, shape_label), dim=0)
        # sat = transforms.ToTensor()(sat)

        # if self.image_set == 'train' and self.transform == True:  # horizontal flip and rotate 90 degree
        #     p = 0.5
        #     if np.random.rand(1) >= p:
        #         temp = torch.cat((image, label), dim=0)
        #         trans_temp = transforms.RandomResizedCrop(512, scale=(0.5, 1.))(temp)
        #         image = trans_temp[:3]
        #         label = trans_temp[-1:]

        if self.image_set == 'train' and self.transform == True:
            tmp = torch.cat([image, tmp_label], dim=0)
            tmp = transforms.RandomAffine(0, translate=(0.5, 0.5), fill=[*[0.5, 0.5, 0.5], *[0] * 38, 1])(tmp)
            if np.random.rand(1) < 0.5:
                scale = np.random.random()*0.7+0.3
                size = int(scale * 512)
                tmp = transforms.Resize((size, size))(tmp)
                padding1 = (512- tmp.shape[-1]) // 2
                padding2 = 512 - tmp.shape[-1] - padding1
                image, tmp_label1, tmp_label2 = tmp[:3], tmp[3:-1], tmp[-1:]
                image = F.pad(image, pad=[padding1, padding2, padding1, padding2], mode= "constant", value = 0.5)
                tmp_label1 = F.pad(tmp_label1, pad=[padding1, padding2, padding1, padding2], mode= "constant", value=0)
                tmp_label2 = F.pad(tmp_label2, pad=[padding1, padding2, padding1, padding2], mode= "constant", value=1)
                tmp_label = torch.cat([tmp_label1, tmp_label2], dim=0)
            else:
                tmp = transforms.RandomResizedCrop(512, scale=(0.3, 1.))(tmp)
                image, tmp_label = tmp[:3], tmp[3:]

        if self.image_set == 's_train' and self.transform == True:
            tmp = torch.cat([image, label], dim=0)
            tmp = transforms.RandomAffine(0, translate=(0.2, 0.2), fill=[*[0.5, 0.5, 0.5], 0])(tmp)
            if np.random.rand(1) < 0.5:
                scale = np.random.random()*0.7+0.3
                size = int(scale * 512)
                tmp = transforms.Resize((size, size))(tmp)
                padding1 = (512- tmp.shape[-1]) // 2
                padding2 = 512 - tmp.shape[-1] - padding1
                image, label = tmp[:3], tmp[-1:]
                image = F.pad(image, pad=[padding1, padding2, padding1, padding2], mode= "constant", value = 0.5)
                label = F.pad(label, pad=[padding1, padding2, padding1, padding2], mode= "constant", value=0)
            else:
                tmp = transforms.RandomResizedCrop(512, scale=(0.3, 1.))(tmp)
                image, label = tmp[:3], tmp[-1:]

        if self.image_set == 'train':
            label, edge_labels, orientation_labels = tmp_label[:1], tmp_label[1:2], tmp_label[2:]

        if (self.image_set == 'train' or self.image_set == 's_train')  and self.transform == True:
            image = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3)(image)
            image = transforms.RandomAdjustSharpness(sharpness_factor=np.random.randint(3), p=0.8)(image)



        # if self.image_set == 'train' and self.transform == True:  # horizontal flip and rotate 90 degree
        #     # temp = torch.cat((image, label), dim=0)
        #
        #     num = int((2.0-0.5)//0.25+1)
        #     scales = np.linspace(2.0, 0.5, num)
        #     np.random.shuffle(scales)
        #     size = int((scales[0]*newsize[0]).tolist())
        #     # trans_temp = transforms.RandomResizedCrop(newsize, scale=(0.5, 1.))(temp)
        #     image = transforms.Resize(size)(image)
        #     tmp_label = transforms.Resize(size)(tmp_label)
        #     # image = transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3),
        #     #                                  value=127.5 / 255)(image)
        #     if image.shape[-1]<newsize[0]:
        #         padding1 = (newsize[0]-image.shape[-1])//2
        #         padding2 = newsize[0]-image.shape[-1]-padding1
        #         # image, label = trans_temp[:3]
        #
        #
        #         image = F.pad(image, pad=(padding1, padding1, padding2, padding2), value=127.5/255)
        #         # image = F.pad(image, pad=(padding1, padding1, padding2, padding2), value=0)
        #         tmp_label1 = F.pad(tmp_label[:-1], pad=(padding1, padding1, padding2, padding2), value=0)
        #         tmp_label2 = F.pad(tmp_label[-1, :].unsqueeze(0), pad=(padding1, padding1, padding2, padding2), value=1)
        #         temp = torch.cat([image, tmp_label1, tmp_label2], dim=0)
        #
        #     else:
        #         temp = torch.cat((image, tmp_label), dim=0)
        #         temp = transforms.RandomCrop(newsize)(temp)
        #     temp = transforms.RandomRotation(180,fill=[*[127.5/255, 127.5/255, 127.5/255], *[0]*38, 1])(temp)
        #     temp = transforms.RandomAffine(0, translate=(0.2, 0.2), shear=[-20, 20, -20, 20],  fill=[*[127.5/255, 127.5/255, 127.5/255], *[0]*38, 1])(temp)
        #     image = temp[:3]
        #     tmp_label = temp[3:]

        # if self.image_set == 'train':
        #     label, dis_labels,  orientation_labels = tmp_label[:1], tmp_label[1:2], tmp_label[2:]

        if self.image_set == 'train':
            return {
                'image': image,
                'label': label,
                'o_label': orientation_labels,
                'edge_label': edge_labels
            }  # transform to tensor as CHW, it will be normalized as 0~1 and 2d Image would also be unsqueezed to 3d
        else:
            return {
                'image': image,
                'label': label,
            }  # transform to tensor as CHW, it will be normalized as 0~1 and 2d Image would also be unsqueezed to 3d


class Self_supervise_Dataset(Dataset):
    '''this class can be used for constructing a basic class to input data consisting of image and label.
    It consists of three methods:
    __init__: where the files root path of images and labels are appointed, and defining other parameters such as transform, scale, image_set
    __len__: where the length of this dataset is defined
    __getitem__: where we construct a generator for the map from image to label'''
    def __init__(self,
                 root,
                 image_set='train',
                 scale=1,
                 transform=False,
                 self_smooth = False,
                 mask_ratio = 0.5):
        self.root = root
        self.scale = scale
        self.transform = transform
        self.image_set = image_set
        self.mask_generator = RandomMaskingGenerator(input_size=32, mask_ratio=mask_ratio)
        self.self_smooth = self_smooth
        data_dir = ['label']
        mode_dir = ['train', 'val', 'test']

        assert 0<scale<=1, "Scale must be between 0 and 1"
        assert (image_set in mode_dir), f"image_set must be set as train or val or test but {image_set}"

        mode = image_set
        mode_dir_path = join(root, mode)

        # self.images = glob(join(mode_dir_path, data_dir[0], '*.png'))  # return a list of files with its extension and abspath
        self.labels = glob(join(mode_dir_path, data_dir[0], '**'))

        # self.images = os.path.abspath(os.listdir(join(mode_dir_path, data_dir[0])))  # return a list of files with its extension and abspath
        # self.labels = os.path.abspath(os.listdir(join(mode_dir_path, data_dir[1])))

        logging.info(f"the number of {mode} examples is {len(self.labels)}")

        # print(self.images)

        # assert len(self.images) == len(self.labels), \
        #     f'in {mode} mode, the number of image and label examples should be equal, but {len(self.images)} and {len(self.labels)}'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        '''
        :param self: 
        :param index(Int): Index 
        :return: dict:{image [Tensor], label [Tensor]} shape = CHW and the image has been normalized as 0~1  
        '''
        # image = Image.open(self.images[index]).convert('RGB')
        label = Image.open(self.labels[index]).convert('L')  # read labels as grayscale mode
        mask = torch.from_numpy(self.mask_generator()).to(torch.bool)


        assert np.array(label).max() in (255, 0), f'the label should have been scaled as 0~255 but got {np.array(label).max()}.'

        '''to tensor and scaled to 0-1'''
        # image = transforms.ToTensor()(image)   # tensor CHW
        label = transforms.ToTensor()(label)
        label_patch = rearrange(label, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=16,
                                 p2=16)
        # make mask
        # print('----------------')
        # print(mask)
        img_mask = torch.ones_like(label_patch).type(torch.float32)
        img_mask = img_mask.masked_fill(mask=mask[:,None], value=torch.tensor(0.))
        img_mask = rearrange(img_mask, 'n (p c) -> n p c', c=1)
        img_mask = rearrange(img_mask, '(h w) (p1 p2) c -> c (h p1) (w p2)', p1=16, p2=16, h=32, w=32)
        image = img_mask * label

        if self.self_smooth:
            image = image.double()
            image = torch.where(image <= 0.5, 0.1, image).double()
            image = torch.where(image > 0.5, 0.9, image).type(torch.float32)

        # if self.image_set == 'train' and self.transform == True:  # horizontal flip and rotate 90 degree
        #     p = 0.5
        #     if np.random.rand(1) >= p:
        #         image = image.transpose(Image.FLIP_LEFT_RIGHT)
        #         label = label.transpose(Image.FLIP_LEFT_RIGHT)
        #     if np.random.rand(1) >= p:
        #         image = image.rotate(90)
        #         label = label.rotate(90)  # PIL.Image HWC
        #     # temp = torch.cat((image, label), dim=0)
        #     # trans_temp = transforms.RandomCrop(128)(temp)
        #     # image = trans_temp[:3]
        #     # label = trans_temp[-1:]

        return {
            'image': image,
            'label': label
        }  # transform to tensor as CHW, it will be normalized as 0~1 and 2d Image would also be unsqueezed to 3d

class SecondDataset(Dataset):
    '''this class can be used for constructing a basic class to input data consisting of image and label.
    It consists of three methods:
    __init__: where the files root path of images and labels are appointed, and defining other parameters such as transform, scale, image_set
    __len__: where the length of this dataset is defined
    __getitem__: where we construct a generator for the map from image to label'''
    def __init__(self,
                 root,
                 image_set='train',
                 scale=1,
                 transform=False):
        self.root = root
        self.scale = scale
        self.transform = transform
        self.image_set = image_set
        data_dir = ['image', 'label']
        mode_dir = ['train', 'val', 'test']

        assert 0<scale<=1, "Scale must be between 0 and 1"
        assert (image_set in mode_dir), f"image_set must be set as train or val or test but {image_set}"

        mode = image_set
        mode_dir_path = join(root, mode)

        self.images = glob(join(mode_dir_path, data_dir[0], '*.png'))  # return a list of files with its extension and abspath
        self.labels = glob(join(mode_dir_path, data_dir[1], '**'))

        # self.images = os.path.abspath(os.listdir(join(mode_dir_path, data_dir[0])))  # return a list of files with its extension and abspath
        # self.labels = os.path.abspath(os.listdir(join(mode_dir_path, data_dir[1])))

        logging.info(f"the number of {mode} examples is {len(self.images)}")

        # print(self.images)

        assert len(self.images) == len(self.labels), \
            f'in {mode} mode, the number of image and label examples should be equal, but {len(self.images)} and {len(self.labels)}'

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        '''
        :param self:
        :param index(Int): Index
        :return: dict:{image [Tensor], label [Tensor]} shape = CHW and the image has been normalized as 0~1
        '''
        image = Image.open(self.images[index]).convert('L')
        label = Image.open(self.labels[index]).convert('L')  # read labels as grayscale mode


        assert np.array(label).max() in (255, 0), f'the label should have been scaled as 0~255 but got {np.array(label).max()}.'



        '''to tensor and scaled to 0-1'''
        image = transforms.ToTensor()(image)    # tensor CHW
        # mask = RandomMaskingGenerator(input_size=32, mask_ratio=0.5)()
        # image_patch = rearrange(image, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=16,
        #                         p2=16)
        # image_patch[torch.tensor(mask).to(torch.bool)] = 0.
        # image = rearrange(image_patch, '(h w) (p1 p2 c) -> c (h p1) (w p2)', p1=16, p2=16, h=32, w=32)

        # image[image>0.5] = 1.
        # image[(image>0.1) == (image <= 0.5)] = 0.5
        # image[image < 0.1] = 0.
        # image = torch.where(image.double()>0.5, 1., image.double()).double()
        # image = torch.where(0.1 < image.double() <= 0.5, 0.5, image.double()).double()
        label = transforms.ToTensor()(label)

        if self.image_set == 'train' and self.transform == True:  # horizontal flip and rotate 90 degree
            p = 0.5
            if np.random.rand(1) >= p:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.rand(1) >= p:
                image = image.rotate(90)
                label = label.rotate(90)  # PIL.Image HWC



        return {
            'image': image,
            'label': label
        }  # transform to tensor as CHW, it will be normalized as 0~1 and 2d Image would also be unsqueezed to 3d
