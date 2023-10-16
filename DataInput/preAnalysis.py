from torch.utils.data import Dataset
from os.path import *
from glob import *
import os
import logging
logging.getLogger().setLevel(logging.INFO)
from PIL import Image
import cv2 as cv
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

class preAnalysis(object):
    def __init__(self,
                 root):

        self.root = root
        self.data_dir = ['image', 'label']
        self.mode_dir = ['train']

    def forward(self):
        num_class = np.array([0, 0])
        for mode in self.mode_dir:
            datapath = os.path.join(self.root, mode, 'label')

            num_class = self.pre_analysis(datapath) + num_class
        return num_class

    def pre_analysis(self, datapath):
        num_class = np.array([0, 0])
        for file in tqdm(glob(datapath + '\\**')):
            file = os.path.abspath(file)
            img = cv.imread(file)

            num_class = num_class + [np.bincount(img.flatten())[0], np.prod(img.shape)-np.bincount(img.flatten())[0]]
        return num_class

class scale_to_255(object):
    def __init__(self, raw_root, tar_root):

        super(scale_to_255, self).__init__()
        self.raw_root = raw_root
        self.tar_root = tar_root
        self.data_dir = ['image', 'label']
        self.mode_dir = ['train', 'val', 'test']

    def forward(self):
        for mode in self.mode_dir:
            filelist = glob(os.path.join(self.raw_root, mode, 'label', '**'))
            logging.info(f'find {len(filelist)} labels!')
            for file in filelist:
                img = Image.open(file)
                table = [255]*256
                table[0] = 0
                if np.array(img).max() < 255:
                    img = img.point(table, mode='L')
                print('-------------------')
                print(np.array(img).max())
                print('-------------------')
                img.save(os.path.join(self.tar_root, mode, 'label', os.path.basename(file)))

class data_cleaner(object):
    def __init__(self, p=0, rootpath = None):
        super(data_cleaner, self).__init__()
        self.name_list = []
        self.p = p
        self.root_path = os.getcwd()
        if rootpath:
            self.root_path = rootpath
            self.image_root_path = os.path.join(rootpath, 'train', 'image')
            self.label_root_path = os.path.join(rootpath, 'train', 'label')

    def load_file(self, rootpath):
        logging.info(f'loadding for {rootpath}...')
        self.root_path = rootpath
        self.image_root_path = os.path.join(rootpath, 'train', 'image')
        self.label_root_path = os.path.join(rootpath, 'train', 'label')

        img_file_list = glob(os.path.join(self.image_root_path, '**'))
        for img_file in img_file_list:
            self.name_list.append(os.path.basename(img_file))
        logging.info(f'selected {len(self.name_list)} images!')

    def class_anlysis_a_label(self, label):
        num_background = np.bincount(label.flatten())[0]
        num_foreground = np.prod(label.shape) - num_background
        return num_foreground/num_background

    def an_image_anlysis(self, img):
        num_nullvalue = np.bincount(img.flatten())[-1]
        num_total = np.prod(img.shape)
        return num_nullvalue/num_total

    def file_class_analysis(self, rootpath):
        to_be_cleaned = []
        analysis_static = [0]*6
        name_group_list = [[], [], [], [], [], []]
        analysis_image = [0]*2
        self.load_file(rootpath=rootpath)
        for name in tqdm(self.name_list):
            label = np.array(Image.open(os.path.join(self.label_root_path, name)))
            # image = np.array(Image.open(os.path.join(self.image_root_path, name)))
            fore_rate = self.class_anlysis_a_label(label)
            # null_rate = self.an_image_anlysis(image)

            if 0<=fore_rate<0.01:
                analysis_static[0] += 1
                name_group_list[0].append(name)
            if 0.01<=fore_rate<0.03:
                analysis_static[1] += 1
                name_group_list[1].append(name)
            if 0.03<=fore_rate<0.05:
                analysis_static[2] += 1
                name_group_list[2].append(name)
            if 0.05<=fore_rate<0.07:
                analysis_static[3] += 1
                name_group_list[3].append(name)
            if 0.07<=fore_rate<0.1:
                analysis_static[4] += 1
                name_group_list[4].append(name)
            if 0.1<=fore_rate<1:
                analysis_static[5] += 1
                name_group_list[5].append(name)


            # if null_rate < 0.5:
            #     analysis_image[0] += 1
            # if null_rate >= 0.5:
            #     analysis_image[1] += 1
            null_rate = 0

        #     if (fore_rate <= self.p) or (null_rate >= 0.1):
        #         to_be_cleaned.append(name)
        # logging.info(f'there are {len(to_be_cleaned)} files to be cleaned.')

        # '''saving the cleaned file'''
        # txt_saver = open(file=os.path.join(self.root_path, 'cleaned_data_futher.txt'), mode='w')
        # cleaned_data = [r'{}{}'.format(data, '\n') for data in to_be_cleaned]
        # txt_saver.writelines(cleaned_data)

        txt_saver = open(file=os.path.join(self.root_path, 'data_analysis.txt'), mode='r+')
        map = {0:'0~0.01',1:'0.01~0.03', 2:'0.03~0.05', 3:'0.05~0.07', 4:'0.07~0.1', 5:'0.1~1'}
        for i, name_list in enumerate(name_group_list):
            cleaned_data = [r'{}{}'.format(data, '\n') for data in name_list]
            txt_saver.read()
            txt_saver.write(r'{}{}'.format(map[i], '\n'))
            txt_saver.writelines(cleaned_data)

        '''visualizing'''
        group1 = ['0~0.01', '0.01~0.03', '0.03~0.05', '0.05~0.07', '0.07~0.1', '0.1~1']
        # group2 = ['null_rate<0.5', 'null_rate>=0.5']
        # plt.subplot(121)
        print(analysis_static)
        plt.bar(group1, analysis_static, width=0.8)
        # plt.subplot(122)
        # plt.bar(group2, analysis_image, width=0.8)
        plt.show()
        return to_be_cleaned

    def clean_data_with_p(self, rootpath, data_given=None):
        if data_given:
            to_be_cleaned = data_given
        else:
            to_be_cleaned = self.file_class_analysis(rootpath)

        for name in to_be_cleaned:
            os.remove(os.path.join(self.image_root_path, name))
            os.remove(os.path.join(self.label_root_path, name))

    def sample_data(self, data_given):
        txt_saver = open(file=os.path.join(self.root_path, 'cleaned_data_256.txt'), mode='r+')
        # 0~0.01
        cleaned_data = data_given[1:4018]
        txt_saver.read()
        txt_saver.writelines(cleaned_data)
        # 0.01~0.03
        cleaned_data_2 = data_given[4019:8300]
        cleaned_data_2 = random.sample(cleaned_data_2, 1731)
        txt_saver.writelines(cleaned_data_2)
        # 0.03~0.05
        cleaned_data_3 = data_given[8301:12682]
        cleaned_data_3 = random.sample(cleaned_data_3, 2161)
        txt_saver.writelines(cleaned_data_3)
        # 0.05~0.07
        cleaned_data_4 = data_given[12683:16656]
        cleaned_data_4 = random.sample(cleaned_data_4, 2473)
        txt_saver.writelines(cleaned_data_4)
        # 0.07~0.1
        cleaned_data_5 = data_given[16657:20962]
        cleaned_data_5 = random.sample(cleaned_data_5, 2655)
        txt_saver.writelines(cleaned_data_5)
        # 0.1~1
        cleaned_data_6 =  data_given[20963:27424]
        cleaned_data_6 = random.sample(cleaned_data_6, 4381)
        txt_saver.writelines(cleaned_data_6)




if __name__ == '__main__':
    root = r'F:\dataset\M_road'
    num_class = preAnalysis(root=root).forward()
    print(num_class)
    # raw_root = r'F:\dataset\OneDrive-2021-09-03\Road_UAV_dataset'
    # tar_root = r'F:\dataset\road_data'
    # scaler = scale_to_255(raw_root=raw_root, tar_root=tar_root)
    # scaler.forward()


    rootpath = r"F:\dataset\M_road"
    data_cleaner = data_cleaner(p=0.05, rootpath=rootpath)
    data_cleaner.file_class_analysis(rootpath)
    # data_given = open(file=r'F:\dataset\M_road\cleaned_data_futher.txt', mode='r')
    # data_given = [data.strip('\n') for data in data_given.readlines()]
    # data_cleaner.clean_data_with_p(rootpath=rootpath, data_given=data_given)
    # data_cleaner.sample_data(data_given)
