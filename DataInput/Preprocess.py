from os.path import *
from glob import *
import os
import logging
logging.getLogger().setLevel(logging.INFO)
from PIL import Image
# from osgeo import gdal
import numpy as np
from tqdm import tqdm


class Tif_operater(object):
    def __init__(self):
        super(Tif_operater, self).__init__()


    def readfile(self, filename):
        tiffile = gdal.Open(filename)
        assert tiffile, f'cannot load the file {filename}'
        img = tiffile.ReadAsArray(0, 0, tiffile.RasterXSize, tiffile.RasterYSize)
        if len(img.shape)==3:
            img = img.transpose((1, 2, 0))  # HWC
        return img

    def writefile(self, im_data, im_geotrans, im_proj, path):
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        elif len(im_data.shape) == 2:
            im_data = np.array([im_data])
            im_bands, im_height, im_width = im_data.shape
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")  # 创建一个带有名字指代驱动器对象
        dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands),
                                datatype)  # 给出文件的路径，宽度，高度，波段数（倒过来）和数据类型，创建空文件，并确定开辟多大内存，像素值的类型用gdal类型
        # 设置头文件信息：仿射变化参数，投影信息，数据体
        if (dataset != None):
            dataset.SetGeoTransform(im_geotrans)  # driver对象创建的dataset对象具有SetGeoTransform方法，写入仿射变换参数GEOTRANS
            dataset.SetProjection(im_proj)  # SetProjection写入投影im_proj
        for i in range(im_bands):  # 对每个波段进行处理 写入数据体
            dataset.GetRasterBand(i + 1).WriteArray(
                im_data[i])  # i从0开始，因此GetRasterBand(i+1),每个波段里用writearray写入图像信息数组im_data
        del dataset  # 写完之后释放内存空间

class img_Cropper(object):
    def __init__(self, cropsize, overlap=0, padding=0):
        super(img_Cropper, self).__init__()
        self.cropsize = cropsize
        self.overlap = overlap
        self.padding = padding
        self.dataset = []
        self.nameset = []

    def load_file(self, rawpath):

        if 'tif' in os.path.splitext(os.listdir(rawpath)[0])[1]:
            logging.info(f'loadding tif file...')
            tif_operater = Tif_operater()
            file_list = glob(os.path.join(rawpath, '**'))
            logging.info(f'the length of filelist is {len(file_list)}')
            for file in tqdm(file_list):
                img = tif_operater.readfile(filename=file)  # HWC

                if self.padding:
                    if len(img.shape) == 2:
                        height_padding = np.zeros((self.padding, *img.shape[1:]), dtype=np.uint8)
                        weight_padding = np.zeros((img.shape[0]+2 * self.padding, self.padding), dtype=np.uint8)
                    elif len(img.shape) == 3:
                        height_padding = np.zeros((self.padding, *img.shape[1:]), dtype=np.uint8)
                        weight_padding = np.zeros((img.shape[0]+2 * self.padding, self.padding, img.shape[2]), dtype=np.uint8)
                    img = np.concatenate((height_padding, img, height_padding), axis=0)
                    img = np.concatenate((weight_padding, img, weight_padding), axis=1)

                name = os.path.splitext(os.path.basename(file))[0]

                self.dataset.append(img)
                self.nameset.append(name)
        else:
            logging.info(f'loadding png/jpg file...')
            file_list = glob(os.path.join(rawpath, '**'))
            for file in tqdm(file_list):
                img = np.array(Image.open(fp=file))
                if self.padding:
                    height_padding = np.zeros((self.padding, *img.shape[1:]), dtype=np.uint8)
                    if len(img.shape) == 2:
                        weight_padding = np.zeros((img.shape[0]+2 * self.padding, self.padding), dtype=np.uint8)
                    elif len(img.shape) == 3:
                        weight_padding = np.zeros((img.shape[0]+2 * self.padding, self.padding, img.shape[2]), dtype=np.uint8)
                    img = np.concatenate((height_padding, img, height_padding), axis=0)
                    img = np.concatenate((weight_padding, img, weight_padding), axis=1)
                name = os.path.splitext(os.path.basename(file))[0]

                self.dataset.append(img)
                self.nameset.append(name)

        logging.info(f'the img size is {self.dataset[0].shape}')
        logging.info(f'there are {len(self.dataset)} images')

    def crop_an_img(self, img, cropsize, overlap):  # Drop out the rest part
        '''construct a list for crop_img, x_point_index and y_point_index respectively'''
        crop_img = []
        x_point_index = []
        y_point_index = []

        '''filling the x_point_index'''
        total_weight = img.shape[1]
        i_x = 0
        while i_x + cropsize <= total_weight:
            x_point_index.append(i_x)
            i_x += (cropsize-overlap)

        '''filling the y_point_index'''
        total_height = img.shape[0]
        i_y = 0
        while i_y + cropsize <= total_height:
            y_point_index.append(i_y)
            i_y += (cropsize - overlap)

        '''cropping'''
        for y_point in y_point_index:
            for x_point in x_point_index:
                sub_img = img[y_point:y_point+cropsize, x_point:x_point+cropsize]
                crop_img.append(sub_img)

        return crop_img

    def crop_file(self, rawpath):
        self.load_file(rawpath)
        crop_data = []
        name_data = []
        for sub_data, sub_name in zip(self.dataset, self.nameset):
            crop_an_img = self.crop_an_img(sub_data, self.cropsize, self.overlap)
            for i in range(len(crop_an_img)):
                name_data.append(sub_name+f'_{i}.png')
            crop_data += crop_an_img
        return crop_data, name_data

    def save_crop_file(self, rawpath, savepath):
        crop_data, name_data = self.crop_file(rawpath)

        for image, name in tqdm(zip(crop_data, name_data)):
            img = Image.fromarray(image)
            img.save(fp=os.path.join(savepath, name))

if __name__ == "__main__":
    rawpath = r'F:\dataset\M_road\train_no_use\label'
    savepath = r'F:\dataset\M_road\train\label'
    cropsize = 256
    padding = 0
    overlap = 0
    img_Cropper = img_Cropper(cropsize=cropsize, padding=padding, overlap=overlap)
    img_Cropper.save_crop_file(rawpath=rawpath, savepath=savepath)