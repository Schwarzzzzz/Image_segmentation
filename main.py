import argparse
import csv
import math
import sys
import os
import shutil
import time
from PIL import Image
from statistics import mean

from gpu_memory_log import gpu_memory_log
# from models.test_model import mem
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from spatial_information.sobel_operator import edges_detection
from spatial_information.test import pred
from spatial_information.sobel_operator_square import edges_detection_square
from spatial_information.test_square import pred_square
from util import util


from combine import combine
from cutBorder import cutBorder
from makeBorder import makeBorder
from segmentation import segmentation

import time

# parser = argparse.ArgumentParser()
# parser.add_argument('-p', '--imgpath', default='in.jpg', type=str)
# IMGpath = parser.parse_args().imgpath


# 启动CycleGAN
# os.system('python test.py --dataroot run\segmentation_out --name test --num_test 9999 --no_dropout')
try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0  # test code only supports num_threads = 0
opt.batch_size = 1  # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
opt.preprocess = 'scale_width'
opt.model = 'test'
# name_list = ['cyclegan_K50', 'cyclegan_K1600']

# 以数据集方式导入Spatial information
class dataset(Dataset):

    def __init__(self):
        super().__init__()
        data = edges_detection(img_name + '\segmentation')
        self.X = torch.FloatTensor(np.array(data.iloc[:, [0, 1, 2, 3]]))
        self.len = len(self.X)

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.len

if __name__ == "__main__":
    # 确定结果文件夹序号
    folder = 1
    while os.path.exists('run' + str(folder)):
        folder = folder + 1

    IMGSpath = opt.s
    IMGs = os.listdir(IMGSpath)
    out_img_split_row = opt.m
    out_img_split_col = opt.n
    resized_height = int(opt.r)

    start = time.time()
    time_count = 0
    time_sum = []

    fenkuai_list = []
    zhuanhuan_list = []
    spatial_time_list = []
    pred_time_list = []
    tensor2img_list = []
    pinjie_list = []


    spatial_information_list = []

    transforms = transforms.ToTensor()

    # 创建结果文件夹
    for img in IMGs:

        output_list = []

        fenkuai = 0
        zhuanhuan = 0
        spatial_time = 0
        pred_time = 0
        tensor2img = 0
        pinjie = 0

        img_name = '.'.join(img.split('.')[0:-1])

        # time_start = time.time()  # 记录开始时间



        if opt.spatial != 2:
            # 创建结果文件夹
            os.mkdir(img_name)
            # 分块
            fenkuai_start = time.time()
            in_img_path = IMGSpath + '\\' + img
            shutil.copyfile(in_img_path, img_name + '\\in.jpg')
            inImg = Image.open(in_img_path)
            print("图片原始分辨率：" + str(inImg.width) + "*" + str(inImg.height))
            if resized_height > 0:
                resized_width = int(math.ceil(resized_height / inImg.height * inImg.width))
                print("输出图片分辨率：" + str(resized_width) + "*" + str(resized_height))
                resizeImg = inImg.resize((resized_width, resized_height))
                resizeImg = resizeImg.convert('RGB')
                resizeImg.save(img_name + '\\resize.jpg')
            else:
                shutil.copyfile(in_img_path, img_name + '\\resize.jpg')

            # makeBorder('run' + str(folder) + '\\resize.jpg', 'run' + str(folder) + '\out.jpg')

            split_size_w, split_size_h = segmentation(img_name + '\\resize.jpg', out_img_split_row, out_img_split_col)
            if opt.spatial == 1:
                # 计算spatial information
                spatial_data = dataset()
                # 进行spatial预测
                spatial_pred_list = pred(spatial_data, './spatial_information/SIweight.pth')
                spatial_information_list.append(spatial_pred_list)

            fenkuai_end = time.time()
            fenkuai = fenkuai + (fenkuai_end - fenkuai_start)

            # 读入分块图片
            opt.dataroot = img_name + '\segmentation'
            opt.load_size = split_size_w
            opt.crop_size = 0

            Dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options



        # initialize logger
        # if opt.use_wandb:
        # wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=opt.name, config=opt) if not wandb.run else wandb.run
        # wandb_run._label(repo='CycleGAN-and-pix2pix')

        # create a website
        web_dir = os.path.join(opt.results_dir, opt.name,
                               '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
        if opt.load_iter > 0:  # load_iter is 0 by default
            web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
        print('creating web directory', web_dir)
        webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

        # test with eval mode. This only affects layers like batchnorm and dropout.
        # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        # if opt.eval:
            # model.eval()

        opt.name = 'cyclegan_K50'
        # 设定模型
        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers


        # 转换图片
        if opt.spatial != 2:
            zhuanhuan_start = time.time()
            for i, data in enumerate(Dataset):
                # 切换模型
                '''
                if i < 4:
                    opt.name = 'cyclegan_K200'
                elif i < 8:
                    opt.name = 'cyclegan_K400'
                elif 8 <= i < 12:
                    opt.name = 'cyclegan_K600'
                else:
                    opt.name = 'cyclegan_K800'
                '''


                if i >= opt.num_test:  # only apply our model to opt.num_test images.
                    break
                model.set_input(data)  # unpack data from data loader
                #print(data)
                model.test()  # run inference
                visuals = model.get_current_visuals()  # get image results
                img_path = model.get_image_paths()  # get image paths
                # if i % 5 == 0:  # save images to an HTML file
                    # print('processing (%04d)-th image... %s' % (i, img_path))
                tensor2img_start = time.time()
                output_img = save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
                if opt.spatial == 1:
                    if spatial_pred_list[i] == 0:
                        output_img = util.tensor2im(data['A'])
                        output_img = Image.fromarray(output_img)
                output_list.append(output_img)
                tensor2img_end = time.time()
                tensor2img = tensor2img + (tensor2img_end - tensor2img_start)
            # webpage.save()  # save the HTML
            zhuanhuan_end = time.time()
            zhuanhuan = zhuanhuan + (zhuanhuan_end - zhuanhuan_start) - tensor2img
            print(gpu_memory_log())
            # torch.cuda.reset_max_memory_allocated()

            # 拼接
            pinjie_start = time.time()
            # shutil.move('results\\' + opt.name + '\\test_latest\images', img_name + '\\transformed_out')
            # shutil.rmtree('results')
            combine(output_list, out_img_split_row, out_img_split_col, split_size_w, split_size_h,
                    opt.d + '\\' + img_name + '.jpg')
            # cutBorder('run' + str(folder) + '\\resize.jpg', 'run' + str(folder) + '\\final.jpg', 'run' + str(folder) + '\\cutted.jpg')
            shutil.rmtree(img_name)
            pinjie_end = time.time()
            pinjie = pinjie + (pinjie_end - pinjie_start)
            pinjie_list.append(pinjie)

            fenkuai_list.append(fenkuai)
            zhuanhuan_list.append(zhuanhuan)
            tensor2img_list.append(tensor2img)
            time_avg = [mean(fenkuai_list), mean(zhuanhuan_list), mean(tensor2img_list), mean(pinjie_list)]
            with open('latency_classified.csv', 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(time_avg)


        else:
            in_img_path = IMGSpath + '\\' + img
            in_img = Image.open(in_img_path)
            image_tensor = transforms(in_img)
            # 计算spatial information
            data_square_list, spatial_time = edges_detection_square(in_img_path, 80, 40, 40)
            # 进行车牌预测，返回预测的车牌坐标列表
            pred_square_list, pred_time = pred_square(data_square_list, './spatial_information/SIweight_square.pth')
            start_time = time.time()
            for square in pred_square_list:
                # cropped_img = image_array[square[1]:square[1]+40, square[0]:square[0]+80]
                # cropped_tensor = torch.tensor(cropped_img)
                cropped_tensor = image_tensor[:, square[1]:square[1]+40, square[0]:square[0]+80]
                cropped_tensor = torch.unsqueeze(cropped_tensor, 0)
                data = {'A': cropped_tensor, 'A_paths': './'}
                model.set_input(data)  # unpack data from data loader
                model.test()  # run inference
                visuals = model.get_current_visuals()  # get image results
                im_tensor = visuals.get('fake')[0] # 获取输出的tensor
                image_tensor[:, square[1]:square[1]+40, square[0]:square[0]+80] = im_tensor
            end_time = time.time()
            zhuanhuan = end_time - start_time
            start_time = time.time()
            im = TF.to_pil_image(image_tensor)
            # im = util.tensor2im(image_tensor)
            im.save(opt.d + '\\' + img_name + '.jpg')
            end_time = time.time()
            tensor2img = end_time - start_time

            time_avg = [spatial_time, pred_time, zhuanhuan, tensor2img]
            # with open('latency_classified.csv', 'a', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerow(time_avg)








        # time_end = time.time()  # 记录结束时间
        # time_sum.append(time_end - time_start)  # 计算的时间差为程序的执行时间，单位为秒
        # print(time_end - time_start)

    # with open('latency_classified.csv', 'w') as csvfile:
    # writer = csv.writer(csvfile)
    # writer.writerow(fenkuai_list)
    # writer.writerow(zhuanhuan_list)
    # writer.writerow(baocun_list)
    # writer.writerow(pinjie_list)

    # with open('latency.csv', 'w') as csvfile:
    # writer = csv.writer(csvfile)
    # writer.writerow(time_sum)
    # with open('mem.csv', 'w') as csvfile:
    # writer = csv.writer(csvfile)
    # writer.writerow(mem)
    end = time.time()  # 记录结束时间
    sum = end - start  # 计算的时间差为程序的执行时间，单位为秒
    print("总耗时：" + str(sum) + "s")
