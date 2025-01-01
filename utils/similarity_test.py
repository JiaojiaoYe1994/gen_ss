# !/usr/bin/env python
# coding: utf-8
#
# Author: Jiaojiao Ye
# Date:   12 June 2024
'''The Unreasonable Effectiveness of Deep Features as a Perceptual Metric'''
import time

import torch
from torchvision import transforms
import torchvision
from tqdm import tqdm
import lpips
import os

start_time = time.time()
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
model_time = time.time()-start_time

img0 = torchvision.io.read_image("/mnt/nas/jiaojiao/data/VOC_new/VOC2012/JPEGImages/2007_000032.jpg")
img1 = torchvision.io.read_image("/mnt/nas/jiaojiao/data/VOC_new/VOC2012/JPEGImages/2007_000032_new.png")
io_time = time.time()-start_time

resize_transform = transforms.Resize(( 64, 64))
# 应用转换
img0_= resize_transform(img0)
img1_= resize_transform(img1)

img0_ = torch.reshape(img0_, (1,3,64,64))
img1_ = torch.reshape(img1_, (1,3,64,64))

d = loss_fn_alex(img0_, img1_)

process_time = time.time()-start_time

print("Perceptual Similarity: ", d.item() )
print("Time consumptation model time: ", model_time, "io_time: ", io_time-model_time, "process time: ", process_time-io_time )

# data_root = "/mnt/nas/jiaojiao/data/VOC_aug/VOC2012/JPEGImages/"
# filelist = os.listdir(data_root)
# # print(filelist)

def calculate_similarity(img_path1, img_path2):
    img1 = torchvision.io.read_image(os.path.join(img_path1))
    img2 = torchvision.io.read_image(os.path.join(img_path2))
    

    img1_ = resize_transform(img1)
    img2_ = resize_transform(img2)

    img1_ = torch.reshape(img1_, (1, 3, 64, 64))
    img2_ = torch.reshape(img2_, (1, 3, 64, 64))

    d = loss_fn_alex(img1_, img2_)
    
    return d
    
    d_all = []
    data_size = len(datalist)
    

    for i, name0 in tqdm(enumerate(datalist)):
        img0 = torchvision.io.read_image(os.path.join(data_root, name0))
        for name1 in tqdm(datalist[i+1:]):
            img1 = torchvision.io.read_image(os.path.join(data_root, name1))

            img0_ = resize_transform(img0)
            img1_ = resize_transform(img1)

            img0_ = torch.reshape(img0_, (1, 3, 64, 64))
            img1_ = torch.reshape(img1_, (1, 3, 64, 64))

            d = loss_fn_alex(img0_, img1_)

            d_all.append(d)

    print(len(d_all))

    return sum(d_all) / len(d_all)

# def calculate_inclass_similarity():
# d = calculate_similarity(filelist)
# print("Perceptual Similarity: ", d.item() )
