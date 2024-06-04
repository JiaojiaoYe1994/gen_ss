# !/usr/bin/env python
# coding: utf-8
#
# Author: Jiaojiao Ye
# Date:   18 May 2024

import os
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
from copy import deepcopy

# image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
# image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

# image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-seg/resolve/main/images/house.png").convert('RGB')

# pixel_values = image_processor(image, return_tensors="pt").pixel_values

# with torch.no_grad():
#   outputs = image_segmentor(pixel_values)

# seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

# color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3

# for label, color in enumerate(palette):
#     color_seg[seg == label, :] = color

# color_seg = color_seg.astype(np.uint8)

# image = Image.fromarray(color_seg)

# Load model
# ControlNet version is compiled with Stable Diffusion version, can be seen here: https://github.com/lllyasviel/ControlNet-v1-1-nightly
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16
) # "lllyasviel/sd-controlnet-seg"， "thibaud/controlnet-sd21-ade20k-diffusers", "lllyasviel/control_v11p_sd21_seg"

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet,
    safety_checker=None, 
    torch_dtype=torch.float16
) # "runwayml/stable-diffusion-v1-5",  "stabilityai/stable-diffusion-2-1"

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
# pipe.enable_xformers_memory_efficient_attention()

# pipe.enable_model_cpu_offload()
pipe = pipe.to("cuda")   

# PASCAl VOC palette 
palette = [0, 0, 0, 
           128, 0, 0,
           0, 128, 0, 
           128, 128, 0, 
           0, 0, 128, 
           128, 0, 128, 
           0, 128, 128, 
           128, 128, 128,
           64, 0, 0, 
           64, 128, 0, 
           192, 128, 0, 
           64, 0, 128, 
           192, 0, 128, 
           64, 128, 128, 
           192, 128, 128,
           0, 64, 0, 
           128, 64, 0,
           0, 192, 0,
           128, 192, 0,
           0, 64, 128,
           128, 64, 128,
           0, 192, 128,
           128, 192, 128,
           64, 64, 0,
           192, 64, 0,
           64, 192, 0,
           192, 192, 0]

classes = ['background','aeroplane','bicycle','bird', 'boat','bottle','bus','car','cat','chair',
           'cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep',
           'sofa','train','tvmonitor']

# ADE20K palette 
palette_ade = np.asarray([
                          [0, 0, 0],
                          [0, 255, 82],
                          [255, 245, 0],
                          [128, 0, 128], # bird
                          [173, 255, 0],
                          [0, 255, 10],
                          [255, 0, 245],
                          [0, 102, 200],
                          [128, 128, 128], # cat
                          [204, 70, 3],
                          [64, 128, 0], # cow
                          [0, 255, 112],
                          [0, 0, 0], # dog [64, 0, 128]
                          [192, 0, 128], # horse
                          [163, 0, 255], 
                          [150, 5, 61],
                          [204, 255, 4], 
                          [ 128, 64, 0], # sheep
                          [11, 102, 255],
                          [0, 192, 0], # train
                          [0, 255, 194] 
                         ])

palette_pascal = np.array(palette).reshape(-1, 3)

class_dict = dict(zip(list(range(0, 21)), classes))

# transfer from PASCAl VOC palette to ADE20K palette, 
# input: color_seg, numpy array
# output: color_seg_ade, numpy array 
palette_pascal_list = palette_pascal.tolist()
def transfer_pascal2ade(color_seg):

    H, W, = color_seg.shape

    # color_seg_ade = deepcopy(color_seg)
    color_seg_ade = np.zeros((H, W, 3), dtype=np.uint8)

    label_list = ""
    for h in range(H):
        for w in range(W):
            # print('pixel pos and pixel value:', h,w, color_seg[h, w])
            label_idx = color_seg[h, w]
            if label_idx in [3, 7, 10, 12, 13, 17, 19]:
                return None
            if label_idx in class_dict:
                color_seg_ade[h, w, :] = palette_ade[label_idx]
                if class_dict[label_idx] not in label_list and label_idx>0:
                    label_list += "{}, ".format(class_dict[label_idx])
            # try:
            #     label_idx = palette_pascal_list.index(color_seg[h, w,:].tolist())
            #
            #     # keep background as (0,0,0)
            #     if label_idx > 0:
            #         color_seg_ade[h,w,:]  = palette_ade[label_idx]
            #
            #         if class_dict[label_idx]  not in label_list:
            #             label_list += "{}, ".format(class_dict[label_idx])
            # except:
            #     # if pixel color not in the predefined Pascal color palette, i.e SAM assigned pixel value out of class vocabulary, assign it as background
            #     color_seg_ade[h, w, :] = [0, 0, 0] # assume [0,0,0] background in ADE20K
            #     print(label_list)
    return color_seg_ade, label_list

# Path for semanetic conditions, outputs
img_list = "/mnt/nas/jiaojiao/2024/wsss_sam/metadata/pascal/train_aug(id).txt"
img_path = "/mnt/nas/jiaojiao/2024/wsss_sam/pascal_train_seg_0521"
output_path = "./output/pascal_train_aug_0523/"
adeseg_path = "./output/pascal_train_adeseg_0523/"

# create ouput path
try:
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(adeseg_path, exist_ok=True)

    print(f"Directories '{output_path}' '{adeseg_path}' created successfully or already exist.")
except Exception as e:
    print(f"An error occurred: {e}")

with open(img_list) as f:
    img_list = []
    for line in f:
        img_list.append(line[:-1])

cnt = 0
for index, name in enumerate(img_list[10000:]):
     # load image
    image_path = os.path.join(img_path, '{}.png'.format(name))
    # print(image_path)
    if os.path.exists(image_path):
        # Test if file already exist:
        if '{}.png'.format(name) in os.listdir(output_path):
            continue
            
        # Todo: better way to see if image is in segmented folders
        image_pil = Image.open(image_path).convert("RGB")  # load image

        try:
            color_seg, label = transfer_pascal2ade(np.asarray(Image.open(image_path), dtype=np.int32))
        except:
            continue

    #     color_seg, label = transfer_pascal2ade(np.array(image_pil))

        print("Index {}, Synthesized {}th image with label {}, File {}".format(index+1, cnt+1, label, name))

        # synthesize image
        # Todo: prompt
        # Prompt: an image of object classes
        image = Image.fromarray(color_seg)

        # Validate transfered color map
        image.save(adeseg_path+'{}.png'.format(name))
        for i in range(1):
            image = pipe("An real image of "+ label, image, num_inference_steps=50).images[0]
            image.save(os.path.join(output_path, '{}.png'.format(name, i)))

        cnt += 1
    else:
        print('Semantic file not exist') # Semantic file not exist, or not segmented by SAM
        pass