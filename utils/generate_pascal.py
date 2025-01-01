# !/usr/bin/env python
# coding: utf-8
#
# Author: Jiaojiao Ye
# Date:   18 May 2024

import os
import json
import argparse

from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import cv2

import sys
sys.path.append('/mnt/nas/jiaojiao/2024/gen_ss')

from datasets.pascal import palette, classes, pascal_ade
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import BlipProcessor, BlipForConditionalGeneration
from datasets.pascal import palette, classes, pascal_ade 


parser = argparse.ArgumentParser("Synthetic Data Generation", add_help=True)
parser.add_argument("--con", type=str, default="seg", required=True, help="conditional for Diffusion model, choice in [edge, seg]")
parser.add_argument("--output_dir", type=str, default="coco_train", required=True, help="output_dir"
    )
parser.add_argument("--syn_size", type=int, default=1, help="Scale Up size of Augmentation")
parser.add_argument("--device", type=str, default="cuda:0", help="cpu or cuda:0,cuda:1")

args = parser.parse_args()


# Load the processor and model
print("Loda BLIP Image Caption Pretrained Model")
model_name = "Mouwiya/BLIP_image_captioning"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name)
# ControlNet version is compiled with Stable Diffusion version, can be seen here: https://github.com/lllyasviel/ControlNet-v1-1-nightly
if args.con == "seg":
    controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16
) # "lllyasviel/sd-controlnet-seg","thibaud/controlnet-sd21-ade20k-diffusers","lllyasviel/control_v11p_sd21_seg", "lllyasviel/control_v11p_sd15_seg"
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet,
    safety_checker=None, 
    torch_dtype=torch.float16
) # "runwayml/stable-diffusion-v1-5",  "stabilityai/stable-diffusion-2-1"
elif args.con == "edge":
    # Canny_edge Mao
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
    )

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Remove if you do not have xformers installed
# see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
# for installation instructions
# pipe.enable_xformers_memory_efficient_attention()

# pipe.enable_model_cpu_offload()
pipe = pipe.to(args.device)   

palette_pascal = np.array(palette).reshape(-1, 3)
palette_pascal_list = palette_pascal.tolist()

class_dict = dict(zip(list(range(0, 21)), classes))

# transfer from PASCAl VOC palette to ADE20K palette, 
# input: color_seg, numpy array
# output: color_seg_ade, numpy array 
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
                color_seg_ade[h, w, :] = pascal_ade[label_idx]
                if class_dict[label_idx] not in label_list and label_idx>0:
                    label_list += "{}, ".format(class_dict[label_idx])
    
    return color_seg_ade, label_list

# Path for semanetic conditions, outputs
img_list = "/mnt/nas/jiaojiao/2024/wsss_sam/metadata/pascal/train_aug(id).txt"
img_path = "/mnt/nas/jiaojiao/2024/gen_ss/output/pascal_train_seg_0521"
# output_path = "./output/pascal_train_0719_class/"
output_path = "./output/pascal/" + args.output_dir # + "_" + str(args.syn_size)
adeseg_path = "./output/pascal_adeseg_0718/"
img_gt_path = "/mnt/nas/jiaojiao/data/VOCdevkit/VOC2012/JPEGImages"
cnt = 0
caption_list = {}

# create output path
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
        
        
def get_inputs(batch_size=1, image=Image.fromarray(np.zeros((500, 375))), prompt= "An image"):                                   
    
    generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]                                                                                                                                                             
    prompts = batch_size * [prompt]                                                                                            
    
    # predifined prompt list
    prompts = prompt
    num_inference_steps = 50                                                                                                                                                                                                                    
    return {"prompt": prompts, "image": image, "generator": generator, "num_inference_steps": num_inference_steps} 


for index, name in tqdm(
        enumerate(img_list),
        total=len(img_list),
        dynamic_ncols=True,
    ):
     # Load image
    image_path = os.path.join(img_path, '{}.png'.format(name))

#     print("image_path: ", image_path)
    
    if os.path.exists(image_path):
        # Test if file already exist:
        if '{}_0.png'.format(name) in os.listdir(output_path):
            continue

        # Use BLIP to caption image
        image = Image.open(os.path.join(img_gt_path, '{}.jpg'.format(name))).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        # Todo: better way to see if image is in segmented folders
        image_pil = Image.open(image_path).convert("RGB")  # load image
                
        try:
            color_seg, label = transfer_pascal2ade(np.asarray(Image.open(image_path), dtype=np.int32))
        except:            
            continue

        print("Index {}, Synthesized {}th image with label {}, File {}".format(index+1, cnt+1, label, name))

        # Synthesize image
        color_seg = color_seg.astype(np.uint8)
        seg = Image.fromarray(color_seg)
        # Validate transfered color map
        seg.save(adeseg_path+'{}.png'.format(name))
        
        if args.con == "edge":
            # Extract canny edge map as conditional generation
            low_threshold = 100
            high_threshold = 200

            canny_image = cv2.Canny(np.array(image), low_threshold, high_threshold)
            canny_image = canny_image[:, :, None]
            canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
            canny_image = Image.fromarray(canny_image)
        
        # Todo: prompt
        # Prompt: an image of object classes
#         prompt = "A real image of " + label # caption
        # Use image caption to generate prompt
        prompt = caption + ", " + label
        print("prompt: ", prompt, seg.size, image.size)
        
        # Load pre-defined prompt for generation, GPT-enhanced prompt
        import json

        # Specify the path to JSON file
        file_path = "./output/prompt/expanded_caption_name.json"

        # Open and load the JSON file
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        prompt_GPT = data[name]
        print("prompt GPT: ", prompt_GPT[:args.syn_size])
        
        images = pipe(**get_inputs(batch_size=args.syn_size, image=seg, prompt=prompt_GPT[:args.syn_size])).images  
        
        for i, image in enumerate(images):
            image = image.resize(seg.size)
            
            image.save(os.path.join(output_path, '{}_{}.png'.format(name, i)))
        
        cnt += 1
#         caption_list[name] = prompt + '\n'
        caption_list[name] = prompt_GPT[:args.syn_size]
    else:
        print('Semantic file not exist') # Semantic file not exist, or not segmented by SAM
        pass

# Serialize data into file:

json.dump( caption_list, open(os.path.join(output_path, "prompt_list.json"), 'w' ) )
