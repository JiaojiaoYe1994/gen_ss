# !/usr/bin/env python
# coding: utf-8
#
# Author: Jiaojiao Ye
# Date:   27 June 2024

import os
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    # Load feature extractor
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    data_folder_root = "/mnt/nas/jiaojiao/data/VOCdevkit/VOC2012/JPEGImages/"
    img_list = "/mnt/nas/jiaojiao/2024/wsss_sam/metadata/pascal/train_aug(id).txt"
    syn_folder_root = "/mnt/nas/jiaojiao/data/VOC_new/VOC2012"
    syn_list = "/mnt/nas/jiaojiao/data/VOC_new/VOC2012/ImageSets/SegmentationAug/train_aug_syn1.txt"
    
    data_folder_root = "/mnt/nas/jiaojiao/data/VOC_new/VOC2012"
    img_list = "/mnt/nas/jiaojiao/data/VOC_new/VOC2012/ImageSets/SegmentationAug/train_aug_syn1.txt"
    syn_folder_root = "/mnt/nas/jiaojiao/data/VOC_new/VOC2012"
    syn_list = "/mnt/nas/jiaojiao/data/VOC_new/VOC2012/ImageSets/SegmentationAug/train_aug_synvoc.txt"

    # Load data
#     with open(img_list) as f:
#         imgs = []
#         for name in f:
#     #         img_list.append(line[:-1])
#             imgs.append(preprocess(Image.open( os.path.join(data_folder_root, name[:-1]+".jpg"))))
    
    with open(img_list) as f:
        imgs = []
        for name in f:
            name, _ = name.split(" ")
            name_new = name.replace(".png", "_new.png")
            try:
                imgs.append(preprocess(Image.open( data_folder_root+ name_new)) )
            except:
                continue
                
                
    # Load synthetic data
    with open(syn_list) as f:
        syns = []
        for name in f:
            name, _ =name.split(" ")
            name_new = name.replace(".png", "_new.png")
            try:
                syns.append(preprocess(Image.open( syn_folder_root+ name_new)) )
            except:
                continue
                
    # Extract features using CLIP encoder
    with torch.no_grad():
        image_embeddings = model.encode_image(torch.stack(imgs[:6000]).to(device))
        syn_embeddings = model.encode_image(torch.stack(syns[:6000]).to(device))

    # PCA on feature matrix
    scaler = StandardScaler()    
    x_scaled = scaler.fit_transform(image_embeddings.cpu())

    # Apply PCA
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)

    # Sample data
    x = x_pca[:,0]
    y = x_pca[:,1]

    # Synthetic data
    x_syn_scaled = scaler.fit_transform(syn_embeddings.cpu())

    # Apply PCA
    pca = PCA(n_components=2)
    x_syn_pca = pca.fit_transform(x_syn_scaled)


    # Sample data
    x_syn = x_syn_pca[:,0]
    y_syn = x_syn_pca[:,1]

    # Create scatter plot
    plt.scatter(x, y, s=1.0,)
    plt.scatter(x_syn, y_syn,s=1.0, alpha=0.9,  c='tab:orange')

    # Add title and labels
    plt.title("Visualization of the synthetic and real data distribution")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    plt.axis("off")
    plt.legend(['real data', 'synthetic data'])

    # Show plot
#     plt.show()
    plt.savefig("./data_distribution.png")
    plt.close()
    return None


main()