#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:48:07 2023

@author: ragavendiranbalasubramanian
"""
#%% Imports
import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

os.chdir("/Users/ragavendiranbalasubramanian/Documents/Adiuvo/3dPlot/3D-generator/acv")

def generate_segmentation_mask(image_path, model_type, checkpoint_path):
    # Set device for model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize SAM model
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Load and process the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Generate segmentation mask
    output_mask = mask_generator.generate(image_rgb)
    
    return output_mask


def save_and_show_output(result_dict, image_path, output_dir):
    image = cv2.imread(image_path)
    _, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)

    sorted_result = sorted(result_dict, key=lambda x: x['area'], reverse=True)

    for index, val in enumerate(sorted_result):
        mask = val['segmentation']
        output_mask_path = os.path.join(output_dir, f'segment_{index}.png')

        # Saving the mask
        cv2.imwrite(output_mask_path, (mask * 255).astype(np.uint8))

        # Visualizing the mask
        img = np.ones((mask.shape[0], mask.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, mask * 0.5)))

    plt.show()
    

def main(model_type, image_path):
    image=cv2.imread(image_path)
    # Generate segmentation mask
    segmentation_mask = generate_segmentation_mask(image_path, model_type[0], model_type[1])
    # Visualize and save the result
    output_visualization_dir = os.path.join('data/outputs/visualizations', model_type[0])
    save_and_show_output(segmentation_mask, image_path, output_visualization_dir)
  


if __name__ == "__main__":
    models = {
        "vit_b": ("vit_b", "data/models/sam_vit_b_01ec64.pth"),
        "vit_h": ("vit_h", "data/models/sam_vit_h_4b8939.pth"),
        "vit_l": ("vit_l", "data/models/sam_vit_l_0b3195.pth")
    }
    selected_model = "vit_l"  # Change this to the model you want to use
    input_image_path = 'data/images/input/Calcada-de-Santo-Andre.jpg'
    main(models[selected_model], input_image_path)
