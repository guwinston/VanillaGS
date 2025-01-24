import os
import sys
import time
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.uint8).astype(np.float32)
    img2 = img2.astype(np.uint8).astype(np.float32)
    mse = np.mean((img1 - img2) ** 2).item()
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return psnr

if __name__ == "__main__":
    scene = "garden"
    flashgs_folder = f"output/{scene}/batch"
    vanilla_folder = f"output/{scene}/vanilla"
    psnrs = []
    img_list = [img_file for img_file in os.listdir(flashgs_folder) if img_file.endswith(".jpg")]
    pbar = tqdm(img_list, total=len(img_list))
    abnormal_img_list = []
    for img_name in pbar:
        flashgs_image = Image.open(os.path.join(flashgs_folder, img_name))
        vanilla_image = Image.open(os.path.join(vanilla_folder, img_name))
        psnr = calculate_psnr(np.array(flashgs_image), np.array(vanilla_image))
        if psnr < 30: abnormal_img_list.append(img_name)
        psnrs.append(psnr)
        pbar.set_description(f"PSNR = {psnr:.3f} | {np.mean(psnrs):.3f}")
    print(abnormal_img_list)
    
    plt.figure()
    plt.plot(psnrs, label="PSNR")
    plt.xlabel("Image Index")
    plt.ylabel("PSNR")
    plt.title("PSNR of FlashGS vs VanillaGS")
    plt.legend()
    plt.savefig("output/psnr.png")
    plt.show()
