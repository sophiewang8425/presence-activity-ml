#present=1 ; empty= 0
#step 2: image canonicalization -> same image size, channel order, and numeric scale
#step 3: normalization-> map pixel values from 0 to 1
#step 4: 
#step 5: train/ validation split
#step 6: augmentation (optional)

import cv2 as cv

from pathlib import Path
import random

from PIL import Image
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

class PresenceDataSet(Dataset):
    def __init__(self,root_dir="data/raw",image_size=160,split="train",val_fraction=0.2,seed=0,augment=False):
        """root_dir: folder with empty and presence image files
        image_size= resize size
        split= train or validate
        val_fraction: frac for validation
        seed=for shuffling
        augment= augment or not"""
        self.image_size=image_size
        self.split=split
        self.augment=augment
        self.samples=[]
        root_dir=Path(root_dir)
        empty_dir=root_dir/ "empty"
        present_dir=root_dir/"present"
        for file in present_dir.glob("*.jpg"):
            self.samples.append((file,1))
        for file in empty_dir.glob("*.jpg"):
            self.samples.append((file,0))
        #shuffle dataset
        ran=random.Random(seed)
        ran.shuffle(self.samples)
        #split data
        split_idx=int(len(self.samples)*(1-val_fraction))

        if split=="train":
            self.samples=self.samples[:split_idx]
        elif split=="val":
            self.samples=self.samples[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        """Returns image tensor and label"""
        file_path,label=self.samples[idx]
        image=cv.imread(str(file_path))
        image=cv.cvtColor(image,cv.COLOR_BGR2RGB) #makes BGR->RGB
        #resize image
        new_dimensions=(self.image_size,self.image_size)
        resized_image=cv.resize(image,new_dimensions,interpolation=cv.INTER_AREA)
        #scale 0->255 to 0->1
        scaled_image=resized_image/255.0
        #(H,W,C)->(C,H,W)
        image=torch.from_numpy(scaled_image).permute(2,0,1).float()
        return (image,label)



    def augmentation(self,image_array):
        """return modified image_array"""
        pass
