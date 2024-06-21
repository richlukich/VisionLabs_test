import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import os
from sklearn.model_selection import train_test_split

def get_image_list_u(unlabeled_dir):
    image_list = []
    for img_name in sorted(os.listdir(unlabeled_dir)):
        img_path = os.path.join(unlabeled_dir,img_name)
        img = cv2.imread(img_path, 0)
        image_list.append(img)
    return image_list

def get_image_list_l(labeled_dir):
    image_list = []
    target_list = []
    open_dir = os.path.join(labeled_dir,'open')
    close_dir = os.path.join(labeled_dir,'close')

    for img_name in sorted(os.listdir(open_dir)):
        img_path = os.path.join(open_dir,img_name)
        img = cv2.imread(img_path, 0)
        image_list.append(img)
        target_list.append(1)

    for img_name in sorted(os.listdir(close_dir)):
        img_path = os.path.join(close_dir,img_name)
        img = cv2.imread(img_path, 0)
        image_list.append(img)
        target_list.append(0)

    X_train, X_test, y_train, y_test = train_test_split(image_list, target_list, test_size=0.3,shuffle=True, random_state=42)
    return X_train, X_test, y_train, y_test

class VaeDataset(Dataset):
    def __init__(self, img_path):
        self.img_path = img_path
        self.img_names = sorted(os.listdir(self.img_path))
    def __getitem__(self, idx):

        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_path, img_name)
        img = cv2.imread(img_path, 0)
        img = img / 255.0
        return img
    def __len__(self):
        return len(self.img_names)
    

class ClassifierDataset(Dataset):
    def __init__(self, imgs, targets):
        self.imgs = imgs
        self.targets = targets
        
    def __getitem__(self, idx):
        image = torch.tensor(self.imgs[idx], dtype=torch.float32).unsqueeze(0)
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)

        return image, targets
    
    def __len__(self):
        return len(self.imgs)
