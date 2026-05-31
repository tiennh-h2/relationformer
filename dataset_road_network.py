import scipy
import os
import sys
import numpy as np
import random
import pickle
import json
import scipy.ndimage
import imageio
import math
from PIL import Image
import torch
import pyvista
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import torchvision.transforms as T
import torchvision.transforms.functional as tvf

train_transform = T.Compose([
    T.Resize((1024, 1024)),
    T.ToTensor(),
])

val_transform = T.Compose([
    T.Resize((1024, 1024)),
    T.ToTensor(),
])

class Sat2GraphDataLoader(Dataset):
    def __init__(self, data, transform=None, return_class=False, return_original_image=False):
        self.data = data
        self.transform = transform
        self.return_class = return_class
        self.return_original_image = return_original_image

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # =========================
        # Image
        # =========================
        org_image = Image.open(sample["img"]).convert("RGB")

        # segmentation
        seg = Image.open(sample["seg"]).convert("RGB")

        # Apply resize
        if self.transform is not None:
            image = self.transform(org_image)

            # Resize segmentation manually
            seg = T.Resize((1024, 1024))(seg)
            seg = np.array(seg).astype(np.float32)

        else:
            image = T.ToTensor()(org_image)
            seg = np.array(seg).astype(np.float32)

        # Normalize segmentation
        if seg.max() > 0:
            seg = seg / seg.max()

        # Normalize image
        image = tvf.normalize(
            image,
            mean=self.mean,
            std=self.std,
        )

        # =========================
        # Convert segmentation
        # =========================
        seg = torch.from_numpy(seg)

        if seg.ndim == 2:
            seg = seg.unsqueeze(0)
        else:
            seg = seg.permute(2, 0, 1)

        seg = seg.float() - 0.5

        # =========================
        # Graph
        # =========================
        vtk_data = pyvista.read(sample["vtp"])

        coordinates = torch.from_numpy(
            np.asarray(vtk_data.points, dtype=np.float32)
        )

        lines = torch.from_numpy(
            vtk_data.lines.reshape(-1, 3)
        ).long()

        classes = torch.from_numpy(
            vtk_data.point_data["class"]
        ).long()

        outputs = [
            image,
            seg,
            coordinates[:, :2],
            lines[:, 1:]
        ]

        if self.return_class:
            outputs.append(classes)

        if self.return_original_image:
            outputs.append(org_image)

        return tuple(outputs)



def build_road_network_data(config, mode='train', split=0.9):
    """[summary]

    Args:
        data_dir (str, optional): [description]. Defaults to ''.
        mode (str, optional): [description]. Defaults to 'train'.
        split (float, optional): [description]. Defaults to 0.8.

    Returns:
        [type]: [description]
    """    
    img_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
    seg_folder = os.path.join(config.DATA.DATA_PATH, 'seg')
    vtk_folder = os.path.join(config.DATA.DATA_PATH, 'vtp')
    img_files = []
    vtk_files = []
    seg_files = []

    for file_ in os.listdir(img_folder):
        file_ = file_[:-8]
        img_files.append(os.path.join(img_folder, file_+'data.png'))
        vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))
        seg_files.append(os.path.join(seg_folder, file_+'seg.png'))

    data_dicts = [
        {"img": img_file, "vtp": vtk_file, "seg": seg_file} for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
        ]
    if mode=='train':
        ds = Sat2GraphDataLoader(
            data=data_dicts,
            transform=train_transform,
            return_class=getattr(config.MODEL, "GIVEN_BOXES", False)
        )
        return ds
    elif mode=='test':
        img_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'raw')
        seg_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'seg')
        vtk_folder = os.path.join(config.DATA.TEST_DATA_PATH, 'vtp')
        img_files = []
        vtk_files = []
        seg_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_+'data.png'))
            vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))
            seg_files.append(os.path.join(seg_folder, file_+'seg.png'))

        data_dicts = [
            {"img": img_file, "vtp": vtk_file, "seg": seg_file} for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
            ]
        ds = Sat2GraphDataLoader(
            data=data_dicts,
            transform=val_transform,
            return_class=getattr(config.MODEL, "GIVEN_BOXES", False),
            return_original_image=True
        )
        return ds
    elif mode=='split':
        img_folder = os.path.join(config.DATA.DATA_PATH, 'raw')
        seg_folder = os.path.join(config.DATA.DATA_PATH, 'seg')
        vtk_folder = os.path.join(config.DATA.DATA_PATH, 'vtp')
        img_files = []
        vtk_files = []
        seg_files = []

        for file_ in os.listdir(img_folder):
            file_ = file_[:-8]
            img_files.append(os.path.join(img_folder, file_+'data.png'))
            vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))
            seg_files.append(os.path.join(seg_folder, file_+'seg.png'))

        data_dicts = [
            {"img": img_file, "vtp": vtk_file, "seg": seg_file} for img_file, vtk_file, seg_file in zip(img_files, vtk_files, seg_files)
            ]
        random.seed(config.DATA.SEED)
        random.shuffle(data_dicts)
        train_split = int(split*len(data_dicts))
        train_files, val_files = data_dicts[:train_split], data_dicts[train_split:]
        train_ds = Sat2GraphDataLoader(
            data=train_files,
            transform=train_transform,
            return_class=getattr(config.MODEL, "GIVEN_BOXES", False)
        )
        val_ds = Sat2GraphDataLoader(
            data=val_files,
            transform=val_transform,
            return_class=getattr(config.MODEL, "GIVEN_BOXES", False)
        )
        return train_ds, val_ds