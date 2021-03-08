import os
import pickle

import h5py
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


__all__ = [
    "DataLoaderPFG", "DenseCapDataset", 'DenseCapDatasetV2'
]


class DataLoaderPFG(DataLoader):
    """
    Prefetch version of DataLoader: https://github.com/IgorSusmelj/pytorch-styleguide/issues/5
    """

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class DenseCapDataset(Dataset):
    """Images are loaded from by open specific file
    """

    @staticmethod
    def collate_fn(batch):
        """Use in torch.utils.data.DataLoader
        """

        return tuple(zip(*batch)) # as tuples instead of stacked tensors

    @staticmethod
    def get_transform():
        """More complicated transform utils in torchvison/references/detection/transforms.py
        """

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return transform

    def __init__(self, img_dir_root, vg_data_path, look_up_tables_path, dataset_type=None, transform=None):

        assert dataset_type in {None, 'train', 'test', 'val'}

        super(DenseCapDataset, self).__init__()

        self.img_dir_root = img_dir_root
        self.vg_data_path = vg_data_path
        self.look_up_tables_path = look_up_tables_path
        self.dataset_type = dataset_type  # if dataset_type is None, all data will be use
        self.transform = transform

        # === load data here ====
        self.look_up_tables = pickle.load(open(look_up_tables_path, 'rb'))

    def set_dataset_type(self, dataset_type, verbose=True):

        assert dataset_type in {None, 'train', 'test', 'val'}

        if verbose:
            print('[DenseCapDataset]: {} switch to {}'.format(self.dataset_type, dataset_type))

        self.dataset_type = dataset_type

    def __getitem__(self, idx):

        with h5py.File(self.vg_data_path, 'r') as vg_data:

            vg_idx = self.look_up_tables['split'][self.dataset_type][idx] if self.dataset_type else idx

            img_path = os.path.join(self.img_dir_root, self.look_up_tables['idx_to_directory'][vg_idx],
                                    self.look_up_tables['idx_to_filename'][vg_idx])

            img = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)

            first_box_idx = vg_data['img_to_first_box'][vg_idx]
            last_box_idx = vg_data['img_to_last_box'][vg_idx]

            boxes = torch.as_tensor(vg_data['boxes'][first_box_idx: last_box_idx+1], dtype=torch.float32)
            caps = torch.as_tensor(vg_data['captions'][first_box_idx: last_box_idx+1], dtype=torch.long)
            caps_len = torch.as_tensor(vg_data['lengths'][first_box_idx: last_box_idx+1], dtype=torch.long)

            targets = {
                'boxes': boxes,
                'caps': caps,
                'caps_len': caps_len,
            }

            info = {
                'idx': vg_idx,
                'dir': self.look_up_tables['idx_to_directory'][vg_idx],
                'file_name': self.look_up_tables['idx_to_filename'][vg_idx]
            }

        return img, targets, info

    def __len__(self):

        if self.dataset_type:
            return len(self.look_up_tables['split'][self.dataset_type])
        else:
            return len(self.look_up_tables['filename_to_idx'])


class DenseCapDatasetV2(Dataset):
    """Images are stored in VG-regions.h5
    """

    @staticmethod
    def collate_fn(batch):
        """Use in torch.utils.data.DataLoader
        """

        return tuple(zip(*batch)) # as tuples instead of stacked tensors

    @staticmethod
    def get_transform():
        """More complicated transform utils in torchvison/references/detection/transforms.py
        """

        transform = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return transform

    def __init__(self, vg_data_path, look_up_tables_path, dataset_type=None, transform=None):

        assert dataset_type in {None, 'train', 'test', 'val'}

        super(DenseCapDatasetV2, self).__init__()

        self.vg_data_path = vg_data_path
        self.look_up_tables_path = look_up_tables_path
        self.dataset_type = dataset_type  # if dataset_type is None, all data will be use
        self.transform = transform

        # === load data here ====
        self.look_up_tables = pickle.load(open(look_up_tables_path, 'rb'))

    def set_dataset_type(self, dataset_type, verbose=True):

        assert dataset_type in {None, 'train', 'test', 'val'}

        if verbose:
            print('[DenseCapDataset]: {} switch to {}'.format(self.dataset_type, dataset_type))

        self.dataset_type = dataset_type

    def __getitem__(self, idx):

        with h5py.File(self.vg_data_path, 'r') as vg_data:

            vg_idx = self.look_up_tables['split'][self.dataset_type][idx] if self.dataset_type else idx

            img = vg_data['images'][vg_idx]
            h = vg_data['image_heights'][vg_idx]
            w = vg_data['image_widths'][vg_idx]

            img = torch.tensor(img[:, :h, :w] / 255., dtype=torch.float32)  # get rid of zero padding

            if self.transform is not None:
                img = self.transform(img)

            first_box_idx = vg_data['img_to_first_box'][vg_idx]
            last_box_idx = vg_data['img_to_last_box'][vg_idx]

            boxes = torch.as_tensor(vg_data['boxes'][first_box_idx: last_box_idx+1], dtype=torch.float32)
            caps = torch.as_tensor(vg_data['captions'][first_box_idx: last_box_idx+1], dtype=torch.long)
            caps_len = torch.as_tensor(vg_data['lengths'][first_box_idx: last_box_idx+1], dtype=torch.long)

            targets = {
                'boxes': boxes,
                'caps': caps,
                'caps_len': caps_len,
            }

            info = {
                'idx': vg_idx,
                'dir': self.look_up_tables['idx_to_directory'][vg_idx],
                'file_name': self.look_up_tables['idx_to_filename'][vg_idx]
            }

        return img, targets, info

    def __len__(self):

        if self.dataset_type:
            return len(self.look_up_tables['split'][self.dataset_type])
        else:
            return len(self.look_up_tables['filename_to_idx'])
