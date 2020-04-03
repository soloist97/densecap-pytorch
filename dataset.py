import os
import pickle

import h5py
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class DenseCapDataset(Dataset):

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

        self.vg_data = h5py.File(vg_data_path, 'r')
        self.look_up_tables = pickle.load(open(look_up_tables_path, 'rb'))

    def set_dataset_type(self, dataset_type, verbose=True):

        assert dataset_type in {None, 'train', 'test', 'val'}

        if verbose:
            print('[DenseCapDataset]: {} switch to {}'.format(self.dataset_type, dataset_type))

        self.dataset_type = dataset_type

    def __getitem__(self, idx):

        vg_idx = self.look_up_tables['split'][self.dataset_type][idx] if self.dataset_type else idx

        img_path = os.path.join(self.img_dir_root, self.look_up_tables['idx_to_directory'][vg_idx],
                                self.look_up_tables['idx_to_filename'][vg_idx])

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        first_box_idx = self.vg_data['img_to_first_box'][vg_idx]
        last_box_idx = self.vg_data['img_to_last_box'][vg_idx]

        boxes = torch.as_tensor(self.vg_data['boxes'][first_box_idx: last_box_idx+1], dtype=torch.float32)
        caps = torch.as_tensor(self.vg_data['captions'][first_box_idx: last_box_idx+1], dtype=torch.long)
        caps_len = torch.as_tensor(self.vg_data['lengths'][first_box_idx: last_box_idx+1], dtype=torch.long)

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

if __name__ == '__main__':

    IMG_DIR_ROOT = './data/visual-genome'
    VG_DATA_PATH = './data/VG-regions.h5'
    LOOK_UP_TABLES_PATH = './data/VG-regions-dicts.pkl'

    dcd = DenseCapDataset(IMG_DIR_ROOT, VG_DATA_PATH, LOOK_UP_TABLES_PATH)

    print('all', len(dcd))
    print(dcd[0])

    for data_type in {'train', 'test', 'val'}:

        dcd.set_dataset_type(data_type)

        print(data_type, len(dcd))
        print(dcd[0])
