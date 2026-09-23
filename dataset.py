from transform import Transforms
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def make_dataset(dir):
    """
    Recursively finds all image files in a directory.
    """
    img_paths = []
    names = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            # You might want to add a check for image extensions, e.g., if fname.endswith(('.png', '.jpg')):
            path = os.path.join(root, fname)
            img_paths.append(path)
            names.append(fname)

    return img_paths, names


class Load_Dataset(Dataset):
    def __init__(self, opt):
        super(Load_Dataset, self).__init__()
        self.opt = opt

        # Define folder names for each dataset
        if self.opt.dataset in ['CAU-Flood', 'Wuhan']:
            img1_folder = 'rgb'
            img2_folder = 'sar'
            label_folder = 'mask'
        elif self.opt.dataset in ['Florence']:
            img1_folder = 'sat'
            img2_folder = 'uav'
            label_folder = 'label'
        else:
            # Raise an error if the dataset name is not recognized
            raise ValueError(f"Dataset '{self.opt.dataset}' is not supported or mis-spelled. "
                             f"Supported datasets are: 'CAU-Flood', 'Florence', 'Wuhan'.")

        # Construct the paths to the respective data folders
        base_path = os.path.join(opt.dataroot, opt.dataset, opt.phase)
        self.dir1 = os.path.join(base_path, img1_folder)
        self.dir2 = os.path.join(base_path, img2_folder)
        self.dir_label = os.path.join(base_path, label_folder)

        # Get the list of all file paths
        self.t1_paths, self.fnames = make_dataset(self.dir1)
        self.t2_paths, _ = make_dataset(self.dir2)
        self.label_paths, _ = make_dataset(self.dir_label)

        self.dataset_size = len(self.t1_paths)

        # Define transformations
        self.normalize = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform = transforms.Compose([Transforms()])  # Custom transforms from .transform file
        self.to_tensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # --- 1. Define Target Size ---
        target_size = (self.opt.input_size, self.opt.input_size)

        # --- 2. Load and Resize Images and Label ---
        t1_path = self.t1_paths[index]
        fname = self.fnames[index]

        img1 = Image.open(t1_path).convert('RGB').resize(target_size, Image.Resampling.BILINEAR)

        t2_path = self.t2_paths[index]
        img2 = Image.open(t2_path).convert('RGB').resize(target_size, Image.Resampling.BILINEAR)

        label_path = self.label_paths[index]

        cd_label = Image.open(label_path).convert('L').resize(target_size, Image.Resampling.NEAREST)

        # --- 3. Apply Augmentations (if in training phase) ---
        if self.opt.phase == 'train':
            _data = self.transform({'img1': img1, 'img2': img2, 'cd_label': cd_label})
            img1, img2, cd_label = _data['img1'], _data['img2'], _data['cd_label']

        # --- 4. Convert to Tensors and Normalize ---
        img1 = self.to_tensor(img1)
        img2 = self.to_tensor(img2)
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        # Convert final label (which is a PIL Image) to a numpy array, binarize to 0/1, and then to a LongTensor
        label_array = np.array(cd_label) // 255
        cd_label = torch.from_numpy(label_array).long()

        input_dict = {'img1': img1, 'img2': img2, 'cd_label': cd_label, 'fname': fname}

        return input_dict


class DataLoader(torch.utils.data.Dataset):
    """
    This is a wrapper class for the PyTorch DataLoader.
    It remains unchanged as its logic is standard.
    """

    def __init__(self, opt):
        self.dataset = Load_Dataset(opt)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=opt.batch_size,
                                                      shuffle=(opt.phase == 'train'),
                                                      pin_memory=True,
                                                      drop_last=(opt.phase == 'train'),
                                                      num_workers=int(opt.num_workers)
                                                      )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)