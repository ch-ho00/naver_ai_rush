import os
from pathlib import Path
import shutil
from tempfile import mkdtemp
from typing import Tuple
from warnings import warn

import pandas as pd
from nsml.constants import DATASET_PATH
from torch.utils import data
import torchvision.transforms.functional as TF
import torchvision
import torch

import numpy as np 
import PIL

UNLABELED = -1 
CLASS2LABEL = {class_ : label for class_, label in zip(['normal', 'monotone', 'screenshot', 'unknown'],[0,1,2,3,-1])}

class Spam_img_dataset(data.Dataset):
    def __init__(self, classes=['normal', 'monotone', 'screenshot', 'unknown'], input_size=[256,256,3], partition=None, transform=None, mode='train', base_dir=None, rgb_mean=[0,0,0], rgb_std=[1,1,1], num_imgs_per_class={}):
        self.partition = partition
        self.classes = classes
        self.input_size = input_size
        self.base_dir = Path(mkdtemp()) if base_dir is None else base_dir
        self._len = None
        self.num_imgs_per_class = num_imgs_per_class
        self.mode = mode
        if transform == None:
            self.transforms = torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(rgb_mean, rgb_std)
            ])
        else:
            self.transforms = transform
        self.validation_fraction = 0.2


    def __getitem__(self,idx):
        if self.mode == 'train' or self.mode == 'val':
            idx2 = self.convert_idx(idx)
            dir_ = self.base_dir / 'train' / idx2[1] 
            file_dir = dir_ / os.listdir(dir_)[idx2[0]]

            img = PIL.Image.open(open(file_dir, 'rb'))
            if self.transforms:
                img = self.transforms(img)
            
            return img, torch.Tensor([CLASS2LABEL[idx2[1]]])
        
        elif self.mode == 'test':
            dir_ = os.listdir(self.base_dir)[idx]
            img = TF.to_tensor(PIL.Image.open(open(dir_, 'rb')))
            # if self.transforms:
            #     img = self.transforms(img)
            return img

    def convert_idx(self, idx):
        sum_ = 0
        for class_ in self.num_imgs_per_class:
            if idx < sum_ + self.num_imgs_per_class[class_]:
                idx -= sum_
                return [idx, class_]
            sum_ += self.num_imgs_per_class[class_]
            
    # def __del__(self):
    #     """
    #     Deletes the temporary folder that we created for the dataset.
    #     """
    #     shutil.rmtree(self.base_dir)

    def len(self, dataset):
        """
        Utility function to compute the number of datapoints in a given dataset.
        """
        # os.walk returns list of tuples containing list (directiory, [folders], [files])
        if self._len is None:
            self._len = {
                dataset: sum([len(files) for r, d, files in os.walk(self.base_dir / dataset)]) for dataset in
                ['train']}
            self._len['train'] = int(self._len['train'] * (1 - self.validation_fraction))
            self._len['val'] = int(self._len['train'] * self.validation_fraction)
        return self._len[dataset]

    def prepare(self):
        """
        The resulting folder structure is compatible with the Keras function that generates a dataset from folders.
        """
        dataset = 'train'
        self._initialize_directory(dataset)
        self._rearrange(dataset)

    def _initialize_directory(self, dataset: str) -> None:
        """
        Initialized directory structure for a given dataset, in a way so that it's compatible with the Keras dataloader.
        """
        dataset_path = self.base_dir / dataset
        dataset_path.mkdir()
        for c in self.classes:
            (dataset_path / c).mkdir()

    def _rearrange(self, dataset: str) -> None:
        """
        Then rearranges the files based on the attached metadata. The resulting format is
        --
         |-train
             |-normal
                 |-img0
                 |-img1
                 ...
             |-montone
                 ...
             |-screenshot
                 ...
             |_unknown
                 ...
        """
        output_dir = self.base_dir / dataset
        src_dir = Path(DATASET_PATH) / dataset
        metadata = pd.read_csv(src_dir / f'{dataset}_label')
        self.num_imgs_per_class = {}
        for class_, label in zip(self.classes, [0,1,2,3,-1]):
            self.num_imgs_per_class[class_] = metadata[metadata['annotation'] == label].shape[0]

        for _, row in metadata.iterrows():
            if row['annotation'] == UNLABELED:
                continue
            src = src_dir / 'train_data' / row['filename']
            if not src.exists():
                raise FileNotFoundError
            dst = output_dir / self.classes[row['annotation']] / row['filename']
            if dst.exists():
                warn(f'File {src} already exists, this should not happen. Please notify 서동필 or 방지환.')
            else:
                shutil.copy(src=src, dst=dst)

    def train_val_gen(self,batch_size: int):
        '''
        Splits the train_data folder into train/val generators. Applies some image augmentation for the train dataset.

        Args:
            batch_size: int

        Returns:
            train_generator: Pytorch dataloader.
            val_generator: Pytorch dataloader.
        '''        
        num_total = self.len('train') + self.len('val') 
        split_num = self.len('train')

        train_idx = np.random.choice(range(num_total), split_num, replace=False)
        val_idx = list(set(range(num_total)) - set(list(train_idx)))

        # oversampling
        # screenshot_ratio = 0.2
        # unknown_ratio = 0.2
        # monotone_ratio = 0.2
        # normal_ratio = 0.4
        # assert unknown_ratio + screenshot_ratio + normal_ratio + monotone_ratio == 1
        # sum_ = 0
        # for i,class_ in enumerate(self.num_imgs_per_class):
        #     prev = sum_
        #     sum_ += self.num_imgs_per_class[class_]
        #     if i == 0:
        #         train_idx = np.random.choice(range(prev, sum_), int(num_total * normal_ratio), replace=False)
        #     elif i == 1:
        #         add = np.random.choice(range(prev, sum_ - int((sum_ - prev) * 0.2)), int(num_total * monotone_ratio), replace=True)
        #         train_idx = np.concatenate([train_idx, add], axis=0)
        #     elif i == 2:
        #         add = np.random.choice(range(prev, sum_- int((sum_ - prev) * 0.2)), int(num_total * screenshot_ratio), replace=True)
        #         train_idx = np.concatenate([train_idx, add], axis=0)
        #     elif i == 3:
        #         add = np.random.choice(range(prev, sum_- int((sum_ - prev) * 0.2)), int(num_total * unknown_ratio), replace=True)
        #         train_idx = np.concatenate([train_idx, add], axis=0)

        # val_idx = list(set(range(num_total)) - set(list(train_idx)))


        # for test
        # train_idx = np.random.choice(range(num_total), 1000, replace=False)
        # val_idx = np.random.choice(range(num_total), 1000, replace=False)


        train_sampler = data.SubsetRandomSampler(train_idx)
        val_sampler = data.SubsetRandomSampler(val_idx)

        partition = {}
        partition['train'] = train_idx
        partition['validation'] = val_idx

        params = {'batch_size': batch_size,
                'shuffle': False,
                'num_workers': 2}
        # dataloader
        training_set = Spam_img_dataset(partition=partition['train'],classes=self.classes, input_size=self.input_size, transform=self.transforms,mode='train',num_imgs_per_class=self.num_imgs_per_class, base_dir=self.base_dir)
        train_loader = data.DataLoader(training_set,sampler=train_sampler, **params)

        validation_set = Spam_img_dataset(partition=partition['validation'],classes=self.classes, input_size=self.input_size, transform=self.transforms,mode='val',num_imgs_per_class=self.num_imgs_per_class,base_dir=self.base_dir)
        val_loader = data.DataLoader(validation_set, sampler=val_sampler, **params)
        
        return train_loader, val_loader

    def test_gen(self, test_dir: str, batch_size: int):
        files = [str(p.name) for p in (Path(test_dir) / 'test_data').glob('*.*') if p.suffix not in ['.gif', '.GIF']]

        test_data = Spam_img_dataset(partition=list(range(len(files))), base_dir=Path(test_dir) / 'test_data', mode='test')
        test_data_loader = data.DataLoader(test_data, batch_size=batch_size, shuffle=False,  num_workers=2)

        return gen, files

    def __len__(self):
        return len(self.partition)