from torch.utils import data
import numpy as np
from PIL import Image
import torch
import utils.preprocessing as preprocessing
import openslide
from myargs import args
import glob
import os
from torchvision import transforms
import utils.filesystem as ufs

class Dataset(data.Dataset):
    def __init__(self, impth, eval):

        self.eval = eval
        ' augmentation settings '
        self.image_aug = preprocessing.standard_augmentor(eval)

        ' build the dataset '
        self.datalist = []
        impths = glob.glob('{}/*_image.png'.format(impth))
        for pth in impths:
            item = {
                'image': pth,
                'label': pth.replace('_image.png', '_gt.png')
            }
            self.datalist.append(item)

        if not self.eval:
            from itertools import chain
            self.datalist = list(chain(*[[i] * 10 for i in self.datalist]))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        'read in image&labels'
        image = Image.open(self.datalist[index]['image'])
        label = Image.open(self.datalist[index]['label'])

        'rotate image by 90*'
        degree = int(torch.randint(0, 4, (1, ))) * 90
        image = image.rotate(degree)
        label = label.rotate(degree)
        image = image.resize((512, 512))
        label = label.resize((512, 512))

        label = np.array(label)
        label = np.concatenate((np.zeros(shape=(*label.shape[:2], 1)), label), -1)
        label = np.argmax(label, -1)

        image = self.image_aug(image)
        label = torch.from_numpy(label.astype(np.uint8)).long()

        return image, label


def GenerateIterator(impth, eval=False):

    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.workers,
        'pin_memory': True,
    }

    return data.DataLoader(Dataset(impth, eval=eval), **params)


class Dataset_cls(data.Dataset):
    def __init__(self, impth, eval):

        self.eval = eval
        ' augmentation settings '
        self.image_aug = preprocessing.standard_augmentor(eval)

        ' build the dataset '
        self.datalist = []
        gt = np.load('{}/gt.npy'.format(impth), allow_pickle=True).flatten()[0]
        for key in gt:
            self.datalist.append([{
                'image': ufs.fix_path(gt[key][tile_id]['image']),
                'label': gt[key][tile_id]['label'],
            } for tile_id in gt[key]])
        self.datalist = [item for sublist in self.datalist for item in sublist]

        if not self.eval:
            from itertools import chain
            self.datalist = list(chain(*[[i] * 10 for i in self.datalist]))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        'read in image&labels'
        image = Image.open(self.datalist[index]['image'])

        'rotate image by 90*'
        degree = int(torch.randint(0, 4, (1, ))) * 90
        image = image.rotate(degree)

        image = self.image_aug(image)

        return image, self.datalist[index]['label']


def GenerateIterator_cls(impth, eval=False):

    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.workers,
        'pin_memory': True,
    }

    return data.DataLoader(Dataset_cls(impth, eval=eval), **params)


