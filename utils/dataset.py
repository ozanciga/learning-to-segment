from torch.utils import data
import numpy as np
from PIL import Image
import torch
import utils.preprocessing as preprocessing
import openslide
from myargs import args
import glob
import os
from tqdm import tqdm
# import staintools
import random
from sklearn.model_selection import train_test_split

class Dataset(data.Dataset):
    def __init__(self, impth, eval, duplicate_dataset, use_seg_ratio, use_cls):

        self.eval = eval

        ' build the dataset '
        self.datalist = []
        gt = np.load('{}/gt.npy'.format(impth), allow_pickle=True).flatten()[0]
        for key in gt:
            self.datalist.append([{
                'wsi': gt[key][tile_id]['wsi'],
                'mask': gt[key][tile_id]['mask'] if 'mask' in gt[key][tile_id] else None,
                'label': gt[key][tile_id]['label'],
            } for tile_id in gt[key]])
        self.datalist = [item for sublist in self.datalist for item in sublist]

        random.Random(10 * args.use_seg_ratio + args.exp_iter).shuffle(self.datalist)

        labels_seg = np.array([(index, item['label']) for index, item in enumerate(self.datalist) if item['mask'] is not None])
        labels_cls = np.array([(index, item['label']) for index, item in enumerate(self.datalist) if item['mask'] is None])

        new_datalist = []

        for cj in range(args.num_classes):

            all_indices_seg = labels_seg[labels_seg[:, 1] == cj][:, 0]
            labels_seg_max = np.maximum(1, int(use_seg_ratio * len(all_indices_seg)))
            keep_indices_seg = all_indices_seg[:labels_seg_max]
            for index in keep_indices_seg:
                new_datalist.append(self.datalist[index])

            if use_cls:
                all_indices_cls = labels_cls[labels_cls[:, 1] == cj][:, 0]
                labels_cls_max = np.maximum(1, int((1-use_seg_ratio) * len(all_indices_cls))) if use_cls < 2 else 99999
                keep_indices_cls = all_indices_cls[:labels_cls_max]
                for index in keep_indices_cls:
                    new_datalist.append(self.datalist[index])

        self.datalist = new_datalist

        self.seg_pths = [item['wsi'] for item in self.datalist if item['mask'] is not None]

        if not self.eval:

            args.dataset_mean = [0, 0, 0]
            args.dataset_std = [0, 0, 0]

            datalist_seg = [item for item in self.datalist if item['mask'] is not None]
            for item in datalist_seg:
                data = np.array(Image.open(item['wsi']))
                args.dataset_mean += data.mean((0, 1)) / 255
            args.dataset_mean /= len(datalist_seg)

            for item in datalist_seg:
                data = np.array(Image.open(item['wsi']))
                args.dataset_std += ((data / 255 - args.dataset_mean) ** 2).sum((0, 1))
            args.dataset_std = np.sqrt(args.dataset_std / (len(datalist_seg) * args.tile_w * args.tile_h))

        ' augmentation settings '
        self.image_aug = preprocessing.standard_augmentor(eval)

        if not self.eval:
            from itertools import chain
            self.datalist = list(chain(*[[i] * duplicate_dataset for i in self.datalist]))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        'read in image&labels'
        image = Image.open(self.datalist[index]['wsi'])
        label = self.datalist[index]['label']

        is_cls = self.datalist[index]['mask'] is None

        if is_cls:
            mask = label * np.ones((args.tile_h, args.tile_w), dtype=np.uint8)
            mask = Image.fromarray(mask)

            #mask = label * np.ones((128, 128), dtype=np.uint8)
            #mask = np.pad(mask, ((args.tile_h-128)//2, (args.tile_w-128)//2), mode='constant')
            #mask = Image.fromarray(mask)

        else:
            mask = Image.open(self.datalist[index]['mask'])

        #if ~is_cls:
        #    image = preprocessing.stain_normalize(image)

        if not self.eval:
            'rotate image by 90*'
            degree = int(torch.randint(0, 4, (1, ))) * 90

            image = image.rotate(degree)
            mask = mask.rotate(degree)

        mask = np.asarray(mask)

        image = self.image_aug(image)
        mask = torch.from_numpy(mask.astype(np.uint8)).long()

        return image, label, mask, is_cls


class Dataset_onlyseg(data.Dataset):

    def __init__(self, impth, eval, use_seg_ratio, use_cls):
        '''
        use seg images as cls labels
        '''

        self.eval = eval

        ' build the dataset '
        self.datalist = []
        gt = np.load('{}/gt.npy'.format(impth), allow_pickle=True).flatten()[0]
        for key in gt:
            self.datalist.append([{
                'wsi': gt[key][tile_id]['wsi'],
                'mask': gt[key][tile_id]['mask'],
                'label': gt[key][tile_id]['label'],
            } for tile_id in gt[key]])

        X_cls_patches = [item for sublist in self.datalist for item in sublist if item['mask'] is None]

        X_seg_patches = [item for sublist in self.datalist for item in sublist if item['mask'] is not None]

        cls_patches, seg_patches = [], []

        if not eval:

            if use_cls < 3:
                random.Random(10 * args.use_seg_ratio + args.exp_iter).shuffle(X_seg_patches)
                seg_patches = X_seg_patches[:int(use_seg_ratio * len(X_seg_patches))] \
                    if use_seg_ratio > 0 else X_seg_patches[:1]
            elif use_cls == 3:
                seg_patches = X_seg_patches

            if use_cls == 1 or use_cls == 3:
                random.Random(10 * args.use_seg_ratio + args.exp_iter).shuffle(X_cls_patches)
                cls_patches = X_cls_patches[:int((1-use_seg_ratio) * len(X_cls_patches))]
            elif use_cls == 2:
                cls_patches = X_cls_patches
        else:
            cls_patches, seg_patches = X_cls_patches, X_seg_patches

        #print(len(cls_patches), len(seg_patches))

        self.datalist = seg_patches

        if use_cls > 0:
            self.datalist.extend(cls_patches)

        ' augmentation settings '
        self.image_aug = preprocessing.standard_augmentor(eval)

    def __len__(self):
        if self.eval:
            return len(self.datalist)

        return len(self.datalist) * 100  # * self.multiplier

    def __getitem__(self, index):

        # return torch.rand((3, 224, 224)), torch.randint(0, 4,(1,)).item(), torch.randint(0,4, (224, 224)).long(), torch.randint(0,2, (1,)).item()

        index = index % len(self.datalist)

        'read in image&labels'
        image = Image.open(self.datalist[index]['wsi'])
        label = self.datalist[index]['label']

        is_cls = self.datalist[index]['mask'] is None

        if is_cls:
            mask = label * np.ones((args.tile_h, args.tile_w), dtype=np.uint8)
            mask = Image.fromarray(mask)

            #mask = label * np.ones((128, 128), dtype=np.uint8)
            #mask = np.pad(mask, ((args.tile_h-128)//2, (args.tile_w-128)//2), mode='constant')
            #mask = Image.fromarray(mask)

        else:
            mask = Image.open(self.datalist[index]['mask'])

        #if ~is_cls:
        #    image = preprocessing.stain_normalize(image)

        if not self.eval:
            'rotate image by 90*'
            degree = int(torch.randint(0, 4, (1, ))) * 90

            image = image.rotate(degree)
            mask = mask.rotate(degree)

        #image = image.resize((96, 96))
        #mask = mask.resize((96, 96))

        mask = np.asarray(mask)

        image = self.image_aug(image)
        mask = torch.from_numpy(mask.astype(np.uint8)).long()

        return image, label, mask, is_cls


def GenerateIterator(impth, eval=False, duplicate_dataset=1, use_seg_ratio=1, use_cls=True):

    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.workers,
        'pin_memory': True,
    }

    return data.DataLoader(Dataset(impth, eval=eval, duplicate_dataset=duplicate_dataset, use_seg_ratio=use_seg_ratio, use_cls=use_cls), **params)


class Dataset_wsis:
    ' all validation wsis '
    def __init__(self, svs_pth, params, bs=args.batch_size):

        self.params = preprocessing.DotDict(params)
        self.wsis = {}

        wsipaths = glob.glob('{}/*.svs'.format(svs_pth))
        with tqdm(enumerate(sorted(wsipaths))) as t:
            for wj, wsipath in t:
                t.set_description('Loading wsis.. {:d}/{:d}'.format(1 + wj, len(wsipaths)))

                filename = os.path.basename(wsipath)
                scan = openslide.OpenSlide(wsipath)
                itr = GenerateIterator_wsi(wsipath, self.params, bs)

                msk_pth = '{}/{}.png'.format(args.wsi_mask_pth, filename)

                if itr is not None:
                    self.wsis[filename] = {
                        'iterator': itr,
                        'wsipath': wsipath,
                        'scan': scan,
                        'maskpath': msk_pth
                    }


class Dataset_wsi(data.Dataset):
    ' use a wsi image to create the dataset '
    def __init__(self, wsipth, params):

        self.params = params

        ' build the dataset '
        self.datalist = []

        'read the wsi scan'
        filename = os.path.basename(wsipth)
        self.scan = openslide.OpenSlide(wsipth)

        ' if a slide has less levels than our desired scan level, ignore the slide'
        if self.scan.level_count >= 3 and \
                (self.scan.level_dimensions[args.scan_level][0] / args.scan_resize >= args.tile_w) and \
                (self.scan.level_dimensions[args.scan_level][1] / args.scan_resize >= args.tile_h):

            self.params.iw, self.params.ih = self.scan.level_dimensions[args.scan_level]

            'gt mask'
            'find nuclei is slow, hence save masks from preprocessing' \
            'for later use'
            msk_pth = '{}/{}.png'.format(args.wsi_mask_pth, filename)
            if not os.path.exists(msk_pth):
                thmb = self.scan.read_region((0, 0), 2, self.scan.level_dimensions[2]).convert('RGB')
                mask = preprocessing.find_nuclei(thmb, fill_mask=True)
                Image.fromarray(mask.astype(np.uint8)).save(msk_pth)
            else:
                mask = Image.open(msk_pth).convert('L')
                mask = np.asarray(mask)

            ' augmentation settings '
            self.image_aug = preprocessing.standard_augmentor(True)

            'downsample multiplier'
            m = self.scan.level_downsamples[args.scan_level]/self.scan.level_downsamples[2]
            dx, dy = int(self.params.pw * m), int(self.params.ph * m)

            for ypos in range(1, self.params.ih - 1 - self.params.ph, self.params.sh):
                for xpos in range(1, self.params.iw - 1 - self.params.pw, self.params.sw):
                    yp, xp = int(ypos * m), int(xpos * m)
                    if not preprocessing.isforeground(mask[yp:yp+dy, xp:xp+dx]):
                        continue
                    # self.datalist.append((xpos, ypos))
                    self.datalist.append((xpos, ypos, self.im_reader(xpos, ypos)))

            xpos = self.params.iw - 1 - self.params.pw
            for ypos in range(1, self.params.ih - 1 - self.params.ph, self.params.sh):
                yp, xp = int(ypos * m), int(xpos * m)
                if not preprocessing.isforeground(mask[yp:yp + dy, xp:xp + dx]):
                    continue
                # self.datalist.append((xpos, ypos))
                self.datalist.append((xpos, ypos, self.im_reader(xpos, ypos)))

            ypos = self.params.ih - 1 - self.params.ph
            for xpos in range(1, self.params.iw - 1 - self.params.pw, self.params.sw):
                yp, xp = int(ypos * m), int(xpos * m)
                if not preprocessing.isforeground(mask[yp:yp + dy, xp:xp + dx]):
                    continue
                # self.datalist.append((xpos, ypos))
                self.datalist.append((xpos, ypos, self.im_reader(xpos, ypos)))

    def im_reader(self, x, y):

        'get top left corner'
        _x = int(self.scan.level_downsamples[args.scan_level] * x)
        _y = int(self.scan.level_downsamples[args.scan_level] * y)
        'read in image'
        image = self.scan.read_region((_x, _y), args.scan_level, (self.params.pw, self.params.ph)).convert('RGB')
        if args.scan_resize != 1:
            image = image.resize((args.tile_w, args.tile_h))

        # image = preprocessing.stain_normalize(image)
        image = self.image_aug(image)

        return image

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        xpos, ypos, image = self.datalist[index]

        return float(xpos), float(ypos), image


def GenerateIterator_wsi(wsipth, p, bs):

    params = {
        'batch_size': bs,
        'shuffle': True,
        'num_workers': args.workers,
        'pin_memory': True,
    }

    dataset = Dataset_wsi(wsipth, p)
    if len(dataset) > 0:
        return data.DataLoader(dataset, **params)

    return None


class Dataset_cls(data.Dataset):

    def __init__(self, impth, eval, duplicate_dataset):

        self.eval = eval
        ' augmentation settings '
        self.image_aug = preprocessing.standard_augmentor(eval)

        ' build the dataset '
        self.datalist = []
        gt = np.load('{}/gt.npy'.format(impth), allow_pickle=True).flatten()[0]
        for key in gt:
            self.datalist.append([{
                'wsi': gt[key][tile_id]['wsi'],
                'label': gt[key][tile_id]['label'],
            } for tile_id in gt[key]])
        self.datalist = [item for sublist in self.datalist for item in sublist]

        if not self.eval:
            from itertools import chain
            self.datalist = list(chain(*[[i] * duplicate_dataset for i in self.datalist]))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        'read in image&labels'

        image = Image.open(self.datalist[index]['wsi'])
        label = self.datalist[index]['label']

        if not self.eval:
            'rotate image by 90*'
            degree = int(torch.randint(0, 4, (1,))) * 90

            image = image.rotate(degree, expand=True)
            image = image.resize((args.tile_w, args.tile_h))

        label = np.asarray(label)

        image = self.image_aug(image)
        label = torch.from_numpy(label.astype(np.uint8)).long()

        return image, label


def GenerateIterator_cls(impth, eval=False, duplicate_dataset=1):
    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.workers,
        'pin_memory': True,
    }

    return data.DataLoader(Dataset_cls(impth, eval=eval, duplicate_dataset=duplicate_dataset), **params)
