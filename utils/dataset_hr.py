from torch.utils import data
import numpy as np
import torch
import utils.preprocessing as preprocessing
import openslide
from myargs import args
import utils.filesystem as ufs
from pathlib import Path
from utils import regiontools
from PIL import Image
import copy
from torchvision import transforms

HR_NUM_CNT_SAMPLES = 8
HR_NUM_PERIM_SAMPLES = 8
HR_SCAN_LEVEL = 1
HR_PATCH_W = 64
HR_PATCH_H = 64


class Dataset(data.Dataset):
    def __init__(self, pth, eval, remove_white, duplicate_dataset):

        self.base_path = Path(__file__).parent
        metadata_pth = (self.base_path / '../{}/gt.npy'.format(pth)).resolve().as_posix()
        metadata = ufs.fetch_metadata(metadata_pth)

        '''
        dataset structure:
        dataset is comprised of patches+wsi regions.
        patches:
        metadata['P'] indicates where all the patches are.
        wsi:
        0. metadata[filename/svs file name]
        1. m[f][connected component id]
        2. m[f][c][region within the connected component]
        @ level 1, we have the connected component
        as given in gt mask. at this level m[f][c][0] 
        always points to the large region
        if the region is large enough, we then split it
        to smaller sub-regions at m[f][c][>=1].
        '''

        ' build the datalist '
        self.datalist = []
        cls = np.zeros(args.num_classes, )

        ' build patch portion of ds '
        if 'P' in metadata:
            P = copy.deepcopy(metadata['P'][0])
            del metadata['P']

            P_dims = {}
            for key in P:
                d = P[key]['dimensions']

                if d not in P_dims:
                    params = {
                        'num_center_points': HR_NUM_CNT_SAMPLES,
                        'num_perim_points': HR_NUM_PERIM_SAMPLES,
                        'scan_level': HR_SCAN_LEVEL,
                        'tile_w': HR_PATCH_W,
                        'tile_h': HR_PATCH_H,
                        'dimensions': d
                    }
                    params = preprocessing.DotDict(params)
                    P_dims[d] = regiontools.get_key_points_for_patch(params)

                item = {**P[key], **P_dims[d]}

                self.datalist.append(item)
                cls[item['label']] += 1

        ' build wsi regions portion '
        self.wsis = {}

        for filename in metadata:
            first_region_id = list(metadata[filename].keys())[0]
            first_sub_region_id = list(metadata[filename][first_region_id].keys())[0]
            pth = metadata[filename][first_region_id][first_sub_region_id]['wsipath']
            pth = (self.base_path / pth).resolve().as_posix()
            self.wsis[pth] = openslide.OpenSlide(pth)

            if remove_white:
                'get low res. nuclei image/foreground mask'
                scan = self.wsis[pth]
                x, y = scan.level_dimensions[-1]
                mask = scan.read_region((0, 0), scan.level_count-1, (x, y)).convert('RGB')
                mask = mask.resize((x//4, y//4))
                mask = preprocessing.find_nuclei(mask)
                mask = Image.fromarray(mask.astype(np.uint8)).resize((x, y))
                mask = np.asarray(mask)

            params = {
                'iw': self.wsis[pth].level_dimensions[0][0],
                'ih': self.wsis[pth].level_dimensions[0][1],
                'tile_w': HR_PATCH_W,
                'tile_h': HR_PATCH_H,
                'scan_level': metadata[filename][first_region_id][first_sub_region_id]['scan_level']
            }
            params = preprocessing.DotDict(params)

            for conncomp in metadata[filename]:
                for id in metadata[filename][conncomp]:
                    region_obj = metadata[filename][conncomp][id].copy()

                    if remove_white:
                        'given points, remove patches that are only white'
                        region_obj['cnt_xy'], num_cnt_pts = regiontools.remove_white_region(mask, region_obj['cnt_xy'], params)
                        region_obj['perim_xy'], num_perim_pts = regiontools.remove_white_region(mask, region_obj['perim_xy'], params)

                    'which points valid for this patch size, scan level combo?'
                    region_obj['cnt_xy'], num_cnt_pts = regiontools.map_points(region_obj['cnt_xy'], params)
                    region_obj['perim_xy'], num_perim_pts = regiontools.map_points(region_obj['perim_xy'], params)

                    if num_cnt_pts >= HR_NUM_CNT_SAMPLES and \
                            num_perim_pts >= HR_NUM_PERIM_SAMPLES:
                        self.datalist.append(region_obj)
                        cls[region_obj['label']] += 1

        self.eval = eval

        cls = np.array(cls)
        '''cls[0] += cls[1]
        cls[1] = cls[2]
        cls[2] = cls[3]
        cls[3] = 0'''

        print(cls)
        cls = cls / cls.sum()
        print(cls)
        if not self.eval:
            args.cls_ratios = cls

        ' augmentation settings '
        self.image_aug = preprocessing.standard_augmentor(self.eval)

        if not self.eval:
            from itertools import chain
            self.datalist = list(chain(*[[i] * duplicate_dataset for i in self.datalist]))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        wsipth = self.datalist[index]['wsipath']
        wsipth = (self.base_path / wsipth).resolve().as_posix()

        step = self.datalist[index]['cnt_xy'].shape[0] // HR_NUM_CNT_SAMPLES
        step = np.maximum(1, step)
        center_pts = self.datalist[index]['cnt_xy'][::step]

        step = self.datalist[index]['perim_xy'].shape[0] // HR_NUM_PERIM_SAMPLES
        step = np.maximum(1, step)
        perim_pts = self.datalist[index]['perim_xy'][::step]

        centers = np.vstack((perim_pts, center_pts)).astype(np.int)
        centers = centers[:(HR_NUM_CNT_SAMPLES+HR_NUM_PERIM_SAMPLES), :]
        remaining_items = (HR_NUM_CNT_SAMPLES+HR_NUM_PERIM_SAMPLES) - centers.shape[0]

        if remaining_items > 0:
            centers = np.vstack((centers, self.datalist[index]['perim_xy'][-remaining_items:, :]))

        label = self.datalist[index]['label']

        images = []

        '''
        import os
        import shutil
        savefolder = 'data/debug/{}_{}/'.format(index, self.datalist[index]['label'])
        if os.path.exists(savefolder):
            shutil.rmtree(savefolder)
        os.makedirs(savefolder, exist_ok=True)
        '''

        if 'dimensions' in self.datalist[index]:
            _image = Image.open(self.datalist[index]['wsipath'])
            __x, __y = _image.size
            ratio = 4 ** self.datalist[index]['scan_level']
            _image = _image.resize((__x // ratio, __y // ratio))

        for cj, (_x, _y) in enumerate(centers):

            'read in image&labels'
            if 'dimensions' in self.datalist[index]:
                image = _image.crop((_x, _y, _x+HR_PATCH_W, _y+HR_PATCH_H))
            else:
                image = self.wsis[wsipth].read_region((_x, _y), HR_SCAN_LEVEL, (HR_PATCH_W, HR_PATCH_H)).convert('RGB')

            # image.save('{}/{}.png'.format(savefolder, cj))

            'rotate image by 90*'
            degree = int(torch.randint(0, 4, (1,))) * 90
            image = image.rotate(degree)
            image = self.image_aug(image)

            images.append(image)

        images = torch.stack(images, 0)

        return images, label


def GenerateIterator(pth, eval=False, remove_white=False, duplicate_dataset=1):

    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.workers,
        'pin_memory': True,
    }

    return data.DataLoader(Dataset(pth, eval, remove_white, duplicate_dataset), **params)


class Dataset_eval(data.Dataset):

    def __init__(self, metadata, eval, remove_white):

        self.base_path = Path(__file__).parent
        first_region_id = list(metadata.keys())[0]
        pth = metadata[first_region_id]['wsipath']
        pth = (self.base_path / pth).resolve().as_posix()
        self.scan = openslide.OpenSlide(pth)

        if remove_white:
            'get low res. nuclei image/foreground mask'
            scan = self.wsis[pth]
            x, y = scan.level_dimensions[-1]
            mask = scan.read_region((0, 0), scan.level_count - 1, (x, y)).convert('RGB')
            mask = mask.resize((x // 4, y // 4))
            mask = preprocessing.find_nuclei(mask)
            mask = Image.fromarray(mask.astype(np.uint8)).resize((x, y))
            mask = np.asarray(mask)

        params = {
            'iw': self.scan.level_dimensions[0][0],
            'ih': self.scan.level_dimensions[0][1],
            'tile_w': HR_PATCH_W,
            'tile_h': HR_PATCH_H,
            'scan_level': metadata[first_region_id]['scan_level']
        }
        params = preprocessing.DotDict(params)

        ' build the datalist '
        self.datalist = []
        for key in metadata:
            region_obj = metadata[key].copy()

            if remove_white:
                'given points, remove patches that are only white'
                region_obj['cnt_xy'], num_cnt_pts = regiontools.remove_white_region(mask, region_obj['cnt_xy'], params)
                region_obj['perim_xy'], num_perim_pts = regiontools.remove_white_region(mask, region_obj['perim_xy'],
                                                                                        params)
            region_obj['cnt_xy'], num_cnt_pts = regiontools.map_points(region_obj['cnt_xy'], params)
            region_obj['perim_xy'], num_perim_pts = regiontools.map_points(region_obj['perim_xy'], params)

            if num_cnt_pts >= HR_NUM_CNT_SAMPLES and \
                    num_perim_pts >= HR_NUM_PERIM_SAMPLES:
                self.datalist.append(region_obj)

        self.eval = eval

        ' augmentation settings '
        self.image_aug = preprocessing.standard_augmentor(self.eval)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):

        center_pts = self.datalist[index]['cnt_xy'][:HR_NUM_CNT_SAMPLES]
        perim_pts = self.datalist[index]['perim_xy'][:HR_NUM_PERIM_SAMPLES]
        centers = np.vstack((perim_pts, center_pts)).astype(np.int)

        images = []

        # degree = int(torch.randint(0, 4, (1,))) * 90
        for cj, (_x, _y) in enumerate(centers):

            'read in image&labels'
            image = self.scan.read_region((_x, _y), HR_SCAN_LEVEL, (HR_PATCH_W, HR_PATCH_H)).convert('RGB')

            'rotate image by 90*'
            # image = image.rotate(degree)
            image = self.image_aug(image)

            images.append(image)

        images = torch.stack(images, 0)

        return images, self.datalist[index]['tile_id']


def GenerateIterator_eval(metadata, eval=True, remove_white=False):

    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.workers,
        'pin_memory': True,
    }

    return data.DataLoader(Dataset_eval(metadata, eval, remove_white), **params)
