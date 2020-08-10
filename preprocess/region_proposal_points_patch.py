from PIL import Image
import numpy as np
from myargs import args
import glob
import utils.filesystem as ufs
from tqdm import tqdm

' map class names to codes '
cls_codes = {
    'Normal': 0,
    'Benign': 1,
    'InSitu': 2,
    'Invasive': 3
}


args.patch_folder = '/home/ozan/ICIAR2018_BACH_Challenge/Photos_Train'

#args.train_hr_image_pth = 'data/val_hr'
args.patch_folder = '/home/ozan/ICIAR2018_BACH_Challenge/Photos_Val'

' check if metadata gt.npy already exists to append to it '
ufs.make_folder('../' + args.train_hr_image_pth, False)
metadata_pth = '../{}/gt.npy'.format(args.train_hr_image_pth)
metadata = ufs.fetch_metadata(metadata_pth)

metadata['P'] = {}
metadata['P'][0] = {}

cls_folders = glob.glob('{}/*/'.format(args.patch_folder))

index = 0

for cls_folder in tqdm(cls_folders):

    cls_name = cls_folder.split('/')[-2]
    cls_code = cls_codes[cls_name]

    image_paths = sorted(glob.glob('{}*.png'.format(cls_folder)))
    for image_path in image_paths:

        dimensions = Image.open(image_path).size

        metadata['P'][0][index] = {
            'cnt_xy': None,
            'perim_xy': None,
            'label': cls_code,
            'wsipath': image_path,
            'scan_level': None,
            'dimensions': dimensions
        }
        index = index + 1

np.save(metadata_pth, metadata)
