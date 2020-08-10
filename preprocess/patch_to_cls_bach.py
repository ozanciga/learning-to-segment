'''

original preprocessing is
done on wsi's (svs)
given patches, this generates
gt mask.
this script is specifically written
for bach iciar 2018 dataset
(part a images)
run by: python patch_to_gt.py --patch_folder your_path/
args.patch_folder = '/home/ozan/ICIAR2018_BACH_Challenge/Photos'

'''
from tqdm import tqdm
import os
import numpy as np
from myargs import args
import glob
import utils.filesystem as ufs
from PIL import Image
from utils import preprocessing

args.patch_folder = '/home/ozan/PycharmProjects/Photos'

istrain = True
num_train = 100

savepath = args.train_image_pth if istrain else args.val_image_pth

if __name__ == '__main__':

    ufs.make_folder('../' + savepath, False)

    ' map class names to codes '
    cls_codes = {
        'Normal': 0,
        'Benign': 1,
        'InSitu': 2,
        'Invasive': 3
    }

    ' check if metadata gt.npy already exists to append to it '
    metadata_pth = '../{}/gt.npy'.format(savepath)
    metadata = ufs.fetch_metadata(metadata_pth)

    cls_folders = glob.glob('{}/*/'.format(args.patch_folder))

    gt = Image.fromarray(np.zeros((args.tile_h, args.tile_w), dtype=np.uint8))

    for cls_folder in tqdm(cls_folders):
        cls_name = cls_folder.split('/')[-2]
        cls_code = cls_codes[cls_name]

        image_paths = sorted(glob.glob('{}*.tif'.format(cls_folder)))
        for ij, image_path in enumerate(image_paths):

            if istrain and ij >= num_train:
                break
            elif not istrain and ij < num_train:
                continue

            filename = os.path.basename(image_path)
            metadata[filename] = {}

            image = Image.open(image_path).convert('RGB')
            image = image.resize((args.tile_h, args.tile_w))
            image = preprocessing.stain_normalize(image)

            '''repeat method'''
            '''image = image.resize((image.size[0]//16, image.size[1]//16))
            image = preprocessing.stain_normalize(image)

            image = np.array(image)
            y, x = image.shape[:2]
            xmax = np.ceil(args.tile_w / x).astype(np.int)
            ymax = np.ceil(args.tile_h / y).astype(np.int)
            image_new = np.zeros((ymax * y, xmax * x, 3), dtype=np.uint8)

            for xmul in range(xmax):
                for ymul in range(ymax):
                    xpos, ypos = xmul * x, ymul * y
                    image_new[ypos:ypos+y, xpos:xpos+x, ...] = image

            image = Image.fromarray(image_new).crop((0, 0, args.tile_w, args.tile_h))
            '''


            '''image = image.resize((128, 128))
            image = preprocessing.stain_normalize(image)
            image = np.array(image)
            image_new = np.zeros((args.tile_h, args.tile_w, 3), dtype=np.uint8)
            for ij in range(image.shape[2]):
                image_new[..., ij] = np.pad(image[..., ij], ((args.tile_h-image.shape[0])//2, (args.tile_w-image.shape[1])//2), mode='constant')
            image = Image.fromarray(image_new)'''

            'save paths'
            tilepth_w = '{}/w_{}_{}.png'.format(savepath, filename, 0)
            tilepth_g = '{}/g_{}_{}.png'.format(savepath, filename, 0)

            ' save metadata '
            metadata[filename][0] = {
                'wsi': tilepth_w,
                'label': cls_code,
            }

            ' save images '
            image.save('../' + tilepth_w)

    np.save('../{}/gt.npy'.format(savepath), metadata)
