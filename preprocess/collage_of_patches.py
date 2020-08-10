'''
make a very large 2d image from
patches
'''
from tqdm import tqdm
import os
import numpy as np
from myargs import args
import glob
import utils.preprocessing as preprocessing
import utils.filesystem as ufs
from PIL import Image


def gallery(array, ncols):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height*nrows, width * ncols, intensity))
    return result


args.patch_folder = '/home/ozan/ICIAR2018_BACH_Challenge/Photos'

if __name__ == '__main__':

    ufs.make_folder('../' + args.train_image_pth, False)

    ' map class names to codes '
    cls_codes = {
        'Normal': 0,
        'Benign': 1,
        'InSitu': 2,
        'Invasive': 3
    }

    ' check if metadata gt.npy already exists to append to it '
    metadata_pth = '../{}/gt.npy'.format(args.train_image_pth)
    metadata = ufs.fetch_metadata(metadata_pth)

    cls_folders = glob.glob('{}/*/'.format(args.patch_folder))

    index = 0

    yy, xx = 1536//(args.scan_resize*4**args.scan_level), 2048//(args.scan_resize*4**args.scan_level)

    images = np.zeros((100 * 4, yy, xx, 3), dtype=np.uint8)
    gts = np.zeros((100 * 4, yy, xx), dtype=np.uint8)

    for cls_folder in tqdm(cls_folders):
        cls_name = cls_folder.split('/')[-2]
        cls_code = cls_codes[cls_name]

        image_paths = sorted(glob.glob('{}*.png'.format(cls_folder)))
        for image_path in image_paths:

            filename = os.path.basename(image_path)
            metadata[filename] = {}

            image = Image.open(image_path)

            x, y = image.size
            x, y = x // (args.scan_resize*4**args.scan_level), y // (args.scan_resize*4**args.scan_level)

            image = image.resize((x, y))
            images[index, ...] = image
            gts[index, ...] = cls_code
            index = index + 1

    'use patches to generate collage of patches'
    num_repeat = 1
    indices = np.concatenate([np.random.permutation(images.shape[0]) for _ in range(num_repeat)])
    collage_images = Image.fromarray(gallery(images[indices, ...],  num_repeat*10).astype(np.uint8))
    collage_gts = gallery(np.expand_dims(gts[indices, ...], -1).astype(np.uint8), num_repeat*10)[..., 0]
    collage_gts = Image.fromarray(collage_gts)

    mask = np.zeros((args.tile_h, args.tile_w), dtype=np.uint8)
    mask = Image.fromarray(mask)

    params = {
        'ih': collage_images.size[1],
        'iw': collage_images.size[0],
        'ph': args.tile_h,
        'pw': args.tile_w,
        'sh': args.tile_stride_h,
        'sw': args.tile_stride_w,
    }

    filename = 'collage_of_patches'
    metadata[filename] = {}

    for tile_id, (tile_w, tile_g) in enumerate(zip(
            preprocessing.tile_image(collage_images, params),
            preprocessing.tile_image(collage_gts, params))):

        'save everything'
        tilepth_w = '{}/w_{}_{}.png'.format(args.train_image_pth, filename, tile_id)
        tilepth_g = '{}/g_{}_{}.png'.format(args.train_image_pth, filename, tile_id)
        tilepth_m = '{}/m_{}_{}.png'.format(args.train_image_pth, filename, tile_id)

        ' save metadata '
        metadata[filename][tile_id] = {
            'wsi': tilepth_w,
            'label': tilepth_g,
            'mask': tilepth_m,
        }
        ' save images '
        tile_w[-1].save('../' + tilepth_w)
        tile_g[-1].save('../' + tilepth_g)
        mask.save('../' + tilepth_m)

    np.save(metadata_pth, metadata)
