import staintools
from PIL import Image
import os
import utils.filesystem as ufs
import numpy as np
from myargs import args
import glob
from tqdm import tqdm

num_iters = 4
args.patch_folder = '/home/ozan/PycharmProjects/wsi-segmentation-pipeline/data/bach/patches/Photos'

datalist = []
gt = np.load('{}/gt.npy'.format(args.train_image_pth), allow_pickle=True).flatten()[0]
for key in gt:
    datalist.append([{
        'wsi': gt[key][tile_id]['wsi'],
    } for tile_id in gt[key]])
datalist = [item for sublist in datalist for item in sublist]

'stain normalize images'
sn = []
for item in datalist:
    try:
        ref = np.array(Image.open(item['wsi']))
        #ref = staintools.LuminosityStandardizer.standardize(ref)
        stain_normalizer = staintools.StainNormalizer(method='macenko')
        stain_normalizer.fit(ref)
        sn.append(stain_normalizer)
    except:
        pass

ufs.make_folder('../' + args.train_image_pth, False)

' map class names to codes '
cls_codes = {
    'Normal': 0,
    'Benign': 1,
    'InSitu': 2,
    'Invasive': 3
}

' check if metadata gt.npy already exists to append to it '
metadata_pth = '{}/gt.npy'.format(args.train_image_pth)
metadata = ufs.fetch_metadata(metadata_pth)

cls_folders = glob.glob('{}/*/'.format(args.patch_folder))

gt = Image.fromarray(np.zeros((args.tile_h, args.tile_w), dtype=np.uint8))

for cls_folder in tqdm(cls_folders):
    cls_name = cls_folder.split('/')[-2]
    cls_code = cls_codes[cls_name]

    image_paths = sorted(glob.glob('{}*.png'.format(cls_folder)))
    for image_path in image_paths:

        filename = os.path.basename(image_path)
        metadata[filename] = {}

        image = Image.open(image_path).convert('RGB')
        image = image.resize((args.tile_h, args.tile_w))
        to_transform = np.array(image)
        to_transform = staintools.LuminosityStandardizer.standardize(to_transform)

        for it in range(num_iters):
            'save paths'
            tilepth_w = '{}/w_{}_{}.png'.format(args.train_image_pth, filename, it)
            tilepth_g = '{}/g_{}_{}.png'.format(args.train_image_pth, filename, it)

            ' save metadata '
            metadata[filename][it] = {
                'wsi': tilepth_w,
                'label': cls_code,
            }

            ' save images '
            ridx = np.random.randint(0, len(sn))
            transformed = sn[ridx].transform(to_transform)
            image_f = Image.fromarray(transformed)
            image_f.save(tilepth_w)

np.save('{}/gt.npy'.format(args.train_image_pth), metadata)




ref_pth = "data/train/w_A01.svs_55.png"
targ_pth = "data/train/w_iv084.png_0.png"

ref = np.array(Image.open(ref_pth))
to_transform = np.array(Image.open(targ_pth))
# Stain normalize
normalizer = staintools.StainNormalizer(method='macenko')
normalizer.fit(ref)
transformed = normalizer.transform(to_transform)
transformed = Image.fromarray(transformed)
