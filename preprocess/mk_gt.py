'''
make gt mask (for validation)
generates an image where the
pixel represent classes from
the xml file (not rgb, just class
codes)
'''
import openslide
import os
import numpy as np
from myargs import args
import glob
from utils.read_xml_sunnybrook import getGT, getTB
from utils.read_xml import getGT, getTB
from PIL import Image
from utils import preprocessing
from tqdm import tqdm
wsipaths = glob.glob('../{}/*.svs'.format(args.raw_val1_pth))

for wsipath in tqdm(sorted(wsipaths)):
    'read scan'
    filename = os.path.basename(wsipath)
    scan = openslide.OpenSlide(wsipath)
    'get actual mask, i.e. the ground truth'
    xmlpath = '../{}/{}.xml'.format(args.raw_val1_pth, filename.split('.svs')[0])
    gt = getGT(xmlpath, scan, level=args.scan_level)

    #tb = getTB(xmlpath, scan, level=args.scan_level)
    tb = getTB(gt, scan, level=args.scan_level)

    tb.save('../{}/{}_tumor_bed.png'.format(args.raw_val1_pth, filename))

    gt = Image.fromarray(gt.astype(np.uint8))

    if args.scan_resize != 1:
        x_rs, y_rs = int(gt.size[0] / args.scan_resize), int(gt.size[1] / args.scan_resize)
        gt = gt.resize((x_rs, y_rs))

    gt.save('../{}/{}_mask.png'.format(args.raw_val1_pth, filename))

    gt = np.array(gt)
    gt = 255*np.eye(args.num_classes)[gt][..., 1:].astype(np.uint8)
    gt = Image.fromarray(gt)
    gt.save('../{}/{}_mask_rgb.png'.format(args.raw_val1_pth, filename))

    wsi = scan.read_region((0, 0), 2, scan.level_dimensions[2])
    mask = preprocessing.find_nuclei(wsi)
    mask = Image.fromarray(mask.astype(np.uint8))
    mask.save('../{}/{}_find_nuclei.png'.format(args.raw_val1_pth, filename))

