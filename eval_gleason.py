'''
evaluate trained models on images
for gleason2019 and digestpath2019 datasets..
'''

import torch
from myargs import args
from tqdm import tqdm
import glob
import numpy as np
import cv2
from utils import preprocessing
from PIL import Image, ImageFont, ImageDraw
import os
from models import models
import copy
import resnet
import shutil


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

IMAGE_SIZE = 1024
fontsize = 130
upleft = (35, 25)
downright = (300, 150)
bgcolor = 'gray'
textcolor = 'white'

save_folder = 'digestpath2019_preds'
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.makedirs(save_folder)

args.tile_stride_w = args.tile_stride_w // 1


def get_foreground_mask(image):

    hsv = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2HSV)

    h = (230 / 360 >= hsv[..., 0] / 255) * (hsv[..., 0] / 255 >= 180 / 360)
    s = hsv[..., 1] / 255 >= 0.1
    v = (hsv[..., 2] / 255 >= 0.5) * (hsv[..., 2] / 255 <= 0.9)

    mask = h * s * v

    mask = cv2.morphologyEx(np.uint8(mask), cv2.MORPH_CLOSE, np.ones((25, 25)))

    return mask


def eval_model(model_folder, dataset_name):

    args.seg_features = 256 * 1

    _seg_and_cls = [0, 1, 2]
    _use_seg_ratio = [0.25, 0.3, 0.4, 0.5, 0.75, 1.0]
    #_use_seg_ratio = [0.0, 0.01, 0.025, 0.05, 0.075, 0.10]
    _use_seg_ratio = [0.0, 0.01, 0.025, 0.05, 0.075, 0.10]

    _models = {}

    for use_seg_ratio in _use_seg_ratio:
        for seg_and_cls in _seg_and_cls:
            model_path = f'{model_folder}/model_{dataset_name}_{seg_and_cls}_{use_seg_ratio}.pt'
            model = resnet.resnet18(pretrained=True, progress=False)
            model.fc = models.domain_classifier(args.seg_features * 4, args.num_classes)
            model.c1 = models.C1(args.num_classes, args.seg_features, (args.tile_h, args.tile_w)).cuda()
            state = torch.load(model_path)
            model.load_state_dict(state['state_dict'])
            model = model.cuda()
            model.eval()
            _models[f'{seg_and_cls}_{use_seg_ratio}'] = copy.deepcopy(model)

    image_aug = preprocessing.standard_augmentor(1)

    with torch.no_grad():

        image_paths = glob.glob('/home/osha/Desktop/ld3/validation_utils/data/segmentation/digestpath2019/val/*.png')
        image_paths = [p for p in image_paths if '_mask.png' not in p]

        for path in tqdm(image_paths, disable=0):

            image_id = os.path.basename(path).replace('.png', '')

            mask_jet = Image.open(path.replace('.png', '_mask.png'))

            mask_jet = np.array(mask_jet, dtype=np.uint8)

            mask_jet = cv2.erode(mask_jet, np.ones((10, 10)))

            #if len(np.unique(mask_jet)) < 2:
            #   continue

            image = Image.open(path)

            image_mask = get_foreground_mask(image)

            image = image_aug(image).unsqueeze(dim=0).cuda()

            im_colors = {}

            vote_count = (float(args.tile_w) / args.tile_stride_w) ** 2

            for use_seg_ratio in _use_seg_ratio:
                for seg_and_cls in _seg_and_cls:

                    t = f'{seg_and_cls}_{use_seg_ratio}'

                    pred_map = torch.zeros((args.num_classes, IMAGE_SIZE, IMAGE_SIZE))

                    for ypos in range(0, IMAGE_SIZE - 1 - args.tile_w, args.tile_stride_w):
                        for xpos in range(0, IMAGE_SIZE - 1 - args.tile_w, args.tile_stride_w):
                            patch = image[..., ypos:ypos + args.tile_w, xpos:xpos + args.tile_w]
                            pred_map[..., ypos:ypos + args.tile_w, xpos:xpos + args.tile_w] += \
                                torch.softmax(_models[t].c1(_models[t](patch.cuda()))[1].cpu(), dim=1).squeeze(0) / vote_count

                    xpos = IMAGE_SIZE - 1 - args.tile_w
                    for ypos in range(0, IMAGE_SIZE - 1 - args.tile_w, args.tile_stride_w):
                        patch = image[..., ypos:ypos + args.tile_w, xpos:xpos + args.tile_w]
                        pred_map[..., ypos:ypos + args.tile_w, xpos:xpos + args.tile_w] += \
                            torch.softmax(_models[t].c1(_models[t](patch.cuda()))[1].cpu(), dim=1).squeeze(0) / vote_count

                    ypos = IMAGE_SIZE - 1 - args.tile_w
                    for xpos in range(0, IMAGE_SIZE - 1 - args.tile_w, args.tile_stride_w):
                        patch = image[..., ypos:ypos + args.tile_w, xpos:xpos + args.tile_w]
                        pred_map[..., ypos:ypos + args.tile_w, xpos:xpos + args.tile_w] += \
                            torch.softmax(_models[t].c1(_models[t](patch.cuda()))[1].cpu(), dim=1).squeeze(0) / vote_count

                    pred = torch.argmax(pred_map, dim=0).squeeze(dim=0).cpu().numpy().astype(np.uint8)

                    pred = (image_mask * (mask_jet >= 0)) * pred

                    acc = np.mean(pred[mask_jet >= 0] == mask_jet[mask_jet >= 0])

                    if 1:  # acc > 0.0 and len(np.unique(mask_jet)) > 2:

                        im_color = np.uint8(255 / args.num_classes * (mask_jet >= 0) * pred)
                        im_color = cv2.applyColorMap(
                            im_color,
                            cv2.COLORMAP_JET)

                        im_color = Image.fromarray(im_color)

                        d = ImageDraw.Draw(im_color)
                        fpth = '/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf'
                        fnt = ImageFont.truetype(fpth, fontsize)
                        d.rectangle((upleft, downright), fill=bgcolor)
                        d.text(upleft, f'{acc:.2f}', font=fnt, fill=textcolor)

                        im_colors[t] = np.array(im_color)

                        # cv2.imwrite(f'{save_folder}/{image_id}_pred_jet_{acc:.2f}.jpg', im_color)

            images = np.concatenate([
                np.concatenate(([im_colors[f'{k}_{use_seg_ratio}'] for k in _seg_and_cls]), axis=0)
                for use_seg_ratio in _use_seg_ratio], axis=1)

            im_color = np.uint8(255 / args.num_classes * np.array(mask_jet))
            im_color = cv2.applyColorMap(
                im_color,
                cv2.COLORMAP_JET)

            empty_col = np.zeros((images.shape[0], im_color.shape[1], 3), dtype=np.uint8)
            empty_col[:im_color.shape[0], :im_color.shape[1], :] = im_color

            compare_images = np.concatenate((empty_col, images), axis=1)

            compare_images[
                (compare_images[..., 0] == 0) *
                (compare_images[..., 1] == 0) *
                (compare_images[..., 2] == 0)] = (128, 0, 0)

            cv2.imwrite(f'{save_folder}/{image_id}_mask_jet.jpg', compare_images)


if __name__ == "__main__":

    args.dataset_name = 'digestpath2019'

    eval_model(args.model_save_pth, args.dataset_name)
