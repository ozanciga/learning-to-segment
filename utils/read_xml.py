# -*- coding: utf-8 -*-
"""
ICIAR2018 - Grand Challenge on Breast Cancer Histology Images
https://iciar2018-challenge.grand-challenge.org/home/
"""

import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os
from PIL import Image
from skimage.morphology.convex_hull import convex_hull_image as chull


def findExtension(directory, extension='.xml'):
    files = []
    for file in os.listdir(directory):
        if file.endswith(extension):
            files += [file]
    files.sort()
    return files


def fillImage(image, coordinates, color=255):
    cv2.fillPoly(image, coordinates, color=color)
    return image


def readXML(filename):
    tree = ET.parse(filename)

    root = tree.getroot()
    regions = root[0][1].findall('Region')

    pixel_spacing = float(root.get('MicronsPerPixel'))

    labels = []
    coords = []
    length = []
    area = []

    for r in regions:
        area += [float(r.get('AreaMicrons'))]
        length += [float(r.get('LengthMicrons'))]
        try:
            label = r[0][0].get('Value')
        except:
            label = r.get('Text')
        if 'benign' in label.lower():
            label = 1
        elif 'in situ' in label.lower():
            label = 2
        elif 'invasive' in label.lower():
            label = 3

        labels += [label]
        vertices = r[1]
        coord = []
        for v in vertices:
            x = int(v.get('X'))
            y = int(v.get('Y'))
            coord += [[x, y]]

        coords += [coord]

    return coords, labels, length, area, pixel_spacing


def saveImage(image_size, coordinates, labels, sample):
    # red is 'benign', green is 'in situ' and blue is 'invasive'
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]

    img = np.zeros(image_size, dtype=np.uint8)

    for c, l in zip(coordinates, labels):
        img1 = fillImage(img, [np.int32(np.stack(c))], color=colors[l])
        img2 = img1[::sample, ::sample, :]
    return img2


def getGT(xmlpath, scan, level):
    dims = scan.dimensions
    img_size = (dims[1], dims[0], 3)

    coords, labels, length, area, pixel_spacing = readXML(xmlpath)
    gt = saveImage(img_size, coords, labels, sample=4 ** level)

    gt = Image.fromarray(gt).convert('RGB').resize(scan.level_dimensions[level])
    gt = np.asarray(gt)
    gt = np.concatenate((np.zeros((gt.shape[0], gt.shape[1], 1)), gt), axis=-1)
    gt = np.argmax(gt, axis=-1)

    return gt


def getTB(gt, scan, level):
    '''
    given gt mask
    with labels
    1 = benign, 2 = dcis, 3 = inv
    get tb, which only includes malginant
    (no benign) cancer's convex hull
    '''
    gt[gt == 1] = 0
    tb = chull((gt > 0).astype(np.uint8))
    return Image.fromarray(tb.astype(np.uint8) * 255).convert('RGB').resize(scan.level_dimensions[level])
