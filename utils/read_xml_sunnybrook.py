'''
author: Tony Xu
'''

import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image
import os
import scipy.ndimage.morphology as morph


# finds xml files that have been annotated and padded properly
def findAnnotatedFiles(root_dir):
    all_files = []
    for path, subdirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('padded.session.xml'):
                all_files.append(os.path.join(path, file))

    return all_files


# fills the image with the outlines from the xml file
def fillImage(image, coordinates, color=255):
    coord_x = []
    coord_y = []
    coords = coordinates[0]

    # important because some labels are drawn OUTSIDE the image boundaries
    for coord in coords:
        if coord[0] > image.shape[1]:
            coord[0] = image.shape[1] - 1
        if coord[1] > image.shape[0]:
            coord[1] = image.shape[0] - 1
        coord_x.append(coord[0])
        coord_y.append(coord[1])

    # this is to reject very small cellularity rectangles
    if max(coord_x) - min(coord_x) > 100 and max(coord_y) - min(coord_y) > 100:  # 100 is a threshold value
        cv2.polylines(image, coordinates, True, color=color, thickness=8)

    return image


# parses the xml file label and tells you what class it belongs in
def class_dictionary(label):
    label = label.lower()
    label = label.replace(' ', '')

    if "cellularity" in label:
        class_out = 0
    elif label == "i" or "invasive" in label or "idc" in label or "ilc" in label:
        class_out = 3
    elif "dcis" in label:
        class_out = 2
    elif "benign" in label or 'udh' in label:
        class_out = 1
    elif "normal" in label:
        class_out = 0
    elif "tb" in label:
        class_out = 0
    else:
        class_out = 0

    if "no dcis" in label:
        if class_out == 2:
            class_out = 0

    return class_out


# this maps the colors to their correct class label
def mapToClass(label):
    label = label.lower()

    # red is 'benign', green is 'in situ' and blue is 'invasive, rest is black
    colors = {
        0: (0, 0, 0),  # normals, backgrounds, non-labels, tumor beds, etc.
        1: (255, 0, 0),  # benign, udh
        2: (0, 255, 0),  # DCIS
        3: (0, 0, 255),  # invasive, i, idc, ilc
    }

    # due to skipping class 0 in lines 115 to 119, the label will never actually be 0, but i keep it here
    # to maintain the correct indices for the 'colors' list
    color = colors[class_dictionary(label)]

    return color


# saves the mask image
def saveImage(image_size, coordinates, labels, sample=8):

    img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    # cases exist where every labelled item in the image is cellularity
    # meaning that there will be no labels after filtering them out
    if len(labels) == 0:
        img2 = img[::sample, ::sample, :]

    else:

        for i, (c, l) in enumerate(zip(coordinates, labels)):
            img1 = fillImage(img, [np.int32(np.stack(c))], color=mapToClass(l))
            img2 = img1[::sample, ::sample, :]

    return img2


# reads xml and returns the information extracted
def readXML(filename):
    tree = ET.parse(filename)

    root = tree.getroot()
    graphics = root[0][3].findall('graphic')

    labels = []
    coords = []

    # finds all info from xml file
    for g in graphics:
        description = g.get('description')

        g_coords = []

        # skips if label is zero, or if the type is not usable
        if not class_dictionary(description)\
                or g.get('type') == 'point' \
                or g.get('type') == 'ellipse'\
                or g.get('type') == 'text':
            continue

        vertices = g[2].findall('point')
        for vertex in vertices:
            g_coords.append(tuple(int(float(i)) for i in vertex.text.split(',')))

        labels += [description]
        coords += [g_coords]

    return coords, labels


# this processes and extracts the labels for sb (sunnybrook) WSIs
def getGT(xmlpath, scan, level):

    coords, labels = readXML(xmlpath)
    dims = scan.dimensions

    img_size = (*dims, 3)
    img_array = saveImage(img_size, coords, labels, sample=8).astype(np.bool)

    for i in range(3):
        # fill holes in polygon images
        img_array[:, :, i] = morph.binary_fill_holes(
            cv2.morphologyEx(
                img_array[:, :, i]/255,
                cv2.MORPH_CLOSE,
                kernel=np.ones((10, 10))
            )
        )

    gt = Image.fromarray(img_array.astype(np.uint8)*255).convert('RGB').resize(scan.level_dimensions[level])

    gt = np.asarray(gt)
    gt = np.concatenate((np.zeros((gt.shape[0], gt.shape[1], 1)), gt), axis=-1)
    gt = np.argmax(gt, axis=-1)

    return gt


# get only the tumor bed
def getTB(xmlpath, scan, level):
    coords, labels = readXML_TB(xmlpath)
    labels = ['benign' for l in labels]
    dims = scan.dimensions

    img_size = (*dims, 3)
    img_array = saveImage(img_size, coords, labels, sample=2).astype(np.bool)

    for i in range(3):
        # fill holes in polygon images
        img_array[:, :, i] = morph.binary_fill_holes(
            cv2.morphologyEx(
                img_array[:, :, i] / 255,
                cv2.MORPH_CLOSE,
                kernel=np.ones((10, 10))
            )
        )

    img_array = np.max(img_array > 0, -1)
    gt = Image.fromarray(img_array.astype(np.uint8) * 255).convert('RGB').resize(scan.level_dimensions[level])

    return gt


def readXML_TB(filename):
    tree = ET.parse(filename)

    root = tree.getroot()
    graphics = root[0][3].findall('graphic')

    labels = []
    coords = []

    # finds all info from xml file
    for g in graphics:
        description = g.get('description').lower().replace(' ', '')

        g_coords = []

        # skips if label is zero, or if the type is not usable
        if 'tb' not in description:
            continue

        vertices = g[2].findall('point')
        for vertex in vertices:
            g_coords.append(tuple(int(float(i)) for i in vertex.text.split(',')))

        labels += [description]
        coords += [g_coords]

    return coords, labels


# saves the mask image
def saveImage_TB(image_size, coordinates, labels, sample=2):
    img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

    # cases exist where every labelled item in the image is cellularity
    # meaning that there will be no labels after filtering them out
    if len(labels) == 0:
        img2 = img[::sample, ::sample, :]

    else:

        for i, (c, l) in enumerate(zip(coordinates, labels)):
            img1 = fillImage(img, [np.int32(np.stack(c))], color=mapToClass(l))
            img2 = img1[::sample, ::sample, :]

    return img2

