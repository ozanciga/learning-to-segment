# for paperrrrrr figure


from PIL import Image
import numpy as np
import cv2

ip = '/home/osha/Downloads/gleason2019 (copy)/train/slide005_core012_image.png'
mp = '/home/osha/Downloads/gleason2019 (copy)/train/slide005_core012_mask.png'

dx = 128
image = Image.open(ip)
mask = Image.open(mp)
mask = cv2.applyColorMap(255 - np.uint8(255 / 5 * np.array(mask)), cv2.COLORMAP_JET)
mask = np.array(image) * 0.35 + mask * 0.65
for k in range(6):
    mask[:, k::dx, :] = 0
    mask[k::dx, :, :] = 0
Image.fromarray(np.uint8(mask)).save('tile_extraction.png')
