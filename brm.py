import argparse
import cv2
import glob
import numpy as np
import os


def inpainting(image_path, mask_path, save_path, enlarge=False, type=0):
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    if enlarge:
        kernel = np.ones((3, 3), np.uint8)  # dilation
        mask = cv2.dilate(mask, kernel, iterations=2)

    # inpainting
    output = cv2.inpaint(img, mask, 3, type)
    cv2.imwrite(save_path, output)


def brm(image_path, mask_path, output_path):
    os.makedirs(output_path, exist_ok=False)
    vessel_list = glob.glob("{}/*.png".format(image_path))
    vessel_list.sort()
    mask_list = glob.glob("{}/*.png".format(mask_path))
    mask_list.sort()
    for vessel, mask in zip(vessel_list, mask_list):
        output_name = "{}/{}".format(output_path, vessel.split("/")[-1])
        inpainting(vessel, mask, output_name, enlarge=True, type=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=None,
                        help='path to the image folder')
    parser.add_argument('--mask', type=str, default=None,
                        help='path to the mask folder')
    parser.add_argument('--out', type=str, default=None,
                        help='path to the output folder')
    args = parser.parse_args()

    brm(args.image, args.mask, args.outp)
