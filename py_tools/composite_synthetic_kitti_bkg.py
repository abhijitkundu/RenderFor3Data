#!/usr/bin/python

"""
This script creates composite image from FG and BG image
"""
import argparse
import json
import os.path as osp
from os import makedirs
from glob import glob
import numpy as np
import cv2

from tqdm import tqdm

from RenderFor3Data.image_dataset import ImageDataset


def main():
    """Main Function"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fg_image_dir", required=True, type=str, help="Path to FG image directory")
    parser.add_argument("-b", "--bg_image_dir", required=True, type=str, help="Path to BG image directory")
    parser.add_argument("-o", "--out_image_dir", type=str, help="Path to output directory for Composite image ")
    args = parser.parse_args()

    assert osp.isdir(args.fg_image_dir), '"{}" does not exist or not a directory'.format(args.fg_image_dir)
    assert osp.isdir(args.bg_image_dir), '"{}" does not exist or not a directory'.format(args.bg_image_dir)

    pause_time = 0
    if args.out_image_dir:
        if not osp.isdir(args.out_image_dir):
            makedirs(args.out_image_dir)
        pause_time = 1

    fg_image_paths = glob(osp.join(args.fg_image_dir, '*.png'))
    assert fg_image_paths, 'No FG images found'

    for fg_image_path in fg_image_paths:
        fg_image = cv2.imread(fg_image_path, cv2.IMREAD_UNCHANGED)
        assert fg_image.ndim == 3, "Image needs to be a 3D tensor"
        assert fg_image.shape[2] == 4, "FG Image needs to be a 4 channel (with alpha channel)"

        image_name = osp.basename(fg_image_path)[:6] + '.png'
        bg_image_path = osp.join(args.bg_image_dir, image_name)
        assert osp.exists(bg_image_path), 'No BG image found at {}'.format(bg_image_path)

        bg_image = cv2.imread(bg_image_path, cv2.IMREAD_COLOR)
        assert bg_image.shape[:2] == fg_image.shape[:2], 'shape mismatch b/w FG and BG'

        alpha = fg_image[:, :, 3].astype(np.float) / 255
        alpha = alpha.reshape(alpha.shape + (1,))
        fg_image = fg_image[:, :, :3].astype(np.float) / 255
        bg_image = bg_image.astype(np.float) / 255

        composite_image = (alpha * fg_image + (1. - alpha) * bg_image) * 255
        composite_image = composite_image.astype(np.uint8)

        if args.out_image_dir:
            out_image_path = osp.join(args.out_image_dir, image_name)
            cv2.imwrite(out_image_path, composite_image)
            print("Saved composite_image at {}".format(out_image_path))

        cv2.namedWindow('composite_image', cv2.WINDOW_NORMAL)
        cv2.imshow('composite_image', composite_image)
        key = cv2.waitKey(pause_time)
        if key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
